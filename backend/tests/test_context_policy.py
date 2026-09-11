"""
Contextual fraud signals and the policy engine.

The behaviour that matters most here is what happens to *unknown* signals, and
the guarantee that no action ever silently rejects a customer's transaction.
"""

import pytest

from app.ml.context import CallContext, ContextAnalyzer, ContextWeights
from app.ml.fusion import ALLOW, ESCALATE, VERIFY, WARN
from app.ml.policy import PolicyEngine


# ---------------------------------------------------------------------------
# unknown is not innocent
# ---------------------------------------------------------------------------


def test_no_context_at_all_scores_zero_and_says_so():
    assessment = ContextAnalyzer().assess(CallContext())

    assert assessment.risk == 0.0
    assert assessment.signals_used == []
    assert "No call context was provided" in assessment.reasons


def test_unknown_signals_are_listed_not_scored():
    """
    "We did not check whether the device was known" and "the device was known"
    are very different statements and must not produce the same number.
    """

    assessment = ContextAnalyzer().assess(CallContext(caller_known=False))

    assert "caller_known" in assessment.signals_used
    assert "device_known" in assessment.missing
    assert "beneficiary_known" in assessment.missing


def test_risk_is_a_fraction_of_available_not_all_conceivable_risk():
    """
    A call where only two signals were collected and both look bad must read as
    high context risk. Dividing by the full weight of every possible signal
    would dilute it to nearly nothing and hide the thing an operator needs.
    """

    assessment = ContextAnalyzer().assess(
        CallContext(caller_known=False, device_known=False)
    )

    assert assessment.risk == pytest.approx(1.0)
    assert len(assessment.signals_used) == 2


def test_all_clear_signals_score_zero():
    assessment = ContextAnalyzer().assess(
        CallContext(
            caller_known=True,
            trusted_contact=True,
            device_known=True,
            recent_password_reset=False,
            failed_authentication_count=0,
            beneficiary_known=True,
        )
    )

    assert assessment.risk == 0.0
    assert assessment.reasons == []


# ---------------------------------------------------------------------------
# individual signals
# ---------------------------------------------------------------------------


def test_country_needs_an_expectation_to_be_meaningful():
    """"Called from Ireland" carries no information on its own."""

    analyzer = ContextAnalyzer()

    alone = analyzer.assess(CallContext(caller_country="IE"))
    assert "caller_country" in alone.missing

    compared = analyzer.assess(
        CallContext(caller_country="IE", expected_country="IN")
    )
    assert "caller_country" in compared.signals_used
    assert compared.risk > 0


def test_matching_country_is_not_risky():
    assessment = ContextAnalyzer().assess(
        CallContext(caller_country="IN", expected_country="in")
    )

    assert assessment.risk == 0.0


def test_amount_is_scored_relative_to_the_account():
    """50,000 is unremarkable for some accounts and extraordinary for others."""

    analyzer = ContextAnalyzer()

    normal = analyzer.assess(
        CallContext(transaction_amount=1000, typical_transaction_amount=900)
    )
    large = analyzer.assess(
        CallContext(transaction_amount=50_000, typical_transaction_amount=1000)
    )

    assert normal.risk == 0.0
    assert large.risk == pytest.approx(1.0)


def test_amount_without_a_baseline_is_unknown():
    assessment = ContextAnalyzer().assess(CallContext(transaction_amount=50_000))

    assert "transaction_amount" in assessment.missing


def test_odd_hour_from_either_signal():
    analyzer = ContextAnalyzer()

    assert analyzer.assess(CallContext(hour_of_day=3)).risk > 0
    assert analyzer.assess(CallContext(hour_of_day=14)).risk == 0.0
    assert analyzer.assess(CallContext(outside_business_hours=True)).risk > 0


def test_brand_new_beneficiary_is_flagged():
    analyzer = ContextAnalyzer()

    assert analyzer.assess(CallContext(beneficiary_age_days=0)).risk > 0
    assert analyzer.assess(CallContext(beneficiary_age_days=400)).risk == 0.0


def test_a_full_fraud_pattern_scores_high():
    """The classic CEO-fraud shape."""

    assessment = ContextAnalyzer().assess(
        CallContext(
            caller_known=False,
            trusted_contact=False,
            device_known=False,
            hour_of_day=23,
            transaction_amount=250_000,
            typical_transaction_amount=2_000,
            beneficiary_known=False,
            beneficiary_age_days=0,
        )
    )

    assert assessment.risk > 0.9
    assert len(assessment.reasons) >= 6


def test_weights_are_flagged_uncalibrated():
    assert not ContextWeights().calibrated
    assert ContextAnalyzer().assess(CallContext()).to_dict()["calibrated"] is False


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def test_low_risk_only_monitors():
    decision = PolicyEngine().decide(ALLOW, 8.0)

    assert [a.code for a in decision.actions] == ["MONITOR"]
    assert not decision.requires_human


def test_actions_escalate_with_risk():
    engine = PolicyEngine()

    counts = [
        len(engine.decide(level, score).actions)
        for level, score in [(ALLOW, 10), (WARN, 45), (VERIFY, 65), (ESCALATE, 90)]
    ]

    assert counts == sorted(counts), "friction must increase with risk"


def test_no_action_ever_rejects_a_transaction():
    """
    The rule that is not a placeholder: never block a high-value financial
    action on one AI voice score. A false positive must cost a customer a
    verification step, never a silently refused legitimate payment.
    """

    engine = PolicyEngine()

    # Structural, not textual. Every action the engine can emit must be a
    # verification step, a reversible hold, or a human escalation - matching on
    # prose would break the moment a description is reworded, and TRANSACTION
    # _HOLD's own text contains the word "reject" precisely because it says it
    # is not one.
    PERMITTED = {
        "MONITOR", "NOTIFY_AGENT", "INCREASE_MONITORING",
        "DEVICE_CONFIRMATION", "SECURITY_QUESTIONS", "MFA",
        "CALLBACK", "TRANSACTION_HOLD", "MANAGER_APPROVAL",
        "ESCALATE_SECURITY",
    }

    for level, score in [(ALLOW, 5), (WARN, 45), (VERIFY, 70), (ESCALATE, 95)]:
        decision = engine.decide(
            level, score, transaction_amount=1_000_000, identity_decision="NO_MATCH"
        )

        codes = {a.code for a in decision.actions}

        assert codes <= PERMITTED, (
            f"unrecognised action(s) {codes - PERMITTED} - every action must be "
            f"a verification, a hold or an escalation, never a refusal"
        )

        # The heaviest action available is a hold pending verification, and it
        # states explicitly that it is not a rejection.
        if "TRANSACTION_HOLD" in codes:
            hold = next(a for a in decision.actions if a.code == "TRANSACTION_HOLD")
            assert "do not reject" in hold.description.lower()


def test_identity_mismatch_forces_an_out_of_band_check():
    """
    A caller who is not who they claim to be must be verified through a channel
    they do not control - a callback to the number on file, not a question they
    can simply answer.
    """

    decision = PolicyEngine().decide(VERIFY, 62.0, identity_decision="NO_MATCH")

    codes = [a.code for a in decision.actions]
    assert "CALLBACK" in codes
    assert "ESCALATE_SECURITY" in codes
    assert decision.requires_human


def test_high_value_raises_friction_even_at_low_risk():
    decision = PolicyEngine().decide(ALLOW, 10.0, transaction_amount=500_000)

    assert "DEVICE_CONFIRMATION" in [a.code for a in decision.actions]


def test_high_value_plus_risk_holds_and_needs_a_second_person():
    decision = PolicyEngine().decide(VERIFY, 70.0, transaction_amount=500_000)

    codes = [a.code for a in decision.actions]
    assert "TRANSACTION_HOLD" in codes
    assert "MANAGER_APPROVAL" in codes
    assert decision.requires_human


def test_uncalibrated_inputs_are_declared_in_the_rationale():
    decision = PolicyEngine().decide(ESCALATE, 90.0, calibrated_inputs=False)

    assert any("uncalibrated" in r.lower() for r in decision.rationale)
    assert any("irreversible" in r.lower() for r in decision.rationale)


def test_decision_serialises():
    data = PolicyEngine().decide(VERIFY, 65.0).to_dict()

    assert data["decision"] == VERIFY
    assert data["actions"]
    assert "requires_human" in data
    assert data["calibrated"] is False


# ---------------------------------------------------------------------------
# untrusted input
# ---------------------------------------------------------------------------


def test_wrong_types_are_refused_at_the_boundary():
    """
    A dataclass does not check types, so CallContext(transaction_amount="lots")
    used to construct happily and then fail four layers away in the policy
    engine - comparing a string to a float, surfacing as a 500 with an empty
    body. The place to catch that is where the untrusted JSON arrives.
    """

    with pytest.raises(ValueError, match="transaction_amount must be int or float"):
        CallContext.from_dict({"transaction_amount": "lots"})


def test_a_boolean_is_not_a_count():
    """``bool`` is a subclass of ``int``, so an unguarded isinstance check lets
    ``True`` through as a failed-authentication count of 1."""

    with pytest.raises(ValueError, match="got a boolean"):
        CallContext.from_dict({"failed_authentication_count": True})


@pytest.mark.parametrize(
    "fields",
    [
        {"hour_of_day": 99},
        {"hour_of_day": -1},
        {"transaction_amount": -5},
        {"beneficiary_age_days": -1},
        {"failed_authentication_count": -3},
    ],
)
def test_out_of_range_values_are_refused(fields):
    with pytest.raises(ValueError, match="must be between"):
        CallContext.from_dict(fields)


def test_unknown_fields_are_refused_not_ignored():
    """
    Silently dropping an unrecognised key means a caller who sends
    ``caller_is_known`` gets a 200 and reasonably assumes the signal counted.
    """

    with pytest.raises(ValueError, match="unknown context fields"):
        CallContext.from_dict({"caller_is_known": False})


def test_a_non_object_is_refused():
    with pytest.raises(ValueError, match="must be a JSON object"):
        CallContext.from_dict([1, 2, 3])


def test_explicit_nulls_are_treated_as_not_collected():
    context = CallContext.from_dict({"caller_known": None, "hour_of_day": 14})

    assert context.caller_known is None
    assert context.hour_of_day == 14


def test_a_valid_payload_round_trips():
    context = CallContext.from_dict(
        {
            "caller_known": False,
            "hour_of_day": 23,
            "transaction_amount": 250000,
            "typical_transaction_amount": 2000,
            "caller_country": "IE",
        }
    )

    assessment = ContextAnalyzer().assess(context)

    assert context.transaction_amount == 250000
    assert assessment.risk > 0.5


def test_an_empty_object_is_valid_and_scores_nothing():
    assessment = ContextAnalyzer().assess(CallContext.from_dict({}))

    assert assessment.risk == 0.0
    assert assessment.signals_used == []
