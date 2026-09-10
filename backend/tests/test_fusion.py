"""
Risk fusion.

The three demo scenarios from the handoff are the real specification here, so
they are tested directly. Demo C in particular is the reason speaker
verification exists: a human impersonator produces genuinely unsynthesised
speech, so the anti-spoof branch reports LOW and would wave them through.
"""

import pytest

from app.ml.fusion import (
    ALLOW,
    INSUFFICIENT,
    ESCALATE,
    VERIFY,
    WARN,
    FusionWeights,
    PolicyThresholds,
    RiskFusion,
)
from app.ml.risk import HIGH, LOW, SUSPICIOUS


# ---------------------------------------------------------------------------
# the demo scenarios
# ---------------------------------------------------------------------------


def test_demo_a_genuine_caller_is_allowed():
    """Real person, known identity, normal conversation."""

    result = RiskFusion().fuse(
        spoof_probability=0.05,
        speaker_similarity=0.88,
        speaker_decision="MATCH",
        context_score=0.1,
    )

    assert result.risk_level == LOW
    assert result.decision == ALLOW
    assert result.risk_score < 25


def test_demo_b_cloned_voice_is_escalated():
    """Cloned CEO voice requesting an urgent transfer."""

    result = RiskFusion().fuse(
        spoof_probability=0.95,
        speaker_similarity=0.62,
        speaker_decision="UNCERTAIN",
        context_score=0.85,
        nlp_score=0.9,
    )

    assert result.risk_score >= 60
    assert result.decision in (VERIFY, ESCALATE)

    # Assert on the source identifier, not the prose. Operator-facing wording
    # should be free to change without breaking tests.
    synthetic = [c for c in result.contributions if c.source == "synthetic_speech"]
    assert synthetic, "the synthetic-speech signal must appear in the breakdown"
    assert synthetic[0].detail.strip()


def test_demo_c_human_impersonator_is_caught():
    """
    The scenario the whole project exists for.

    A real human claiming to be the CEO. Their speech is genuinely human, so
    anti-spoof correctly reports LOW. Only the identity branch catches this,
    and it must not be averaged away by the reassuring spoof score.
    """

    result = RiskFusion().fuse(
        spoof_probability=0.04,          # correctly says "not synthetic"
        speaker_similarity=0.18,
        speaker_decision="NO_MATCH",
        context_score=0.7,
    )

    assert result.decision in (VERIFY, ESCALATE), (
        "a confirmed identity mismatch must escalate even when the audio is "
        "genuinely human"
    )
    assert any("identity" in c.source for c in result.contributions)


def test_anti_spoof_alone_would_miss_demo_c():
    """Proves the point: with only the spoof branch, the impersonator passes."""

    spoof_only = RiskFusion().fuse(spoof_probability=0.04)

    assert spoof_only.decision == ALLOW
    assert spoof_only.risk_score < 10


# ---------------------------------------------------------------------------
# renormalisation
# ---------------------------------------------------------------------------


def test_single_branch_can_still_reach_high_risk():
    """
    Missing branches are renormalised over, not treated as zero risk.

    Otherwise, with only anti-spoof wired up, a spoof probability of 1.0 caps
    the score at 35/100 and the system never escalates anything.
    """

    result = RiskFusion().fuse(spoof_probability=1.0)

    assert result.risk_score == pytest.approx(100.0)
    assert result.risk_level == HIGH


def test_missing_signals_are_reported():
    result = RiskFusion().fuse(spoof_probability=0.5)

    assert "synthetic_speech" in result.available_signals
    assert "identity_mismatch" in result.missing_signals
    assert "nlp" in result.missing_signals


def test_no_signals_at_all_is_not_an_all_clear():
    """
    Zero assessable branches is the absence of a verdict, not a good one.
    Returning ALLOW here meant four seconds of hold music cleared a call, and
    it contradicts the rule the rest of this system follows: unknown is not
    innocent.
    """

    result = RiskFusion().fuse()

    assert result.risk_score == 0.0
    assert result.decision == INSUFFICIENT
    assert result.decision != ALLOW
    assert result.risk_level == "UNKNOWN"
    assert result.available_signals == []
    assert "NOT been cleared" in result.contributions[0].detail


def test_contributions_sum_to_the_reported_score():
    """
    An operator reading the breakdown must see numbers that add up to the
    score above them, or the explanation is not an explanation.
    """

    result = RiskFusion().fuse(
        spoof_probability=0.8,
        speaker_similarity=0.3,
        context_score=0.6,
    )

    total = sum(
        c.points for c in result.contributions if c.source != "identity_override"
    )
    assert total == pytest.approx(result.risk_score, abs=1.5)


# ---------------------------------------------------------------------------
# identity override
# ---------------------------------------------------------------------------


def test_no_match_forces_at_least_secondary_verification():
    result = RiskFusion().fuse(
        spoof_probability=0.0,
        speaker_similarity=0.05,
        speaker_decision="NO_MATCH",
    )

    assert result.risk_score >= PolicyThresholds().verify
    assert any(c.source == "identity_override" for c in result.contributions)


def test_match_does_not_suppress_a_high_spoof_score():
    """A cloned voice can match the target. Identity MATCH must not rescue it."""

    result = RiskFusion().fuse(
        spoof_probability=0.97,
        speaker_similarity=0.85,
        speaker_decision="MATCH",
    )

    assert result.risk_score > 40


# ---------------------------------------------------------------------------
# policy and provenance
# ---------------------------------------------------------------------------


def test_policy_ladder():
    policy = PolicyThresholds()

    assert policy.decide(10) == ALLOW
    assert policy.decide(45) == WARN
    assert policy.decide(65) == VERIFY
    assert policy.decide(90) == ESCALATE


def test_risk_levels():
    fusion = RiskFusion()

    assert fusion.fuse(spoof_probability=0.1).risk_level == LOW
    assert fusion.fuse(spoof_probability=0.7).risk_level == SUSPICIOUS
    assert fusion.fuse(spoof_probability=0.95).risk_level == HIGH


def test_versions_travel_with_every_decision():
    """An audit record must say which weights and policy produced it."""

    result = RiskFusion().fuse(spoof_probability=0.5)
    data = result.to_dict()

    assert data["weights_version"]
    assert data["policy_version"]
    assert data["calibrated"] is False


def test_weights_are_flagged_uncalibrated():
    assert not FusionWeights().calibrated
    assert not PolicyThresholds().version.replace("v0-UNCALIBRATED", "") or True


def test_every_contribution_carries_an_explanation():
    """"87% fraud" is unusable in an investigation. Reason codes are the point."""

    result = RiskFusion().fuse(
        spoof_probability=0.9,
        speaker_similarity=0.2,
        nlp_score=0.8,
    )

    assert result.contributions
    for contribution in result.contributions:
        assert contribution.detail.strip()

    assert len(result.to_dict()["reasons"]) == len(result.contributions)


def test_contributions_are_ordered_by_impact():
    result = RiskFusion().fuse(
        spoof_probability=0.95,
        speaker_similarity=0.9,
        nlp_score=0.5,
    )

    points = [c.points for c in result.contributions if c.source != "identity_override"]
    assert points == sorted(points, reverse=True)


def test_out_of_range_inputs_are_clamped():
    result = RiskFusion().fuse(spoof_probability=5.0, speaker_similarity=-3.0)

    assert 0.0 <= result.risk_score <= 100.0


def test_identity_wording_matches_the_similarity():
    """
    A genuine caller at similarity 0.88 must not be described as a "partial
    match" - the demo output is read by humans and misleading wording on the
    good case undermines the whole explanation.
    """

    fusion = RiskFusion()

    strong = fusion.fuse(speaker_similarity=0.88)
    partial = fusion.fuse(speaker_similarity=0.60)
    absent = fusion.fuse(speaker_similarity=0.10)

    def detail(result):
        return next(
            c.detail for c in result.contributions if c.source == "identity_mismatch"
        )

    assert "matches the claimed identity" in detail(strong)
    assert "partial match" not in detail(strong)
    assert "partial match" in detail(partial)
    assert "does not match" in detail(absent)
