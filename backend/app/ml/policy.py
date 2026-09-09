"""
Policy — turning a risk score into something a system can actually do.

The handoff is emphatic on this point and it is worth restating: VoxShield is
meant to *prevent* impersonation, not to display a probability. A dashboard
that shows "87% fraud" and stops there has moved the decision onto whoever is
reading it, usually under time pressure, usually while someone is talking at
them urgently. That is precisely the condition social engineering exploits.

So a decision comes with actions attached, and the actions are graded.

The one rule that is not a placeholder
--------------------------------------
**Never block a high-value financial action on one AI voice score alone.**

Every action below is a verification step, a hold, or a human escalation.
None of them is "reject the transaction because the model said so". A false
positive here means a real customer is asked to confirm on a registered device;
it must never mean their legitimate payment is silently refused. That asymmetry
is deliberate and should survive any later tuning of the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.fusion import ALLOW, ESCALATE, VERIFY, WARN


@dataclass
class Action:
    """One thing to do, and why."""

    code: str
    description: str
    blocking: bool = False

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "blocking": self.blocking,
        }


# Ordered roughly by friction imposed on the caller.
MONITOR = Action("MONITOR", "Continue the call and record the risk event")
NOTIFY_AGENT = Action("NOTIFY_AGENT", "Flag the elevated risk to the handling agent")
INCREASE_MONITORING = Action(
    "INCREASE_MONITORING", "Raise the analysis rate for the rest of this call"
)
DEVICE_CONFIRMATION = Action(
    "DEVICE_CONFIRMATION", "Ask the caller to confirm on a registered device"
)
SECURITY_QUESTIONS = Action(
    "SECURITY_QUESTIONS", "Ask knowledge-based verification questions"
)
MFA = Action("MFA", "Require multi-factor authentication before proceeding")
CALLBACK = Action(
    "CALLBACK",
    "End the call and call back on the number already held on file",
    blocking=True,
)
TRANSACTION_HOLD = Action(
    "TRANSACTION_HOLD",
    "Place a hold on the transaction pending verification - do not reject it",
    blocking=True,
)
MANAGER_APPROVAL = Action(
    "MANAGER_APPROVAL", "Require a second authorised person to approve", blocking=True
)
ESCALATE_SECURITY = Action(
    "ESCALATE_SECURITY", "Raise an incident with the security team"
)


@dataclass
class PolicyDecision:
    decision: str
    actions: list[Action] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    requires_human: bool = False
    version: str = "v0-UNCALIBRATED"

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "actions": [a.to_dict() for a in self.actions],
            "rationale": self.rationale,
            "requires_human": self.requires_human,
            "version": self.version,
            "calibrated": "UNCALIBRATED" not in self.version,
        }


class PolicyEngine:
    """
    Map a fused risk decision plus its context to concrete actions.

    ``high_value_threshold`` is the amount above which a transaction gets the
    heavier treatment regardless of what the audio said. It is a business
    parameter, not a model one.
    """

    def __init__(
        self,
        high_value_threshold: float = 10_000.0,
        version: str = "v0-UNCALIBRATED",
    ) -> None:
        self.high_value_threshold = high_value_threshold
        self.version = version

    def decide(
        self,
        risk_decision: str,
        risk_score: float,
        transaction_amount: float | None = None,
        identity_decision: str | None = None,
        calibrated_inputs: bool = False,
    ) -> PolicyDecision:
        actions: list[Action] = []
        rationale: list[str] = []

        if risk_decision == ALLOW:
            actions.append(MONITOR)
            rationale.append(f"Risk {risk_score:.0f}/100 is within normal range")

        elif risk_decision == WARN:
            actions.extend([MONITOR, NOTIFY_AGENT, INCREASE_MONITORING])
            rationale.append(f"Risk {risk_score:.0f}/100 is mildly elevated")

        elif risk_decision == VERIFY:
            actions.extend(
                [NOTIFY_AGENT, INCREASE_MONITORING, DEVICE_CONFIRMATION, MFA]
            )
            rationale.append(
                f"Risk {risk_score:.0f}/100 warrants a second factor before "
                f"acting on the caller's request"
            )

        elif risk_decision == ESCALATE:
            actions.extend(
                [NOTIFY_AGENT, MFA, CALLBACK, ESCALATE_SECURITY]
            )
            rationale.append(
                f"Risk {risk_score:.0f}/100 requires verification through a "
                f"channel the caller does not control"
            )

        # A confirmed identity mismatch is not a matter of degree. Someone is
        # claiming to be a person they are not.
        if identity_decision == "NO_MATCH":
            for action in (CALLBACK, ESCALATE_SECURITY):
                if action not in actions:
                    actions.append(action)
            rationale.append(
                "Speaker verification returned NO_MATCH - the caller is not "
                "the person they claim to be"
            )

        # Money raises the stakes independently of the audio.
        if transaction_amount is not None and transaction_amount >= self.high_value_threshold:
            if risk_decision != ALLOW:
                for action in (TRANSACTION_HOLD, MANAGER_APPROVAL):
                    if action not in actions:
                        actions.append(action)
                rationale.append(
                    f"High-value request ({transaction_amount:,.0f}) combined "
                    f"with elevated risk"
                )
            else:
                if DEVICE_CONFIRMATION not in actions:
                    actions.append(DEVICE_CONFIRMATION)
                rationale.append(
                    f"High-value request ({transaction_amount:,.0f}) - confirm "
                    f"on a registered device even at low risk"
                )

        if not calibrated_inputs and risk_decision != ALLOW:
            rationale.append(
                "Risk weights are uncalibrated placeholders - treat this as "
                "advisory and do not automate an irreversible action on it"
            )

        return PolicyDecision(
            decision=risk_decision,
            actions=actions,
            rationale=rationale,
            requires_human=any(a.blocking for a in actions),
            version=self.version,
        )
