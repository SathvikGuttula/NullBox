"""
Fusion — turning several weak signals into one decision a human can act on.

This is the layer the whole project is named for. Anti-spoof alone answers
"is this synthetic?"; speaker verification alone answers "is this the right
person?". Neither question is the one a bank actually asks, which is
"should this transaction proceed?"

The three cases that matter, and why one branch is never enough:

    genuine caller        spoof LOW    identity MATCH      -> ALLOW
    cloned voice          spoof HIGH   identity MATCH?     -> caught by anti-spoof
    human impersonator    spoof LOW    identity MISMATCH   -> caught ONLY by identity

The third row is the reason speaker verification exists. An impersonator's
speech is genuinely human, so the anti-spoof branch correctly reports LOW and
would wave them straight through.

On the weights
--------------
The contribution weights below are **engineering placeholders**, not measured
quantities. The handoff is explicit that its illustrative percentages must not
be hard-coded as established science, and they are not: they live in one
dataclass, carry a version string that travels with every decision, and are
meant to be fitted once there is labelled fraud data to fit them on.

What is *not* arbitrary is the structure: contributions are additive, every one
is reported with its own magnitude, and the output carries reason codes. A
score of 87 with no explanation is unusable in an investigation; "+34 synthetic
speech, +28 identity mismatch, +12 urgency language" is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.risk import HIGH, LOW, SUSPICIOUS, UNKNOWN


ALLOW = "ALLOW"
WARN = "WARN"
VERIFY = "SECONDARY_VERIFICATION"
ESCALATE = "HIGH_RISK_WORKFLOW"

# Not a risk level - the absence of one. Every branch was unavailable, so
# there is nothing to be confident about in either direction. This used to
# return ALLOW with a score of 0, which reads as "checked and cleared" and is
# exactly backwards: four seconds of hold music would clear a call.
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass
class FusionWeights:
    """
    How much each branch can contribute to the 0-100 risk score.

    Placeholders. Version them, calibrate them, and never present them as
    derived from data until they are.
    """

    synthetic_speech: float = 35.0
    identity_mismatch: float = 30.0
    audio_forensics: float = 10.0
    prosody: float = 10.0
    context: float = 10.0
    nlp: float = 5.0

    version: str = "v0-UNCALIBRATED"

    @property
    def calibrated(self) -> bool:
        return "UNCALIBRATED" not in self.version

    @property
    def total(self) -> float:
        return (
            self.synthetic_speech
            + self.identity_mismatch
            + self.audio_forensics
            + self.prosody
            + self.context
            + self.nlp
        )


@dataclass
class PolicyThresholds:
    """Risk score bands. Also placeholders — see the handoff's §55."""

    warn: float = 40.0
    verify: float = 60.0
    escalate: float = 80.0
    version: str = "v0-UNCALIBRATED"

    def decide(self, risk: float) -> str:
        if risk >= self.escalate:
            return ESCALATE
        if risk >= self.verify:
            return VERIFY
        if risk >= self.warn:
            return WARN
        return ALLOW


@dataclass
class Contribution:
    """One signal's push on the risk score, with its own explanation."""

    source: str
    points: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "points": round(self.points, 1),
            "detail": self.detail,
        }


@dataclass
class FusedRisk:
    risk_score: float
    risk_level: str
    decision: str
    contributions: list[Contribution] = field(default_factory=list)
    available_signals: list[str] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)
    weights_version: str = "v0-UNCALIBRATED"
    policy_version: str = "v0-UNCALIBRATED"

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "contributions": [c.to_dict() for c in self.contributions],
            "reasons": [c.detail for c in self.contributions],
            "available_signals": self.available_signals,
            "missing_signals": self.missing_signals,
            "weights_version": self.weights_version,
            "policy_version": self.policy_version,
            "calibrated": (
                "UNCALIBRATED" not in self.weights_version
                and "UNCALIBRATED" not in self.policy_version
            ),
        }


class RiskFusion:
    """
    Combine whatever branches are available into one risk score.

    Missing branches are *renormalised over*, not treated as zero risk. With
    only anti-spoof wired up, a spoof probability of 1.0 must still be able to
    reach a high score — otherwise the system silently caps its own maximum
    risk at 35/100 and never escalates anything. The trade is that the score's
    meaning shifts as branches come online, which is why ``available_signals``
    and ``missing_signals`` are part of the output rather than a footnote.
    """

    def __init__(
        self,
        weights: FusionWeights | None = None,
        policy: PolicyThresholds | None = None,
    ) -> None:
        self.weights = weights or FusionWeights()
        self.policy = policy or PolicyThresholds()

    def fuse(
        self,
        spoof_probability: float | None = None,
        speaker_similarity: float | None = None,
        speaker_decision: str | None = None,
        forensics_score: float | None = None,
        prosody_score: float | None = None,
        context_score: float | None = None,
        nlp_score: float | None = None,
    ) -> FusedRisk:
        """
        Each argument is a risk in [0, 1], except ``speaker_similarity`` which
        is a *similarity* and therefore inverted before use.
        """

        contributions: list[Contribution] = []
        available: list[str] = []
        missing: list[str] = []
        weight_total = 0.0
        weighted_risk = 0.0

        def add(name: str, risk: float | None, weight: float, describe) -> None:
            nonlocal weight_total, weighted_risk

            if risk is None:
                missing.append(name)
                return

            risk = float(min(max(risk, 0.0), 1.0))
            available.append(name)
            weight_total += weight
            weighted_risk += risk * weight

            points = risk * weight
            if points >= 1.0:
                contributions.append(Contribution(name, points, describe(risk)))

        add(
            "synthetic_speech",
            spoof_probability,
            self.weights.synthetic_speech,
            lambda r: (
                f"AI-generated speech artifacts detected ({r * 100:.0f}% confidence)"
                if r >= 0.6
                else f"Weak synthetic-speech indicators ({r * 100:.0f}%)"
            ),
        )

        # Similarity -> risk. A perfect match is zero identity risk; no match is
        # full identity risk. Clamped because cosine can be negative.
        identity_risk = None
        if speaker_similarity is not None:
            identity_risk = 1.0 - max(0.0, min(1.0, speaker_similarity))

        def describe_identity(risk: float) -> str:
            similarity = 1.0 - risk
            if risk >= 0.5:
                return f"Voice does not match the claimed identity (similarity {similarity:.2f})"
            if risk >= 0.25:
                return f"Voice is only a partial match for the claimed identity (similarity {similarity:.2f})"
            return f"Voice matches the claimed identity (similarity {similarity:.2f})"

        add(
            "identity_mismatch",
            identity_risk,
            self.weights.identity_mismatch,
            describe_identity,
        )

        add("audio_forensics", forensics_score, self.weights.audio_forensics,
            lambda r: f"Spectral/phase anomalies present ({r * 100:.0f}%)")
        add("prosody", prosody_score, self.weights.prosody,
            lambda r: f"Speaking style inconsistent with the enrolled speaker ({r * 100:.0f}%)")
        add("context", context_score, self.weights.context,
            lambda r: f"Call context is unusual ({r * 100:.0f}%)")
        add("nlp", nlp_score, self.weights.nlp,
            lambda r: f"Social-engineering language detected ({r * 100:.0f}%)")

        if weight_total == 0:
            return FusedRisk(
                risk_score=0.0,
                risk_level=UNKNOWN,
                decision=INSUFFICIENT,
                contributions=[
                    Contribution(
                        "none",
                        0.0,
                        "No signal could be assessed - this call has NOT been "
                        "cleared, it has not been checked",
                    )
                ],
                available_signals=[],
                missing_signals=missing,
                weights_version=self.weights.version,
                policy_version=self.policy.version,
            )

        # Renormalise over the branches that actually reported.
        risk_score = (weighted_risk / weight_total) * 100.0

        # Rescale the reported contributions to the same basis, so the points
        # shown to an operator sum to the score they are looking at.
        scale = 100.0 / weight_total
        for contribution in contributions:
            contribution.points *= scale

        contributions.sort(key=lambda c: -c.points)

        # An explicit identity mismatch is decisive on its own. A confirmed
        # wrong speaker is the impersonation case the whole system exists for,
        # and it must not be averaged away by a reassuring anti-spoof score.
        if speaker_decision == "NO_MATCH":
            risk_score = max(risk_score, self.policy.verify)
            contributions.insert(
                0,
                Contribution(
                    "identity_override",
                    0.0,
                    "Speaker verification returned NO_MATCH - escalated "
                    "regardless of other signals",
                ),
            )

        risk_score = float(min(max(risk_score, 0.0), 100.0))

        return FusedRisk(
            risk_score=risk_score,
            risk_level=self._level(risk_score),
            decision=self.policy.decide(risk_score),
            contributions=contributions,
            available_signals=available,
            missing_signals=missing,
            weights_version=self.weights.version,
            policy_version=self.policy.version,
        )

    @staticmethod
    def _level(risk: float) -> str:
        if risk >= 85:
            return HIGH
        if risk >= 60:
            return SUSPICIOUS
        return LOW
