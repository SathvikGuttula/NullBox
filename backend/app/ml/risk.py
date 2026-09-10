"""
Temporal risk aggregation for a live call.

A single 3-second window is a weak piece of evidence. Codec glitches, a cough,
a moment of silence - any of these can spike one window's spoof probability.
Acting on a single window would make VoxShield fire constantly on genuine
calls, which is the failure mode that gets a fraud system switched off.

This module turns a stream of per-window probabilities into a stable risk
score with hysteresis, so the displayed risk level only changes when the
evidence actually persists.

On the thresholds
-----------------
60 and 85 are engineering starting points, not calibrated values. They exist
so the pipeline runs end to end. Replace them with thresholds derived from a
validation DET curve - ``app.ml.metrics.threshold_at_false_alarm`` gives you
the operating point for a chosen false-alarm budget - and record which
threshold version produced any decision you audit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


LOW = "LOW"
SUSPICIOUS = "SUSPICIOUS"
HIGH = "HIGH"


@dataclass
class RiskThresholds:
    """
    Levels plus the hysteresis margin.

    ``margin`` is how far the score must fall back below a boundary before the
    level downgrades. Without it, a score hovering at 59.9/60.1 flips the
    dashboard between LOW and SUSPICIOUS several times a second.
    """

    suspicious: float = 60.0
    high: float = 85.0
    margin: float = 5.0
    version: str = "v0-uncalibrated"

    def level_for(self, score: float, current: str) -> str:
        if current == HIGH:
            if score < self.high - self.margin:
                return SUSPICIOUS if score >= self.suspicious - self.margin else LOW
            return HIGH

        if current == SUSPICIOUS:
            if score >= self.high:
                return HIGH
            if score < self.suspicious - self.margin:
                return LOW
            return SUSPICIOUS

        if score >= self.high:
            return HIGH
        if score >= self.suspicious:
            return SUSPICIOUS
        return LOW


@dataclass
class RiskState:
    risk_score: float
    risk_level: str
    raw_probability: float
    smoothed_probability: float
    samples_seen: int
    windows_above_suspicious: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "risk_score": round(self.risk_score, 2),
            "risk_level": self.risk_level,
            "raw_probability": round(self.raw_probability, 4),
            "smoothed_probability": round(self.smoothed_probability, 4),
            "samples_seen": self.samples_seen,
            "windows_above_suspicious": self.windows_above_suspicious,
            "reasons": self.reasons,
        }


class TemporalRiskEngine:
    """
    Exponentially smoothed spoof probability -> 0-100 risk score with levels.

        engine = TemporalRiskEngine()
        state = engine.update(0.91)
        state["risk_level"]   # "LOW" on the first window - see below

    The first update deliberately does not jump to the raw value. The smoother
    starts at 0, so a single high window moves the score to alpha * p only.
    That is the intended behaviour: one window is not evidence. Call
    ``update`` repeatedly, as the streaming engine does, and the score
    converges within a few windows.
    """

    def __init__(
        self,
        window_size: int = 5,
        smoothing_alpha: float = 0.35,
        thresholds: RiskThresholds | None = None,
    ) -> None:
        self.history: deque = deque(maxlen=window_size)
        self.alpha = smoothing_alpha
        self.thresholds = thresholds or RiskThresholds()

        self.smoothed = 0.0
        self.level = LOW
        self.windows_above_suspicious = 0

    def reset(self) -> None:
        self.history.clear()
        self.smoothed = 0.0
        self.level = LOW
        self.windows_above_suspicious = 0

    def update(self, spoof_probability: float) -> dict:
        probability = min(max(float(spoof_probability), 0.0), 1.0)

        self.history.append(probability)

        self.smoothed = (
            self.alpha * probability + (1.0 - self.alpha) * self.smoothed
        )

        score = self.smoothed * 100.0

        if score >= self.thresholds.suspicious:
            self.windows_above_suspicious += 1
        else:
            self.windows_above_suspicious = 0

        self.level = self.thresholds.level_for(score, self.level)

        state = RiskState(
            risk_score=score,
            risk_level=self.level,
            raw_probability=probability,
            smoothed_probability=self.smoothed,
            samples_seen=len(self.history),
            windows_above_suspicious=self.windows_above_suspicious,
            reasons=self._reasons(probability, score),
        )

        return state.to_dict()

    def _reasons(self, probability: float, score: float) -> list[str]:
        """
        Human-readable contributors.

        The dashboard must never show a bare number. "87% fraud" is not
        actionable; "AI-generated speech artifacts, sustained over 6 windows"
        is. This is the audio-only subset - identity, prosody and context
        reasons get appended by the fusion layer once those branches exist.
        """

        reasons: list[str] = []

        # Graded against the same thresholds that set the level, not against a
        # second hardcoded pair. Those used to be 0.85 / 0.60 while the level
        # boundaries were configurable, so a window could be called "strong"
        # by one rule and LOW by the other. The thresholds are 0-100; the
        # probability is 0-1.
        strong = self.thresholds.high / 100.0
        possible = self.thresholds.suspicious / 100.0

        if probability >= strong:
            reasons.append("Strong synthetic-speech artifacts in this window")
        elif probability >= possible:
            reasons.append("Possible synthetic-speech artifacts in this window")

        if self.windows_above_suspicious >= 3:
            reasons.append(
                f"Elevated spoof score sustained over "
                f"{self.windows_above_suspicious} consecutive windows"
            )

        if len(self.history) >= 3:
            spread = max(self.history) - min(self.history)
            if spread > 0.5:
                reasons.append("Unstable authenticity score across the call")

        if not reasons and score < self.thresholds.suspicious:
            reasons.append("No synthetic-speech indicators detected")

        return reasons
