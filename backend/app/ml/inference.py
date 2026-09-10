"""
Streaming inference: a stream of PCM chunks in, a stream of risk updates out.

The window/hop arrangement is the standard sliding analysis:

    0s----3s----6s----9s
    |-----|
       |-----|
          |-----|

with a 3 s window and a 1 s hop, so the risk score updates once a second while
each decision still sees three seconds of speech.

Note that the model was trained on 4 s windows. The detector tiles a 3 s window
up to 4 s before scoring rather than feeding the encoder a length it never saw
in training. Keeping the *live* window shorter than the *training* window is a
deliberate latency trade, not an oversight - but the two lengths must stay
coupled through DetectorConfig, not hardcoded in two places.
"""

from __future__ import annotations

import numpy as np

from app.audio.speech_check import assess_speech
from app.ml.calibration import ProbabilityCalibrator
from app.ml.config import DetectorConfig
from app.ml.detector import VoiceSpoofDetector
from app.ml.risk import RiskThresholds, TemporalRiskEngine


class StreamingInferenceEngine:
    """
    Buffers incoming audio and emits one result per hop once the window fills.

    ``add_audio`` returns a list because a single large chunk can complete
    several hops at once; callers should treat the last element as current and
    may log the rest.
    """

    def __init__(
        self,
        detector: VoiceSpoofDetector,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.0,
        smoothing_alpha: float | None = None,
        calibrator: ProbabilityCalibrator | None = None,
        risk_engine: TemporalRiskEngine | None = None,
        config: DetectorConfig | None = None,
    ) -> None:
        config = config or DetectorConfig()

        self.detector = detector
        self.sample_rate = sample_rate

        self.window_size = int(sample_rate * window_seconds)
        self.hop_size = max(int(sample_rate * hop_seconds), 1)

        # One contiguous array rather than a deque of chunks: the buffer holds
        # at most a few seconds, the concatenation is trivial next to a
        # transformer forward pass, and the sample accounting becomes obvious
        # instead of being spread across two counters that can drift apart.
        #
        # There is deliberately no "samples since last inference" counter. Each
        # emitted window consumes exactly one hop from the front of the buffer,
        # so the buffer length alone already encodes how much new audio has
        # arrived - a second counter can only disagree with it.
        self.buffer = np.zeros(0, dtype=np.float32)

        # Deliberately the identity by default. Calibration is applied once,
        # inside VoiceSpoofDetector.predict, so that every consumer sees the
        # same number; calibrating again here would square the map and push
        # every score to an extreme. This hook stays for tests that drive the
        # engine with a stub detector emitting deliberately raw scores.
        self.calibrator = calibrator or ProbabilityCalibrator()

        alpha = (
            smoothing_alpha
            if smoothing_alpha is not None
            else config.smoothing_alpha
        )

        # The config's thresholds used to be dead: nothing read them, and the
        # live risk levels came from RiskThresholds' own defaults of 60/85.
        # They are derived from the DET curve now (see DetectorConfig), and
        # the score they grade is a smoothed probability scaled to 0-100.
        self.risk = risk_engine or TemporalRiskEngine(
            smoothing_alpha=alpha,
            thresholds=RiskThresholds(
                suspicious=config.suspicious_threshold * 100.0,
                high=config.high_risk_threshold * 100.0,
                version="v1-derived-from-la-eval-det",
            ),
        )

        self.windows_scored = 0

    # -- state -------------------------------------------------------------

    def reset(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.windows_scored = 0
        self.risk.reset()

    @property
    def buffered_seconds(self) -> float:
        return self.buffer.size / self.sample_rate

    # -- ingest ------------------------------------------------------------

    def add_audio(self, audio: np.ndarray) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return []

        self.buffer = np.concatenate([self.buffer, audio])

        results: list[dict] = []

        while self.buffer.size >= self.window_size:
            results.append(self._infer(self.buffer[: self.window_size]))

            # Advance exactly one hop. A single large chunk therefore emits
            # every window it covers, rather than one and then stalling.
            self.buffer = self.buffer[self.hop_size :]

        return results

    # -- scoring -----------------------------------------------------------

    def _infer(self, window: np.ndarray) -> dict:
        # The energy VAD upstream is not enough on its own: measured, it
        # passes 31 of 31 white-noise chunks as speech, and the detector then
        # calls that noise synthetic with 0.997 confidence. Check the window
        # actually carries speech before asking a speech model about it.
        presence = assess_speech(window, self.sample_rate)

        if not presence.has_speech:
            self.windows_scored += 1

            # Report the state that stands, and do NOT feed the risk engine -
            # a window with no speech in it is not evidence of anything, and
            # smoothing it in as a zero would read as "confirmed genuine".
            return {
                "synthetic_probability": None,
                "raw_probability": None,
                "smoothed_probability": round(self.risk.smoothed, 4),
                "risk_score": round(self.risk.smoothed * 100.0, 2),
                "risk_level": self.risk.level,
                "reasons": [f"Not scored: {presence.reason}"],
                "confidence": 0.0,
                "model_status": "no_speech",
                "speech_check": presence.to_dict(),
                "window_seconds": round(window.size / self.sample_rate, 3),
                "window_index": self.windows_scored,
            }

        result = self.detector.predict(window, self.sample_rate)

        # The detector already calibrated this. self.calibrator is the identity
        # unless a test injected one, so this is a no-op in production.
        probability = float(result.get("synthetic_probability", 0.0))
        calibrated = self.calibrator.calibrate(probability)

        risk = self.risk.update(calibrated)

        self.windows_scored += 1

        return {
            "synthetic_probability": round(calibrated, 4),
            # What the checkpoint actually emitted, before calibration. A
            # stub detector in a test may not report one; fall back to the
            # value we were given rather than inventing a zero.
            "raw_probability": round(
                float(result.get("raw_probability", probability)), 4
            ),
            "smoothed_probability": risk["smoothed_probability"],
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "reasons": risk["reasons"],
            "confidence": round(float(result.get("confidence", 0.0)), 4),
            "model_status": result.get("model_status", "unknown"),
            "speech_check": presence.to_dict(),
            "window_seconds": round(window.size / self.sample_rate, 3),
            "window_index": self.windows_scored,
        }
