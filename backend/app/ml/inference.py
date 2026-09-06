from collections import deque

import numpy as np

from app.ml.calibration import (
    ProbabilityCalibrator,
)
from app.ml.detector import (
    VoiceSpoofDetector,
)


class StreamingInferenceEngine:

    def __init__(
        self,
        detector: VoiceSpoofDetector,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.0,
    ):

        self.detector = detector

        self.sample_rate = sample_rate

        self.window_size = int(
            sample_rate *
            window_seconds
        )

        self.hop_size = int(
            sample_rate *
            hop_seconds
        )

        self.buffer = deque()

        self.buffer_samples = 0

        self.samples_since_inference = 0

        self.calibrator = (
            ProbabilityCalibrator()
        )

        self.previous_score = 0.0

    def add_audio(
        self,
        audio: np.ndarray,
    ) -> list[dict]:

        if audio.size == 0:
            return []

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        self.buffer.append(audio)

        self.buffer_samples += len(
            audio
        )

        self.samples_since_inference += len(
            audio
        )

        results = []

        while (
            self.buffer_samples
            >= self.window_size
            and
            self.samples_since_inference
            >= self.hop_size
        ):

            window = self._get_window()

            result = self._infer(
                window
            )

            results.append(
                result
            )

            self.samples_since_inference -= (
                self.hop_size
            )

            self._discard(
                self.hop_size
            )

        return results

    def _get_window(
        self,
    ) -> np.ndarray:

        chunks = []

        remaining = (
            self.window_size
        )

        for chunk in self.buffer:

            if remaining <= 0:
                break

            take = min(
                len(chunk),
                remaining,
            )

            chunks.append(
                chunk[:take]
            )

            remaining -= take

        if not chunks:

            return np.array(
                [],
                dtype=np.float32,
            )

        return np.concatenate(
            chunks
        )

    def _discard(
        self,
        samples: int,
    ):

        remaining = samples

        while (
            self.buffer
            and remaining > 0
        ):

            chunk = self.buffer[0]

            if len(chunk) <= remaining:

                remaining -= len(chunk)

                self.buffer.popleft()

            else:

                self.buffer[0] = (
                    chunk[remaining:]
                )

                remaining = 0

        self.buffer_samples -= (
            samples - remaining
        )

    def _infer(
        self,
        audio: np.ndarray,
    ) -> dict:

        raw = self.detector.predict(
            audio,
            self.sample_rate,
        )

        probability = (
            raw[
                "synthetic_probability"
            ]
        )

        calibrated = (
            self.calibrator.calibrate(
                probability
            )
        )

        # Exponential temporal smoothing.
        smoothed = (
            0.35 * calibrated
            + 0.65 * self.previous_score
        )

        self.previous_score = smoothed

        return {
            "synthetic_probability": round(
                calibrated,
                4,
            ),
            "smoothed_probability": round(
                smoothed,
                4,
            ),
            "confidence": round(
                raw.get(
                    "confidence",
                    0.0,
                ),
                4,
            ),
            "model_status": raw.get(
                "model_status",
                "unknown",
            ),
            "window_seconds": round(
                len(audio)
                / self.sample_rate,
                3,
            ),
        }