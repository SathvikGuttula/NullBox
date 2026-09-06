import numpy as np


class VoiceActivityDetector:
    """
    Lightweight streaming voice activity detector.

    This is a baseline implementation.

    It estimates the noise floor and determines
    whether the current audio frame contains speech.

    Later this interface can be backed by a neural
    VAD model without changing the rest of VoxShield.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        initial_noise_floor: float = 0.003,
        speech_multiplier: float = 2.5,
    ):
        self.sample_rate = sample_rate

        self.noise_floor = initial_noise_floor

        self.speech_multiplier = speech_multiplier

        self.initialized = False

        self.speech_frames = 0
        self.silence_frames = 0

    def _rms(self, audio: np.ndarray) -> float:

        if audio.size == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

    def process(
        self,
        audio: np.ndarray,
    ) -> dict:

        if audio.size == 0:

            return {
                "speech": False,
                "confidence": 0.0,
                "rms": 0.0,
                "noise_floor": self.noise_floor,
            }

        audio = audio.astype(
            np.float32,
            copy=False,
        )

        rms = self._rms(audio)

        # Slowly estimate background noise.
        if not self.initialized:

            self.noise_floor = max(
                rms,
                0.0005,
            )

            self.initialized = True

        else:

            # Only adapt aggressively while
            # the signal appears quiet.
            if rms < self.noise_floor * 2.0:

                self.noise_floor = (
                    0.95 * self.noise_floor
                    + 0.05 * rms
                )

        threshold = max(
            self.noise_floor
            * self.speech_multiplier,
            0.008,
        )

        if rms > threshold:

            self.speech_frames += 1
            self.silence_frames = 0

        else:

            self.silence_frames += 1
            self.speech_frames = 0

        speech = rms > threshold

        ratio = rms / max(
            threshold,
            1e-8,
        )

        confidence = min(
            max(
                (ratio - 1.0) / 3.0,
                0.0,
            ),
            1.0,
        )

        return {
            "speech": speech,
            "confidence": round(
                confidence,
                4,
            ),
            "rms": round(
                rms,
                6,
            ),
            "noise_floor": round(
                self.noise_floor,
                6,
            ),
        }