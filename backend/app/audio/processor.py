import numpy as np

from app.audio.metrics import (
    calculate_peak,
    calculate_rms,
    calculate_zero_crossing_rate,
    estimate_speech_activity,
)


class AudioProcessor:
    """
    Processes incoming PCM audio chunks.

    Input:
        signed 16-bit PCM

    Output:
        audio metrics used by the real-time dashboard.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate

    def process(self, audio_bytes: bytes) -> dict:
        if not audio_bytes:
            return {
                "samples": 0,
                "duration_ms": 0,
                "rms": 0.0,
                "peak": 0.0,
                "zero_crossing_rate": 0.0,
                "speech_detected": False,
            }

        audio = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        ).astype(np.float32)

        # Normalize int16 PCM → [-1, 1]
        audio /= 32768.0

        rms = calculate_rms(audio)
        peak = calculate_peak(audio)
        zcr = calculate_zero_crossing_rate(audio)

        speech_detected = estimate_speech_activity(rms)

        duration_ms = (
            len(audio) / self.sample_rate
        ) * 1000

        return {
            "samples": len(audio),
            "duration_ms": round(duration_ms, 2),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "zero_crossing_rate": round(zcr, 6),
            "speech_detected": speech_detected,
        }