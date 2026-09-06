import numpy as np


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate root mean square amplitude.
    """

    if audio.size == 0:
        return 0.0

    audio = audio.astype(np.float32)

    return float(np.sqrt(np.mean(np.square(audio))))


def calculate_peak(audio: np.ndarray) -> float:
    """
    Calculate peak absolute amplitude.
    """

    if audio.size == 0:
        return 0.0

    audio = audio.astype(np.float32)

    return float(np.max(np.abs(audio)))


def calculate_zero_crossing_rate(audio: np.ndarray) -> float:
    """
    Calculate zero-crossing rate.
    """

    if audio.size < 2:
        return 0.0

    audio = audio.astype(np.float32)

    crossings = np.sum(
        np.abs(np.diff(np.sign(audio))) > 0
    )

    return float(crossings / (audio.size - 1))


def estimate_speech_activity(
    rms: float,
    threshold: float = 0.01,
) -> bool:
    """
    Very simple energy-based voice activity estimate.

    This is intentionally a baseline.
    A proper VAD will be introduced later.
    """

    return rms > threshold