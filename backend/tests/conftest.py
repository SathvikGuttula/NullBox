"""
Shared pytest configuration.

The previous file was named ``confest.py``, so pytest never loaded it and none
of its fixtures existed. That also meant ``@pytest.mark.anyio`` had no backend
fixture, and the async API tests errored on collection rather than running.
"""

import sys

import numpy as np
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """
    Run every ``@pytest.mark.anyio`` test on asyncio only.

    Without this fixture anyio's pytest plugin has no backend to parametrise
    over and the async tests error out.
    """

    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _windows_event_loop_policy():
    """
    Use the selector event loop on Windows.

    The proactor loop that Python 3.8+ defaults to on Windows emits spurious
    "Event loop is closed" errors when asyncpg and httpx teardown interleave.
    """

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    yield


@pytest.fixture
def tone():
    """A 1-second 220 Hz tone at 16 kHz - deterministic test audio."""

    def _make(seconds: float = 1.0, frequency: float = 220.0, sample_rate: int = 16000):
        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    return _make


class StubDetector:
    """
    A detector that returns a scripted probability without loading a model.

    The streaming engine's job is buffering, smoothing and risk aggregation;
    testing that does not require 360 MB of wav2vec2 weights, and pulling them
    in would make the unit suite depend on a network round trip.
    """

    def __init__(self, probability: float = 0.5):
        self.probability = probability
        self.calls = 0

    def predict(self, audio, sample_rate=None) -> dict:
        self.calls += 1
        return {
            "synthetic_probability": self.probability,
            "bonafide_probability": 1.0 - self.probability,
            "confidence": max(self.probability, 1.0 - self.probability),
            "model_status": "stub",
        }


@pytest.fixture
def stub_detector():
    return StubDetector


@pytest.fixture
def speech_like():
    """
    Bursty synthetic audio that passes the speech-presence gate.

    A pure tone or a block of zeros is not usable as stand-in audio any more.
    ``app.audio.speech_check`` refuses to score steady signals, because the
    detector is a speech model and answers confidently and wrongly when handed
    noise or silence. What it looks for is the thing that makes speech speech:
    gaps. So this generates a harmonic carrier under a syllable envelope with
    real pauses in it, which is enough structure to clear the gate while
    staying completely deterministic.
    """

    def _make(seconds: float = 3.0, sample_rate: int = 16000, seed: int = 0):
        rng = np.random.default_rng(seed)
        count = int(seconds * sample_rate)
        t = np.arange(count) / sample_rate

        carrier = sum(
            np.sin(2 * np.pi * 140.0 * (k + 1) * t) / (k + 1) for k in range(12)
        )

        envelope = np.zeros(count)
        position = 0
        while position < count:
            on = int(rng.uniform(0.12, 0.25) * sample_rate)
            off = int(rng.uniform(0.08, 0.20) * sample_rate)
            envelope[position : position + on] = 1.0
            position += on + off

        # Soften the syllable edges so this is speech-shaped rather than a
        # click train, without filling in the gaps the gate measures.
        taper = np.hanning(max(int(0.02 * sample_rate), 3))
        envelope = np.convolve(envelope, taper / taper.sum(), mode="same")

        return (carrier * envelope * 0.3).astype(np.float32)

    return _make
