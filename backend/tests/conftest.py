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
