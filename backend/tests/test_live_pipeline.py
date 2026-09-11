"""
The live call path: rolling speaker verification and fused risk over a stream.

The encoder and detector are stubbed - what is under test is the streaming
behaviour, not model quality.
"""

import numpy as np
import pytest

from app.ml.speaker import (
    LiveSpeakerTracker,
    SpeakerRegistry,
    SpeakerThresholds,
    l2_normalize,
)


DIM = 192
SR = 16000


class StubEncoder:
    """Returns a fixed identity direction with a little per-call jitter."""

    model_name = "stub"

    def __init__(self, seed=1, spread=0.15):
        self.anchor = l2_normalize(np.random.default_rng(seed).normal(size=DIM))
        self.spread = spread
        self.calls = 0

    def embed(self, audio, sample_rate=None):
        self.calls += 1
        rng = np.random.default_rng(1000 + self.calls)
        return l2_normalize(
            self.anchor + self.spread * l2_normalize(rng.normal(size=DIM))
        )


def speech(seconds=1.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    signal = sum(
        (1.0 / (k + 1)) * np.sin(2 * np.pi * 140 * (k + 1) * t) for k in range(20)
    )
    return (signal / np.abs(signal).max() * 0.5).astype(np.float32)


def enrolled_registry(encoder, identity="ceo"):
    registry = SpeakerRegistry(SpeakerThresholds(match=0.5, no_match=0.3))
    registry.enroll(identity, [encoder.embed(speech()) for _ in range(5)])
    return registry


# ---------------------------------------------------------------------------
# the rolling tracker
# ---------------------------------------------------------------------------


def test_no_result_until_enough_speech():
    encoder = StubEncoder()
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), encoder, "ceo", SR, interval_seconds=3.0
    )

    assert tracker.add_speech(speech(1.0)) is None
    assert tracker.add_speech(speech(1.0)) is None
    assert tracker.to_dict() is None

    assert tracker.add_speech(speech(1.0)) is not None


def test_verification_improves_as_speech_accumulates():
    """
    The point of accumulating a centroid rather than averaging per-window
    similarities: more speech is a better estimate of the voice, so confidence
    should rise over a call rather than jitter.
    """

    encoder = StubEncoder(spread=0.5)
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), encoder, "ceo", SR, interval_seconds=3.0
    )

    similarities = []
    for _ in range(8):
        result = tracker.add_speech(speech(3.0))
        if result:
            similarities.append(result.similarity)

    assert len(similarities) >= 5
    # Later estimates should be at least as good as the very first one.
    assert similarities[-1] >= similarities[0] - 0.02


def test_speech_seconds_are_reported():
    encoder = StubEncoder()
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), encoder, "ceo", SR, interval_seconds=3.0
    )

    for _ in range(3):
        tracker.add_speech(speech(3.0))

    assert tracker.to_dict()["speech_seconds_analysed"] == pytest.approx(9.0)
    assert tracker.to_dict()["updates"] == 3


def test_embeddings_are_bounded():
    """
    A caller who hands the phone to someone else should eventually be scored as
    that someone else, not as an average of both for the rest of the call.
    """

    encoder = StubEncoder()
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), encoder, "ceo", SR,
        interval_seconds=1.0, max_embeddings=5,
    )

    for _ in range(20):
        tracker.add_speech(speech(1.0))

    assert len(tracker.embeddings) == 5


def test_a_different_speaker_does_not_match():
    encoder = StubEncoder(seed=1)
    registry = enrolled_registry(encoder)

    impostor = StubEncoder(seed=99)
    tracker = LiveSpeakerTracker(registry, impostor, "ceo", SR, interval_seconds=3.0)

    for _ in range(3):
        result = tracker.add_speech(speech(3.0))

    assert result.decision == "NO_MATCH"


def test_encoder_failure_does_not_break_the_call():
    """A broken encoder must degrade the call, not drop it."""

    class Broken:
        model_name = "broken"

        def embed(self, audio, sample_rate=None):
            raise RuntimeError("no model")

    encoder = StubEncoder()
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), Broken(), "ceo", SR, interval_seconds=1.0
    )

    assert tracker.add_speech(speech(2.0)) is None
    assert tracker.to_dict() is None


def test_empty_audio_is_a_noop():
    encoder = StubEncoder()
    tracker = LiveSpeakerTracker(
        enrolled_registry(encoder), encoder, "ceo", SR
    )

    assert tracker.add_speech(np.array([], dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline(monkeypatch):
    """AudioFeaturePipeline with both models stubbed."""

    from app.audio import feature_pipeline as fp
    from app.ml.detector import VoiceSpoofDetector

    class StubDetector:
        def __init__(self, probability=0.9):
            self.probability = probability

        def predict(self, audio, sample_rate=None):
            return {
                "synthetic_probability": self.probability,
                "bonafide_probability": 1 - self.probability,
                "confidence": self.probability,
                "model_status": "neural",
            }

    detector = StubDetector()
    monkeypatch.setattr(
        VoiceSpoofDetector, "get_shared", classmethod(lambda cls, cfg=None: detector)
    )

    return fp, detector


def pcm(seconds=1.0):
    return (speech(seconds) * 32767).astype("<i2").tobytes()


def test_pipeline_without_identity_skips_the_speaker_branch(pipeline):
    fp, _ = pipeline

    p = fp.AudioFeaturePipeline(sample_rate=SR)

    for _ in range(5):
        result = p.process(pcm(1.0))

    assert result["speaker"] is None
    assert result["stream"]["speaker_branch"] is False
    assert result["risk"] is not None
    assert "identity_mismatch" in result["risk"]["missing_signals"]


def test_pipeline_produces_a_fused_risk(pipeline):
    fp, _ = pipeline

    p = fp.AudioFeaturePipeline(sample_rate=SR)

    for _ in range(5):
        result = p.process(pcm(1.0))

    risk = result["risk"]
    assert 0.0 <= risk["risk_score"] <= 100.0
    assert risk["decision"]
    assert risk["reasons"]


def test_pipeline_survives_odd_byte_counts(pipeline):
    """A chunk split mid-sample must not kill the call."""

    fp, _ = pipeline
    p = fp.AudioFeaturePipeline(sample_rate=SR)

    assert p.process(pcm(1.0)[:-1]) is not None


def test_pipeline_reset_clears_state(pipeline):
    fp, _ = pipeline
    p = fp.AudioFeaturePipeline(sample_rate=SR)

    for _ in range(5):
        p.process(pcm(1.0))

    assert p.latest_risk is not None

    p.reset()

    assert p.latest_risk is None
    assert p.chunk_index == 0
    assert p.total_audio_ms == 0.0
