import numpy as np
import pytest

from app.ml.inference import StreamingInferenceEngine


def make_engine(stub_detector, probability=0.5):
    return StreamingInferenceEngine(
        detector=stub_detector(probability),
        sample_rate=16000,
        window_seconds=3.0,
        hop_seconds=1.0,
    )


def test_no_result_until_window_fills(stub_detector):
    engine = make_engine(stub_detector)

    assert engine.add_audio(np.zeros(16000, dtype=np.float32)) == []
    assert engine.add_audio(np.zeros(16000, dtype=np.float32)) == []

    results = engine.add_audio(np.zeros(16000, dtype=np.float32))
    assert len(results) == 1


def test_result_shape(stub_detector):
    engine = make_engine(stub_detector, probability=0.9)

    engine.add_audio(np.zeros(48000, dtype=np.float32))
    result = engine.add_audio(np.zeros(16000, dtype=np.float32))[-1]

    for key in (
        "synthetic_probability",
        "smoothed_probability",
        "risk_score",
        "risk_level",
        "reasons",
        "window_seconds",
    ):
        assert key in result

    assert 0.0 <= result["synthetic_probability"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["window_seconds"] == pytest.approx(3.0)


def test_hop_advances_by_exactly_one_second(stub_detector):
    """
    Feeding 3 s then 1 s at a time must yield exactly one score per second.

    The earlier implementation tracked buffered samples and
    samples-since-inference in two counters that drifted apart, so a single
    large chunk could emit a burst of windows over audio it had already scored.
    """

    engine = make_engine(stub_detector)

    engine.add_audio(np.zeros(48000, dtype=np.float32))

    for _ in range(5):
        results = engine.add_audio(np.zeros(16000, dtype=np.float32))
        assert len(results) == 1

    assert engine.windows_scored == 6


def test_one_large_chunk_emits_the_right_number_of_windows(stub_detector):
    """10 s in one call: one window at 3 s, then one per hop -> 8 windows."""

    engine = make_engine(stub_detector)

    results = engine.add_audio(np.zeros(160000, dtype=np.float32))

    assert len(results) == 8


def test_smoothing_converges_towards_the_raw_score(stub_detector):
    engine = make_engine(stub_detector, probability=1.0)

    engine.add_audio(np.zeros(48000, dtype=np.float32))
    first = engine.risk.smoothed

    for _ in range(20):
        engine.add_audio(np.zeros(16000, dtype=np.float32))

    assert first < engine.risk.smoothed
    assert engine.risk.smoothed > 0.95


def test_reset_clears_state(stub_detector):
    engine = make_engine(stub_detector, probability=0.9)

    engine.add_audio(np.zeros(160000, dtype=np.float32))
    assert engine.windows_scored > 0

    engine.reset()

    assert engine.windows_scored == 0
    assert engine.buffered_seconds == 0.0
    assert engine.risk.smoothed == 0.0


def test_empty_audio_is_a_noop(stub_detector):
    engine = make_engine(stub_detector)
    assert engine.add_audio(np.array([], dtype=np.float32)) == []


@pytest.mark.slow
def test_real_detector_reports_untrained_without_a_checkpoint():
    """
    With no checkpoint on disk the detector must say so rather than emitting a
    plausible-looking number from randomly initialised weights.
    """

    from app.ml.detector import VoiceSpoofDetector

    detector = VoiceSpoofDetector(model_path=None, device="cpu")
    result = detector.predict(np.zeros(16000, dtype=np.float32), 16000)

    assert result["model_status"] == "untrained"
    assert result["confidence"] == 0.0
