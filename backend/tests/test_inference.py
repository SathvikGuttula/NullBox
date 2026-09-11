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


def test_result_shape(stub_detector, speech_like):
    engine = make_engine(stub_detector, probability=0.9)

    engine.add_audio(speech_like(3.0))
    result = engine.add_audio(speech_like(1.0, seed=1))[-1]

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


def test_smoothing_converges_towards_the_raw_score(stub_detector, speech_like):
    engine = make_engine(stub_detector, probability=1.0)

    engine.add_audio(speech_like(3.0))
    first = engine.risk.smoothed

    for index in range(20):
        engine.add_audio(speech_like(1.0, seed=index + 1))

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


# ---------------------------------------------------------------------------
# the speech-presence gate
# ---------------------------------------------------------------------------


def test_silence_is_not_scored(stub_detector):
    """
    The detector has only ever seen speech. Handed four seconds of digital
    silence it reports 0.999 synthetic - a confident answer about a class it
    was never trained on. The window must be refused instead.
    """

    engine = make_engine(stub_detector, probability=0.9)

    engine.add_audio(np.zeros(48000, dtype=np.float32))
    result = engine.add_audio(np.zeros(16000, dtype=np.float32))[-1]

    assert result["model_status"] == "no_speech"
    assert result["synthetic_probability"] is None
    assert "silent" in result["reasons"][0].lower()


def test_steady_noise_is_not_scored(stub_detector):
    """
    The energy VAD upstream passes white noise as speech - measured, 31 chunks
    of 31. This is the gate that stops it reaching the model.
    """

    engine = make_engine(stub_detector, probability=0.9)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 64000).astype(np.float32)

    result = engine.add_audio(noise)[-1]

    assert result["model_status"] == "no_speech"
    assert result["synthetic_probability"] is None


def test_a_refused_window_does_not_move_the_risk(stub_detector, speech_like):
    """
    Silence is not evidence of innocence. Smoothing a refused window in as a
    zero would drag an established risk score down towards ALLOW during a
    pause - exactly when an attacker would want it to.
    """

    engine = make_engine(stub_detector, probability=1.0)

    for index in range(8):
        engine.add_audio(speech_like(1.0, seed=index))

    assert engine.risk.smoothed > 0.5

    # Flush the buffer first. Until the last speech sample has left it, the
    # sliding window still straddles speech and silence, and those mixed
    # windows are legitimately scored.
    engine.add_audio(np.zeros(16000 * 6, dtype=np.float32))
    established = engine.risk.smoothed

    for _ in range(10):
        engine.add_audio(np.zeros(16000, dtype=np.float32))

    assert engine.risk.smoothed == pytest.approx(established)


def test_a_refused_window_still_reports_the_standing_risk(stub_detector, speech_like):
    engine = make_engine(stub_detector, probability=1.0)

    for index in range(8):
        engine.add_audio(speech_like(1.0, seed=index))

    engine.add_audio(np.zeros(16000 * 6, dtype=np.float32))
    established = engine.risk.smoothed

    result = engine.add_audio(np.zeros(16000, dtype=np.float32))[-1]

    assert result["model_status"] == "no_speech"
    assert result["smoothed_probability"] == pytest.approx(established, abs=1e-4)
    assert result["risk_level"] == engine.risk.level


def test_speech_clears_the_gate(stub_detector, speech_like):
    engine = make_engine(stub_detector, probability=0.9)

    result = engine.add_audio(speech_like(4.0))[-1]

    assert result["model_status"] == "stub"
    assert result["synthetic_probability"] is not None
    assert result["speech_check"]["has_speech"] is True
