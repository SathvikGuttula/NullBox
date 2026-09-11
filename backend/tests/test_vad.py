import numpy as np
import pytest

from app.audio.vad import VoiceActivityDetector


SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 10  # 100 ms chunks


def silence(level: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, level, FRAME)).astype(np.float32)


def speech(level: float = 0.15) -> np.ndarray:
    t = np.linspace(0, FRAME / SAMPLE_RATE, FRAME, endpoint=False)
    signal = sum(
        (1.0 / (k + 1)) * np.sin(2 * np.pi * 140 * (k + 1) * t) for k in range(12)
    )
    return (signal / np.abs(signal).max() * level).astype(np.float32)


def test_silence_is_not_speech():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    results = [vad.process(silence()) for _ in range(30)]

    assert not any(r["speech"] for r in results[5:])


def test_speech_after_silence_is_detected():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    for _ in range(20):
        vad.process(silence())

    assert vad.process(speech())["speech"]


def test_stream_that_opens_mid_speech_still_detects_speech():
    """
    The regression that mattered.

    The old detector seeded its noise floor from the very first chunk. A
    websocket opens when the call connects, so the first chunk is normally
    someone already talking - the floor got seeded to speech level, the
    threshold sat above every subsequent speech frame, and adaptation could
    never bring it down. The VAD reported silence for the whole call and the
    anti-spoof model was never fed a single window.
    """

    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    results = [vad.process(speech()) for _ in range(40)]

    detected = sum(1 for r in results if r["speech"])

    assert detected > 30, (
        f"only {detected}/40 speech frames detected - the noise floor is "
        f"tracking the speech instead of the silence"
    )


def test_noise_floor_converges_below_speech_level():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    for _ in range(40):
        vad.process(speech())

    # Even fed nothing but speech, the floor must stay well under the signal.
    assert vad.noise_floor < 0.15


def test_hangover_bridges_a_short_gap():
    """One quiet frame mid-word must not end the utterance."""

    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, hangover_frames=3)

    for _ in range(20):
        vad.process(silence())
    for _ in range(10):
        vad.process(speech())

    assert vad.process(silence())["speech"] is True


def test_hangover_eventually_expires():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, hangover_frames=3)

    for _ in range(20):
        vad.process(silence())
    for _ in range(10):
        vad.process(speech())

    results = [vad.process(silence()) for _ in range(10)]

    assert results[-1]["speech"] is False


def test_empty_audio():
    vad = VoiceActivityDetector()
    result = vad.process(np.array([], dtype=np.float32))

    assert result["speech"] is False
    assert result["rms"] == 0.0


def test_confidence_is_bounded():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    for chunk in (silence(), speech(), speech(level=1.0)):
        assert 0.0 <= vad.process(chunk)["confidence"] <= 1.0


def test_reset_restores_the_initial_floor():
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)

    for _ in range(30):
        vad.process(speech())

    vad.reset()

    assert vad.noise_floor == pytest.approx(vad.initial_noise_floor)
    assert len(vad.history) == 0
