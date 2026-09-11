"""
Identity validation, calibration loading, and the speech-presence gate.

Each of these covers a defect found by running the system rather than reading
it - a live registry that had accumulated "../../../../tmp/pwned" and a
5,000-character name, a calibration file that vanished on a clean clone, and a
detector that called four seconds of white noise synthetic with 0.997
confidence.
"""

import json

import numpy as np
import pytest

from app.audio.speech_check import assess_speech
from app.ml.speaker import SpeakerRegistry, SpeakerThresholds, l2_normalize


def anchor(seed: int, spread: float = 0.05, count: int = 5):
    """A consistent set of enrolment embeddings for one synthetic speaker."""

    rng = np.random.default_rng(seed)
    base = l2_normalize(rng.normal(size=192))
    return [
        l2_normalize(base + spread * l2_normalize(rng.normal(size=192)))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# identity validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identity",
    [
        "../../../../tmp/pwned",
        "..\\..\\windows\\system32",
        "a/b",
        "a\\b",
        "with\x00null",
        "with\nnewline",
        "  ",
        "",
        "X" * 129,
        "-leading-dash",
        "<script>alert(1)</script>",
    ],
)
def test_dangerous_identities_are_refused(identity):
    with pytest.raises(ValueError):
        SpeakerRegistry.validate_identity(identity)


@pytest.mark.parametrize(
    "identity",
    ["ceo", "Karthikeya", "finance.director", "user_01", "a@b.com", "Jane Doe", "X" * 128],
)
def test_reasonable_identities_are_accepted(identity):
    assert SpeakerRegistry.validate_identity(identity) == identity


def test_identity_is_trimmed():
    assert SpeakerRegistry.validate_identity("  ceo  ") == "ceo"


def test_enroll_refuses_a_traversal_identity(tmp_path):
    registry = SpeakerRegistry(store_path=tmp_path / "speakers.json")

    with pytest.raises(ValueError, match="may contain only"):
        registry.enroll("../../etc/passwd", anchor(1))

    assert len(registry) == 0


def test_enroll_stores_the_trimmed_identity(tmp_path):
    registry = SpeakerRegistry(store_path=tmp_path / "speakers.json")

    profile = registry.enroll("  ceo  ", anchor(2))

    assert profile.identity == "ceo"
    assert "ceo" in registry


# ---------------------------------------------------------------------------
# loading a store written before validation existed
# ---------------------------------------------------------------------------


def test_invalid_stored_identities_are_skipped_with_a_warning(tmp_path, caplog):
    """
    The real registry had four such entries. They must not keep working just
    because they predate the rule - but the file is left alone, so nothing is
    destroyed by merely reading it.
    """

    store = tmp_path / "speakers.json"
    good = SpeakerRegistry(store_path=store)
    good.enroll("ceo", anchor(3))

    data = json.loads(store.read_text(encoding="utf-8"))
    hostile = dict(data["profiles"][0])
    hostile["identity"] = "../../../../tmp/pwned"
    data["profiles"].append(hostile)
    store.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level("WARNING"):
        reloaded = SpeakerRegistry(store_path=store)

    assert "ceo" in reloaded
    assert "../../../../tmp/pwned" not in reloaded
    assert len(reloaded.rejected) == 1
    assert "no longer valid" in caplog.text

    # The file itself is untouched until something saves.
    assert len(json.loads(store.read_text(encoding="utf-8"))["profiles"]) == 2


def test_the_store_does_not_override_the_active_thresholds(tmp_path):
    """
    ``load`` used to overwrite self.thresholds from the store, so a stale
    speakers.json silently replaced the calibrated thresholds the caller had
    just read from disk - and the calibration file was effectively ignored
    whenever a store existed.
    """

    store = tmp_path / "speakers.json"

    stale = SpeakerRegistry(
        thresholds=SpeakerThresholds(match=0.9, no_match=0.8, version="v0-UNCALIBRATED"),
        store_path=store,
    )
    stale.enroll("ceo", anchor(4))

    calibrated = SpeakerThresholds(match=0.44, no_match=0.33, version="v1-calibrated")
    reloaded = SpeakerRegistry(thresholds=calibrated, store_path=store)

    assert reloaded.thresholds.version == "v1-calibrated"
    assert reloaded.thresholds.match == pytest.approx(0.44)
    assert reloaded.enrolled_under["version"] == "v0-UNCALIBRATED"


def test_calibration_is_found_without_the_gitignored_models_directory(monkeypatch):
    """
    models/ is gitignored, so a clean clone has no speaker_thresholds.json
    there. The committed copy under results/ must be found instead, or the
    registry answers MATCH / NO_MATCH from uncalibrated placeholders while
    reporting itself calibrated=False that nobody reads.
    """

    from app.api import speakers as module

    monkeypatch.setattr(
        module, "CALIBRATION_CANDIDATES",
        ("models/does-not-exist.json", "results/speaker-calibration/speaker_thresholds_dev.json"),
    )

    thresholds, source = module.load_speaker_thresholds()

    assert thresholds.calibrated
    assert source is not None
    assert "results" in source


def test_missing_calibration_degrades_instead_of_raising(monkeypatch):
    from app.api import speakers as module

    monkeypatch.setattr(module, "CALIBRATION_CANDIDATES", ("nope/a.json", "nope/b.json"))

    thresholds, source = module.load_speaker_thresholds()

    assert not thresholds.calibrated
    assert source is None


# ---------------------------------------------------------------------------
# the speech-presence gate
# ---------------------------------------------------------------------------


SR = 16000


def steady(kind: str, seconds: float = 4.0):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)

    return {
        "silence": np.zeros(n, dtype=np.float32),
        "white noise": rng.normal(0, 0.1, n).astype(np.float32),
        "tone": (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
        "dc": np.full(n, 0.4, dtype=np.float32),
        "square": (np.sign(np.sin(2 * np.pi * 200 * t)) * 0.9).astype(np.float32),
        "hum": (
            0.2 * sum(np.sin(2 * np.pi * 50 * (k + 1) * t) / (k + 1) for k in range(6))
        ).astype(np.float32),
        "fading noise": (
            rng.normal(0, 0.1, n) * np.linspace(0.3, 1.0, n)
        ).astype(np.float32),
    }[kind]


@pytest.mark.parametrize(
    "kind", ["silence", "white noise", "tone", "dc", "square", "hum", "fading noise"]
)
def test_steady_signals_are_not_speech(kind):
    assert not assess_speech(steady(kind), SR).has_speech


def test_the_threshold_sits_between_the_two_populations(speech_like):
    """
    The gate is a measured boundary, not a guess, and it has to hold from both
    sides: above every steady signal, and below speech that has been buried in
    noise. Pinning both ends means a later tweak that collapses either margin
    fails loudly instead of silently degrading one of them.
    """

    from app.audio.speech_check import MINIMUM_DYNAMIC_RANGE

    worst_non_speech = max(
        assess_speech(steady(kind), SR).dynamic_range
        for kind in ("white noise", "hum", "fading noise", "tone", "dc", "square")
    )

    clean = speech_like(4.0)
    rng = np.random.default_rng(2)
    power = np.sqrt(np.mean(clean**2))
    noise = rng.normal(0, power / (10 ** (10 / 20)), clean.size).astype(np.float32)
    noisy_speech = assess_speech(clean + noise, SR).dynamic_range

    assert worst_non_speech < MINIMUM_DYNAMIC_RANGE < noisy_speech


def test_speech_is_speech(speech_like):
    result = assess_speech(speech_like(4.0), SR)

    assert result.has_speech
    assert result.dynamic_range > 100


def test_the_verdict_carries_its_reason():
    result = assess_speech(steady("white noise"), SR)

    assert "steady" in result.reason
    assert result.dynamic_range < 5.0


def test_speech_survives_heavy_background_noise(speech_like):
    """
    The threshold is deliberately far below the quietest real speech, because
    continuous background noise floors the quiet parts and compresses the
    measurement. This is the case that margin protects.
    """

    clean = speech_like(4.0)
    rng = np.random.default_rng(1)
    power = np.sqrt(np.mean(clean**2))

    for snr_db in (20, 15, 10):
        noise = rng.normal(0, power / (10 ** (snr_db / 20)), clean.size).astype(np.float32)
        assert assess_speech(clean + noise, SR).has_speech, f"{snr_db} dB SNR failed"


def test_an_empty_clip_is_not_speech():
    assert not assess_speech(np.array([], dtype=np.float32), SR).has_speech


def test_a_very_short_clip_is_allowed_through():
    """
    Too short to measure the gaps between words. The endpoints already enforce
    a minimum length, so refusing here as well would reject on the wrong
    grounds and give a misleading reason.
    """

    result = assess_speech(np.full(int(SR * 0.2), 0.1, dtype=np.float32), SR)

    assert result.has_speech
    assert "too short" in result.reason


def test_nan_and_inf_do_not_propagate():
    audio = np.array([np.nan, np.inf, -np.inf] + [0.1] * SR, dtype=np.float32)

    result = assess_speech(audio, SR)

    assert np.isfinite(result.rms)
    assert np.isfinite(result.dynamic_range)
