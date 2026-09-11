"""
Speaker verification.

The encoder needs real speech to evaluate meaningfully — synthetic tones are
far out of distribution for a model trained on human voices, and it maps them
all into a similar region. So these tests exercise the parts that are pure
vector maths and decision logic, which is where the bugs that would actually
hurt live. Encoder quality is a property of the published pretrained model and
is measured by scripts/calibrate_speaker.py against real trials.
"""

import numpy as np
import pytest

from app.ml.speaker import (
    MATCH,
    NO_MATCH,
    UNCERTAIN,
    SpeakerProfile,
    SpeakerRegistry,
    SpeakerThresholds,
    centroid,
    cosine_similarity,
    enrollment_consistency,
    l2_normalize,
)


DIM = 192


def voice(seed: int, spread: float = 0.0, base: int | None = None) -> np.ndarray:
    """
    A synthetic embedding.

    ``base`` gives several samples a shared identity direction with ``spread``
    of per-sample variation, which is what a real enrolment set looks like.
    """

    rng = np.random.default_rng(seed)
    vector = rng.normal(size=DIM)

    if base is not None:
        anchor = np.random.default_rng(base).normal(size=DIM)
        vector = l2_normalize(anchor) + spread * l2_normalize(vector)

    return l2_normalize(vector)


def enrollment(base: int, count: int = 5, spread: float = 0.35) -> list[np.ndarray]:
    return [voice(base * 1000 + i, spread=spread, base=base) for i in range(count)]


# ---------------------------------------------------------------------------
# vector maths
# ---------------------------------------------------------------------------


def test_l2_normalize_gives_unit_length():
    assert np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)


def test_l2_normalize_leaves_zero_alone():
    zero = np.zeros(DIM)
    assert np.array_equal(l2_normalize(zero), zero)


def test_cosine_of_identical_vectors_is_one():
    v = voice(1)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_is_scale_invariant():
    v = voice(2)
    assert cosine_similarity(v, v * 7.3) == pytest.approx(1.0)


def test_cosine_of_orthogonal_is_zero():
    a = np.zeros(DIM); a[0] = 1.0
    b = np.zeros(DIM); b[1] = 1.0
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_centroid_normalises_before_averaging():
    """
    Raw ECAPA embeddings vary in magnitude with utterance length and loudness.
    Averaging without normalising first lets the longest enrolment clip
    dominate the prototype.
    """

    a = voice(10)
    b = voice(11)

    balanced = centroid([a, b])
    # b supplied 50x longer - must not drag the centroid onto itself.
    lopsided = centroid([a, b * 50.0])

    assert np.allclose(balanced, lopsided, atol=1e-9)


def test_centroid_is_unit_length():
    assert np.linalg.norm(centroid(enrollment(3))) == pytest.approx(1.0)


def test_centroid_of_one_sample_is_that_sample():
    v = voice(5)
    assert np.allclose(centroid([v]), v)


def test_centroid_rejects_empty():
    with pytest.raises(ValueError):
        centroid([])


def test_consistency_is_high_for_one_speaker():
    assert enrollment_consistency(enrollment(7, spread=0.2)) > 0.8


def test_consistency_is_low_for_mixed_speakers():
    """Five unrelated voices must not look like a coherent enrolment."""

    mixed = [voice(i) for i in range(100, 105)]
    assert enrollment_consistency(mixed) < 0.3


def test_consistency_of_single_sample_is_one():
    assert enrollment_consistency([voice(1)]) == 1.0


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


def test_three_way_decision():
    t = SpeakerThresholds(match=0.6, no_match=0.4)

    assert t.decide(0.9) == MATCH
    assert t.decide(0.5) == UNCERTAIN
    assert t.decide(0.1) == NO_MATCH


def test_uncertain_band_exists():
    """
    A two-way split forces a guess exactly where the system knows least. The
    band lets the policy engine ask for a second factor instead.
    """

    t = SpeakerThresholds.balanced()
    assert t.no_match < t.match
    assert t.decide((t.match + t.no_match) / 2) == UNCERTAIN


def test_defaults_are_flagged_uncalibrated():
    """
    There is no universal cosine threshold for 'same speaker'. Shipping one
    silently as if it were measured is the mistake the handoff warns about.
    """

    for t in (
        SpeakerThresholds(),
        SpeakerThresholds.balanced(),
        SpeakerThresholds.high_security(),
        SpeakerThresholds.low_friction(),
    ):
        assert not t.calibrated
        assert "UNCALIBRATED" in t.version


def test_operating_modes_are_ordered():
    assert (
        SpeakerThresholds.high_security().match
        > SpeakerThresholds.balanced().match
        > SpeakerThresholds.low_friction().match
    )


# ---------------------------------------------------------------------------
# enrollment
# ---------------------------------------------------------------------------


def test_enroll_and_verify_the_same_speaker():
    registry = SpeakerRegistry()
    registry.enroll("alice", enrollment(1, spread=0.2))

    result = registry.verify("alice", voice(1999, spread=0.2, base=1))

    assert result.decision == MATCH
    assert result.similarity > 0.6


def test_a_different_speaker_does_not_match():
    registry = SpeakerRegistry()
    registry.enroll("alice", enrollment(1, spread=0.2))

    result = registry.verify("alice", voice(4242))

    assert result.decision == NO_MATCH
    assert "does not match" in result.reasons[0]


def test_enrollment_requires_enough_samples():
    """One sample captures a recording condition, not a person."""

    registry = SpeakerRegistry()

    with pytest.raises(ValueError, match="at least 3"):
        registry.enroll("bob", [voice(1), voice(2)])


def test_inconsistent_enrollment_is_refused():
    """
    Five clips that are not all the same person produce a prototype that
    happily matches several people - a permanently weakened account, not a
    one-off error.
    """

    registry = SpeakerRegistry()
    mixed = [voice(i) for i in range(200, 205)]

    with pytest.raises(ValueError, match="inconsistent"):
        registry.enroll("mixed", mixed)


def test_inconsistent_enrollment_can_be_overridden():
    registry = SpeakerRegistry()
    mixed = [voice(i) for i in range(300, 305)]

    profile = registry.enroll("mixed", mixed, allow_inconsistent=True)
    assert profile.sample_count == 5


def test_unknown_identity_is_no_match_not_an_error():
    registry = SpeakerRegistry()
    result = registry.verify("nobody", voice(1))

    assert result.decision == NO_MATCH
    assert result.enrollment_samples == 0
    assert "No enrolled voice profile" in result.reasons[0]


# ---------------------------------------------------------------------------
# explanations
# ---------------------------------------------------------------------------


def test_uncalibrated_thresholds_are_declared_in_every_result():
    registry = SpeakerRegistry()
    registry.enroll("alice", enrollment(1, spread=0.2))

    result = registry.verify("alice", voice(1999, spread=0.2, base=1))

    assert any("uncalibrated" in r.lower() for r in result.reasons)
    assert result.to_dict()["calibrated"] is False


def test_thin_enrollment_is_reported():
    registry = SpeakerRegistry()
    registry.enroll("alice", enrollment(1, count=3, spread=0.2))

    result = registry.verify("alice", voice(1999, spread=0.2, base=1))

    assert any("only 3 samples" in r for r in result.reasons)


def test_result_serialises():
    registry = SpeakerRegistry()
    registry.enroll("alice", enrollment(1, spread=0.2))

    data = registry.verify("alice", voice(1999, spread=0.2, base=1)).to_dict()

    for key in ("identity", "similarity", "decision", "reasons", "calibrated"):
        assert key in data


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_registry_roundtrips(tmp_path):
    path = tmp_path / "speakers.json"

    registry = SpeakerRegistry(store_path=path)
    registry.enroll("alice", enrollment(1, spread=0.2))
    registry.enroll("bob", enrollment(2, spread=0.2))

    restored = SpeakerRegistry(store_path=path)

    assert len(restored) == 2
    assert "alice" in restored

    probe = voice(1999, spread=0.2, base=1)
    assert restored.verify("alice", probe).similarity == pytest.approx(
        registry.verify("alice", probe).similarity
    )


def test_store_contains_no_audio(tmp_path):
    """Embeddings are stored, never waveforms. Privacy model, handoff §15."""

    path = tmp_path / "speakers.json"
    registry = SpeakerRegistry(store_path=path)
    registry.enroll("alice", enrollment(1, spread=0.2))

    import json

    data = json.loads(path.read_text())
    entry = data["profiles"][0]

    assert set(entry) == {
        "identity", "centroid", "sample_count", "consistency",
        "encoder", "created_at", "threshold_version",
    }
    assert len(entry["centroid"]) == DIM


def test_identify_ranks_the_right_speaker_first():
    registry = SpeakerRegistry()
    for base in (1, 2, 3):
        registry.enroll(f"speaker{base}", enrollment(base, spread=0.2))

    ranked = registry.identify(voice(1999, spread=0.2, base=2))

    assert ranked[0][0] == "speaker2"
