"""
Speaker-threshold calibration.

Tested on synthetic embeddings with a known structure, so the answers are known
in advance rather than being whatever the implementation produced. Real audio
only changes how separable the classes are; the trial construction and
threshold arithmetic are the parts that can be silently wrong.
"""

import numpy as np
import pytest

from app.ml.speaker import l2_normalize
from app.ml.speaker_calibration import (
    build_trials,
    calibrate,
    score_trials,
    _threshold_at_false_reject,
)


DIM = 192


def make_corpus(speakers=12, per_speaker=25, spread=0.55, seed=0):
    """
    Synthetic embeddings with a real speaker structure: each speaker has an
    identity direction, each utterance is that direction plus noise.
    """

    rng = np.random.default_rng(seed)

    embeddings = []
    by_speaker = {}

    for s in range(speakers):
        anchor = l2_normalize(rng.normal(size=DIM))
        indices = []
        for _ in range(per_speaker):
            noise = l2_normalize(rng.normal(size=DIM))
            indices.append(len(embeddings))
            embeddings.append(l2_normalize(anchor + spread * noise))
        by_speaker[f"spk{s:02d}"] = indices

    return np.stack(embeddings), by_speaker


# ---------------------------------------------------------------------------
# trial construction
# ---------------------------------------------------------------------------


def test_enrollment_is_held_out_of_probes():
    """
    The leak that would silently ruin the calibration.

    Scoring a probe against a prototype that contains that probe inflates
    target similarity towards 1.0, and the chosen threshold would then be one
    no live comparison can reach.
    """

    _, by_speaker = make_corpus()
    trial_set = build_trials(by_speaker, enroll_samples=5)

    probes = {t.probe_index for t in trial_set.trials}

    for identity, enrolled in trial_set.enrollment.items():
        assert not (set(enrolled) & probes), (
            f"{identity}: enrolment samples appear as probes"
        )


def test_trials_contain_both_classes():
    _, by_speaker = make_corpus()
    trial_set = build_trials(by_speaker, enroll_samples=5)

    assert trial_set.targets > 0
    assert trial_set.nontargets > 0
    assert trial_set.nontargets > trial_set.targets  # 4 impostors per probe


def test_nontarget_claims_a_different_identity():
    _, by_speaker = make_corpus()
    trial_set = build_trials(by_speaker, enroll_samples=5)

    owner = {}
    for identity, indices in by_speaker.items():
        for i in indices:
            owner[i] = identity

    for trial in trial_set.trials:
        if trial.is_target:
            assert owner[trial.probe_index] == trial.identity
        else:
            assert owner[trial.probe_index] != trial.identity


def test_too_few_speakers_raises():
    _, by_speaker = make_corpus(speakers=1)

    with pytest.raises(ValueError, match="at least 2 speakers"):
        build_trials(by_speaker, enroll_samples=5)


def test_speakers_with_too_little_audio_are_dropped():
    _, by_speaker = make_corpus(speakers=6, per_speaker=25)
    by_speaker["thin"] = [0, 1]        # fewer than enroll_samples

    trial_set = build_trials(by_speaker, enroll_samples=5)

    assert "thin" not in trial_set.enrollment


def test_trials_are_reproducible():
    _, by_speaker = make_corpus()

    a = build_trials(by_speaker, seed=7)
    b = build_trials(by_speaker, seed=7)

    assert [(t.identity, t.probe_index, t.is_target) for t in a.trials] == [
        (t.identity, t.probe_index, t.is_target) for t in b.trials
    ]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_targets_score_higher_than_nontargets():
    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)

    labels, scores = score_trials(trial_set, embeddings)

    assert scores[labels == 1].mean() > scores[labels == 0].mean() + 0.2


def test_labels_use_the_metrics_convention():
    """
    1 = target, so metrics' "false alarm" is an impostor accepted and "miss" is
    a genuine speaker rejected. Getting this backwards inverts every rate.
    """

    embeddings, by_speaker = make_corpus()
    trial_set = build_trials(by_speaker, enroll_samples=5)

    labels, _ = score_trials(trial_set, embeddings)

    assert set(np.unique(labels)) == {0, 1}
    assert labels.sum() == trial_set.targets


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibration_produces_three_ordered_modes():
    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    assert set(result.modes) == {"high_security", "balanced", "low_friction"}
    assert (
        result.modes["high_security"].match
        >= result.modes["balanced"].match
        >= result.modes["low_friction"].match
    )


def test_high_security_accepts_fewer_impostors_than_low_friction():
    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    strict = result.operating_points["high_security"]["false_accept_rate"]
    lenient = result.operating_points["low_friction"]["false_accept_rate"]

    assert strict <= lenient


def test_low_friction_rejects_fewer_genuine_than_high_security():
    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    strict = result.operating_points["high_security"]["false_reject_rate"]
    lenient = result.operating_points["low_friction"]["false_reject_rate"]

    assert lenient <= strict


def test_calibrated_thresholds_are_no_longer_flagged_uncalibrated():
    """The whole point: replace the placeholder with something measured."""

    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores, version_tag="20260910")

    for thresholds in result.modes.values():
        assert thresholds.calibrated
        assert "UNCALIBRATED" not in thresholds.version


def test_no_match_floor_is_never_above_a_match_threshold():
    """
    The UNCERTAIN band must never be inverted. On perfectly separable data the
    two boundaries may coincide - that is a legitimate degenerate case, not a
    bug - but no_match must never sit above match.
    """

    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    for name, thresholds in result.modes.items():
        assert thresholds.no_match <= thresholds.match, f"{name} band is inverted"


def test_overlapping_data_produces_a_real_uncertain_band():
    """
    With classes that actually overlap - the realistic case - there must be a
    genuine UNCERTAIN region, otherwise the three-way decision collapses to a
    binary one and the policy engine loses its "ask for a second factor" path.
    """

    # In 192 dimensions independent anchors are near-orthogonal, so speakers
    # separate trivially however much per-utterance noise is added. Real
    # confusability comes from speakers who SOUND ALIKE, so the anchors here
    # share a common base and differ only slightly.
    rng = np.random.default_rng(11)
    base = l2_normalize(rng.normal(size=DIM))

    embeddings, by_speaker = [], {}
    for s in range(14):
        anchor = l2_normalize(base + 0.40 * l2_normalize(rng.normal(size=DIM)))
        indices = []
        for _ in range(30):
            indices.append(len(embeddings))
            embeddings.append(
                l2_normalize(anchor + 1.10 * l2_normalize(rng.normal(size=DIM)))
            )
        by_speaker[f"spk{s:02d}"] = indices

    embeddings = np.stack(embeddings)

    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    assert result.eer > 0.01, "fixture is too separable to test the band"

    band = result.modes["balanced"]
    assert band.no_match < band.match


def test_separable_data_gives_low_eer():
    embeddings, by_speaker = make_corpus(spread=0.25)   # tight clusters
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    assert calibrate(labels, scores).eer < 0.05


def test_unstructured_data_gives_high_eer():
    """Random embeddings carry no speaker identity - EER must expose that."""

    rng = np.random.default_rng(3)
    embeddings = np.stack([l2_normalize(rng.normal(size=DIM)) for _ in range(300)])
    by_speaker = {f"spk{i}": list(range(i * 25, (i + 1) * 25)) for i in range(12)}

    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    assert calibrate(labels, scores).eer > 0.3


def test_calibration_rejects_single_class():
    with pytest.raises(ValueError, match="target and nontarget"):
        calibrate(np.ones(10, dtype=int), np.random.random(10))


def test_calibration_rejects_empty():
    with pytest.raises(ValueError, match="no trials"):
        calibrate(np.array([], dtype=int), np.array([]))


def test_result_serialises():
    embeddings, by_speaker = make_corpus(spread=0.5)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    data = calibrate(labels, scores).to_dict()

    assert data["modes"]["balanced"]["match"]
    assert data["operating_points"]["balanced"]["false_accept_rate"] >= 0
    assert "eer_percent" in data


# ---------------------------------------------------------------------------
# the false-reject helper
# ---------------------------------------------------------------------------


def test_false_reject_threshold_respects_its_budget():
    labels = np.array([1] * 1000 + [0] * 1000)
    scores = np.concatenate([
        np.linspace(0.4, 0.9, 1000),      # genuine
        np.linspace(0.0, 0.5, 1000),      # impostor
    ])

    threshold = _threshold_at_false_reject(labels, scores, 0.01)
    achieved = float((scores[labels == 1] < threshold).mean())

    assert achieved <= 0.012


def test_tighter_budget_gives_a_lower_threshold():
    labels = np.array([1] * 1000 + [0] * 1000)
    scores = np.concatenate([np.linspace(0.4, 0.9, 1000), np.linspace(0.0, 0.5, 1000)])

    assert _threshold_at_false_reject(labels, scores, 0.001) <= (
        _threshold_at_false_reject(labels, scores, 0.05)
    )


def test_low_friction_is_actually_more_permissive_than_balanced():
    """
    The regression from the first real calibration run.

    low_friction was derived from a false-REJECT budget, which inverts it:
    allowing more genuine rejections permits a STRICTER threshold. So
    "low friction" came out at least as strict as "balanced" and collapsed
    onto it - both the dev and eval runs produced identical thresholds for the
    two modes, leaving no low-friction option at all.

    All three modes now sit on the false-accept axis and differ only in how
    much of that budget they spend.
    """

    # Overlapping enough that the budgets genuinely separate.
    rng = np.random.default_rng(5)
    base = l2_normalize(rng.normal(size=DIM))

    embeddings, by_speaker = [], {}
    for s in range(14):
        anchor = l2_normalize(base + 0.40 * l2_normalize(rng.normal(size=DIM)))
        indices = []
        for _ in range(30):
            indices.append(len(embeddings))
            embeddings.append(
                l2_normalize(anchor + 1.10 * l2_normalize(rng.normal(size=DIM)))
            )
        by_speaker[f"spk{s:02d}"] = indices

    embeddings = np.stack(embeddings)
    trial_set = build_trials(by_speaker, enroll_samples=5)
    labels, scores = score_trials(trial_set, embeddings)

    result = calibrate(labels, scores)

    assert result.eer > 0.005, "fixture too separable to distinguish the modes"

    strict = result.modes["high_security"].match
    balanced = result.modes["balanced"].match
    lenient = result.modes["low_friction"].match

    assert strict > lenient, "high_security must be stricter than low_friction"
    assert lenient < balanced, (
        "low_friction must be MORE PERMISSIVE than balanced, not equal to it"
    )

    points = result.operating_points
    assert (
        points["low_friction"]["false_reject_rate"]
        < points["high_security"]["false_reject_rate"]
    ), "low friction means fewer genuine callers turned away"
    assert (
        points["low_friction"]["false_accept_rate"]
        > points["high_security"]["false_accept_rate"]
    ), "and that is paid for with more impostors admitted"
