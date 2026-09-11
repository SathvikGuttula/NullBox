"""
Affine (Platt) calibration, and the reason temperature scaling was not enough.

The trained detector ranks well but its raw output is not a probability: the
equal-error point on the ASVspoof 2019 LA evaluation set falls at a raw score
of 0.044, so ``p >= 0.5`` misses 12% of attacks. These tests pin down why a
temperature cannot fix that, that the affine fit can, and that the calibration
is applied exactly once on the serving path.
"""

import json

import numpy as np
import pytest

from app.ml.calibration import ProbabilityCalibrator
from app.ml.config import DetectorConfig, resolve_path


# ---------------------------------------------------------------------------
# why an affine map is required
# ---------------------------------------------------------------------------


def test_temperature_cannot_move_the_boundary():
    """
    Dividing a logit by a positive T never changes its sign, so a decision
    boundary below 0.5 stays below 0.5 for every T. This is the whole reason
    ``fit_affine`` exists - no temperature could have repaired this detector.
    """

    below_half = 0.044  # the detector's measured equal-error operating point

    for temperature in (0.05, 0.5, 1.0, 2.0, 10.0, 1000.0):
        moved = ProbabilityCalibrator(temperature=temperature).calibrate(below_half)
        assert moved < 0.5, f"T={temperature} pushed it to {moved}"


def test_bias_moves_the_boundary():
    calibrator = ProbabilityCalibrator(scale=1.0, bias=4.0)

    assert calibrator.calibrate(0.044) > 0.5


def test_identity_is_still_the_default():
    calibrator = ProbabilityCalibrator()

    assert calibrator.is_identity
    assert calibrator.calibrate(0.73) == pytest.approx(0.73)


def test_temperature_and_scale_are_two_views_of_one_number():
    calibrator = ProbabilityCalibrator(temperature=4.0)

    assert calibrator.scale == pytest.approx(0.25)
    assert calibrator.temperature == pytest.approx(4.0)

    calibrator.temperature = 2.0
    assert calibrator.scale == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def shifted_scores(n=20000, seed=0, true_scale=0.3, true_bias=-3.0):
    """
    Scores whose implied boundary sits well away from 0.5.

    Labels are generated from the *calibrated* probability, so a correct fit
    recovers the inverse of (true_scale, true_bias).
    """

    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 3.0, n)
    probability = 1.0 / (1.0 + np.exp(-(true_scale * logits + true_bias)))
    labels = (rng.random(n) < probability).astype(int)
    raw = 1.0 / (1.0 + np.exp(-logits))
    return raw, labels


def test_fit_affine_recovers_known_parameters():
    raw, labels = shifted_scores()

    calibrator = ProbabilityCalibrator()
    scale, bias = calibrator.fit_affine(raw, labels)

    assert scale == pytest.approx(0.3, rel=0.15)
    assert bias == pytest.approx(-3.0, rel=0.15)


def test_fit_affine_beats_temperature_when_the_boundary_is_shifted():
    raw, labels = shifted_scores()

    baseline = ProbabilityCalibrator.expected_calibration_error(raw, labels)

    temperature_only = ProbabilityCalibrator()
    temperature_only.fit(raw, labels)
    after_temperature = ProbabilityCalibrator.expected_calibration_error(
        temperature_only.calibrate_array(raw), labels
    )

    affine = ProbabilityCalibrator()
    affine.fit_affine(raw, labels)
    after_affine = ProbabilityCalibrator.expected_calibration_error(
        affine.calibrate_array(raw), labels
    )

    assert after_affine < after_temperature < baseline


def test_calibration_preserves_ranking():
    """
    Affine scaling in log-odds is monotone, so EER and ROC-AUC are untouched.
    That is what makes it safe to apply to an already-evaluated model.
    """

    rng = np.random.default_rng(3)
    scores = rng.random(2000)

    calibrated = ProbabilityCalibrator(scale=3.75, bias=9.5).calibrate_array(scores)

    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))


def test_fit_affine_rejects_a_single_class():
    with pytest.raises(ValueError, match="both classes"):
        ProbabilityCalibrator().fit_affine([0.1, 0.2, 0.3], [1, 1, 1])


def test_fit_affine_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="probabilities and"):
        ProbabilityCalibrator().fit_affine([0.1, 0.2], [1, 0, 1])


def test_fit_affine_rejects_empty_input():
    with pytest.raises(ValueError, match="no samples"):
        ProbabilityCalibrator().fit_affine([], [])


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------


def test_prior_shift_lowers_scores_when_spoof_is_rarer():
    """
    A bias fitted on 90%-spoof data encodes that base rate. Re-expressing it at
    a 5% base rate must make every score less alarming, not more.
    """

    fitted = ProbabilityCalibrator(scale=3.75, bias=11.66)
    deployed = fitted.with_prior(0.05, fitted_prior=0.897)

    for raw in (0.01, 0.1, 0.5, 0.9):
        assert deployed.calibrate(raw) < fitted.calibrate(raw)


def test_prior_shift_is_reversible():
    original = ProbabilityCalibrator(scale=2.0, bias=1.5)

    there = original.with_prior(0.05, fitted_prior=0.9)
    back = there.with_prior(0.9, fitted_prior=0.05)

    assert back.bias == pytest.approx(original.bias)
    assert back.scale == pytest.approx(original.scale)


def test_prior_shift_does_not_change_the_ranking():
    rng = np.random.default_rng(7)
    scores = rng.random(500)

    fitted = ProbabilityCalibrator(scale=3.75, bias=11.66)
    deployed = fitted.with_prior(0.05, fitted_prior=0.897)

    assert np.array_equal(
        np.argsort(fitted.calibrate_array(scores)),
        np.argsort(deployed.calibrate_array(scores)),
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_impossible_priors_are_rejected(bad):
    with pytest.raises(ValueError):
        ProbabilityCalibrator(scale=2.0).with_prior(bad, fitted_prior=0.5)

    with pytest.raises(ValueError):
        ProbabilityCalibrator(scale=2.0).with_prior(0.5, fitted_prior=bad)


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def test_extreme_inputs_stay_in_range():
    calibrator = ProbabilityCalibrator(scale=3.75, bias=9.5)

    for probability in (0.0, 1.0, 1e-300, 1.0 - 1e-16):
        value = calibrator.calibrate(probability)
        assert 0.0 <= value <= 1.0
        assert np.isfinite(value)


def test_a_large_negative_bias_does_not_overflow():
    """``exp(-z)`` overflows for very negative z; the sigmoid must branch."""

    calibrator = ProbabilityCalibrator(scale=10.0, bias=-900.0)

    assert calibrator.calibrate(0.001) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isfinite(calibrator.calibrate_array([0.001, 0.5, 0.999])))


def test_array_and_scalar_paths_agree():
    calibrator = ProbabilityCalibrator(scale=3.75, bias=9.5)
    scores = [0.001, 0.044, 0.5, 0.97]

    scalar = [calibrator.calibrate(s) for s in scores]
    array = calibrator.calibrate_array(scores)

    assert np.allclose(scalar, array)


def test_log_odds_discriminates_where_probability_saturates():
    """
    Past a calibrated log-odds of about 36 the probability rounds to exactly
    1.0, so two very different scores compare equal. Operating points in that
    region have to be read in log-odds.
    """

    calibrator = ProbabilityCalibrator(scale=3.75, bias=9.5)

    # Both of these land past the saturation point and compare exactly equal
    # as probabilities, while remaining an order of magnitude apart in the
    # evidence they actually carry.
    assert calibrator.calibrate(0.999999) == 1.0
    assert calibrator.calibrate(0.9999999) == 1.0
    assert calibrator.log_odds(0.999999) < calibrator.log_odds(0.9999999)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path):
    calibrator = ProbabilityCalibrator(scale=3.754526, bias=9.499884)
    calibrator.fitted = True
    calibrator.metadata = {"prior": 0.5}

    restored = ProbabilityCalibrator.load(calibrator.save(tmp_path / "c.json"))

    assert restored.scale == pytest.approx(3.754526)
    assert restored.bias == pytest.approx(9.499884)
    assert restored.fitted
    assert restored.metadata["prior"] == 0.5


def test_a_temperature_only_file_still_loads(tmp_path):
    """
    Files written before affine calibration existed carry only a temperature.
    Reading one must honour it, not silently reset to the identity.
    """

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"temperature": 2.5, "fitted": True}), encoding="utf-8")

    restored = ProbabilityCalibrator.load(path)

    assert restored.temperature == pytest.approx(2.5)
    assert restored.bias == 0.0
    assert restored.fitted


# ---------------------------------------------------------------------------
# the shipped artifact
# ---------------------------------------------------------------------------


def test_the_shipped_calibration_exists_and_is_committed():
    """
    A calibration that vanishes on clone is worse than none: the detector then
    reports raw scores as though they were probabilities. This is why the
    artifact lives under results/ and not under the gitignored models/.
    """

    path = resolve_path(DetectorConfig().calibration_path)

    assert path.exists(), f"missing {path} - run scripts/calibrate_detector.py"
    assert "results" in path.parts


def test_the_shipped_calibration_moves_the_boundary_to_where_it_belongs():
    calibrator = ProbabilityCalibrator.load(
        resolve_path(DetectorConfig().calibration_path)
    )

    assert calibrator.fitted
    assert not calibrator.is_identity

    # A typical bonafide raw score stays low; a typical spoof raw score
    # becomes decisive. Both numbers are medians measured on the held-out
    # half of the evaluation set.
    assert calibrator.calibrate(0.031) < 0.10
    assert calibrator.calibrate(0.9705) > 0.99

    # And the measured equal-error point lands near the natural boundary.
    assert 0.05 < calibrator.calibrate(0.0443) < 0.35


def test_the_shipped_calibration_records_how_it_was_made():
    calibrator = ProbabilityCalibrator.load(
        resolve_path(DetectorConfig().calibration_path)
    )

    metadata = calibrator.metadata

    assert metadata["n_holdout"] > 30000
    assert metadata["held_out"]["ece_after"] < metadata["held_out"]["ece_before"] / 4
    assert metadata["at_half"]["detection_rate"] > 0.90
    assert metadata["at_half"]["false_alarm_rate"] < 0.02


def test_config_thresholds_are_reachable_on_the_calibrated_scale():
    """
    The thresholds grade a calibrated probability. If they were left on the raw
    scale they would be unreachable - a raw bonafide score never exceeds ~0.05.
    """

    config = DetectorConfig()

    assert 0.0 < config.suspicious_threshold < config.high_risk_threshold <= 1.0
    assert config.decision_threshold == config.high_risk_threshold
