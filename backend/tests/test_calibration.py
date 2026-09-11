import numpy as np
import pytest

from app.ml.calibration import ProbabilityCalibrator


def test_default_calibrator_is_the_identity():
    calibrator = ProbabilityCalibrator()
    assert calibrator.calibrate(0.73) == pytest.approx(0.73)


def test_temperature_actually_changes_the_probability():
    """
    The previous implementation stored a temperature and returned its input
    unchanged, so calibration was silently a no-op.
    """

    calibrator = ProbabilityCalibrator(temperature=2.0)
    calibrated = calibrator.calibrate(0.95)

    assert calibrated != pytest.approx(0.95)
    assert 0.5 < calibrated < 0.95


def test_temperature_above_one_softens_and_below_one_sharpens():
    assert ProbabilityCalibrator(temperature=3.0).calibrate(0.99) < 0.99
    assert ProbabilityCalibrator(temperature=0.4).calibrate(0.80) > 0.80


def test_half_is_a_fixed_point():
    for temperature in (0.2, 1.0, 5.0):
        assert ProbabilityCalibrator(temperature).calibrate(0.5) == pytest.approx(0.5)


def test_calibration_preserves_ranking():
    """
    Temperature scaling is monotone, so EER and ROC-AUC must be untouched.
    That is precisely why it is the safe calibration method to use here.
    """

    rng = np.random.default_rng(0)
    scores = rng.random(500)

    calibrated = ProbabilityCalibrator(temperature=2.7).calibrate_array(scores)

    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))


def test_fit_recovers_a_known_temperature():
    """
    Generate labels from logits divided by a known T, then check the fit
    finds it back.
    """

    rng = np.random.default_rng(42)

    true_temperature = 2.5
    logits = rng.normal(0, 3.0, 20000)

    true_probability = 1.0 / (1.0 + np.exp(-logits / true_temperature))
    labels = (rng.random(20000) < true_probability).astype(int)

    overconfident = 1.0 / (1.0 + np.exp(-logits))

    calibrator = ProbabilityCalibrator()
    fitted = calibrator.fit(overconfident, labels)

    assert fitted == pytest.approx(true_temperature, rel=0.15)


def test_fit_reduces_calibration_error():
    rng = np.random.default_rng(1)

    logits = rng.normal(0, 3.0, 20000)
    probability = 1.0 / (1.0 + np.exp(-logits / 3.0))
    labels = (rng.random(20000) < probability).astype(int)

    overconfident = 1.0 / (1.0 + np.exp(-logits))

    before = ProbabilityCalibrator.expected_calibration_error(overconfident, labels)

    calibrator = ProbabilityCalibrator()
    calibrator.fit(overconfident, labels)

    after = ProbabilityCalibrator.expected_calibration_error(
        calibrator.calibrate_array(overconfident), labels
    )

    assert after < before / 2


def test_save_and_load_roundtrip(tmp_path):
    calibrator = ProbabilityCalibrator(temperature=1.83)
    calibrator.fitted = True

    path = calibrator.save(tmp_path / "calibration.json")
    restored = ProbabilityCalibrator.load(path)

    assert restored.temperature == pytest.approx(1.83)
    assert restored.fitted


def test_extreme_probabilities_do_not_overflow():
    calibrator = ProbabilityCalibrator(temperature=0.1)

    assert 0.0 <= calibrator.calibrate(0.0) <= 1.0
    assert 0.0 <= calibrator.calibrate(1.0) <= 1.0


def test_fit_on_empty_input_raises():
    with pytest.raises(ValueError):
        ProbabilityCalibrator().fit([], [])


def test_hitting_the_search_bound_is_flagged():
    """
    An uninformative score set wants T -> infinity. The grid stops at its
    upper bound and reports that value, which looks like a fit and is not.
    Silence there would hide a broken model behind a plausible number.
    """

    import warnings

    calibrator = ProbabilityCalibrator()

    # Scores carry no information about the labels.
    rng = np.random.default_rng(2)
    scores = np.full(500, 0.5) + rng.normal(0, 1e-4, 500)
    labels = rng.integers(0, 2, 500)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator.fit(scores, labels)

    assert calibrator.hit_bound
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_a_good_fit_is_not_flagged():
    rng = np.random.default_rng(4)

    logits = rng.normal(0, 3.0, 20000)
    probability = 1.0 / (1.0 + np.exp(-logits / 2.0))
    labels = (rng.random(20000) < probability).astype(int)
    overconfident = 1.0 / (1.0 + np.exp(-logits))

    calibrator = ProbabilityCalibrator()
    calibrator.fit(overconfident, labels)

    assert not calibrator.hit_bound


def test_hit_bound_survives_a_save_load_roundtrip(tmp_path):
    calibrator = ProbabilityCalibrator(temperature=10.0)
    calibrator.fitted = True
    calibrator.hit_bound = True

    restored = ProbabilityCalibrator.load(calibrator.save(tmp_path / "c.json"))

    assert restored.hit_bound
