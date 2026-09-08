"""
Metric correctness.

These are the numbers the project will be judged on, so they are tested
against cases whose answer is known analytically rather than against whatever
the implementation happened to produce.
"""

import numpy as np
import pytest

from app.ml.metrics import (
    calculate_metrics,
    equal_error_rate,
    minimum_dcf,
    per_attack_breakdown,
    threshold_at_false_alarm,
)


def test_perfect_separation_gives_zero_eer():
    labels = [0] * 50 + [1] * 50
    scores = [0.1] * 50 + [0.9] * 50

    eer, threshold = equal_error_rate(labels, scores)

    assert eer == pytest.approx(0.0, abs=1e-9)
    assert 0.1 < threshold <= 0.9


def test_random_scores_give_eer_near_half():
    rng = np.random.default_rng(0)

    labels = np.array([0] * 2000 + [1] * 2000)
    scores = rng.random(4000)

    eer, _ = equal_error_rate(labels, scores)

    assert 0.45 < eer < 0.55


def test_eer_is_where_the_two_error_rates_meet():
    """
    Construct a case with a known crossing.

    Bonafide scores are uniform on [0, 1] and spoof scores are uniform on
    [0.5, 1.5]. The distributions overlap on [0.5, 1.0]; the analytic EER for
    this pair is 0.25.
    """

    rng = np.random.default_rng(7)

    bonafide = rng.uniform(0.0, 1.0, 20000)
    spoof = rng.uniform(0.5, 1.5, 20000)

    labels = np.concatenate([np.zeros(20000, int), np.ones(20000, int)])
    scores = np.concatenate([bonafide, spoof])

    eer, _ = equal_error_rate(labels, scores)

    assert eer == pytest.approx(0.25, abs=0.01)


def test_inverted_scores_give_eer_above_half():
    """A model that is systematically backwards must not look good."""

    labels = [0] * 50 + [1] * 50
    scores = [0.9] * 50 + [0.1] * 50

    eer, _ = equal_error_rate(labels, scores)

    assert eer == pytest.approx(1.0, abs=1e-9)


def test_min_dcf_is_zero_for_a_perfect_system():
    labels = [0] * 50 + [1] * 50
    scores = [0.0] * 50 + [1.0] * 50

    dcf, _ = minimum_dcf(labels, scores)

    assert dcf == pytest.approx(0.0, abs=1e-9)


def test_min_dcf_of_a_useless_system_is_one():
    """Normalised DCF caps at 1.0 - the cost of the best trivial decision."""

    rng = np.random.default_rng(3)

    labels = np.array([0] * 2000 + [1] * 2000)
    scores = rng.random(4000)

    dcf, _ = minimum_dcf(labels, scores)

    assert 0.8 <= dcf <= 1.0


def test_threshold_at_false_alarm_respects_the_budget():
    rng = np.random.default_rng(11)

    bonafide = rng.normal(0.2, 0.1, 5000).clip(0, 1)
    spoof = rng.normal(0.8, 0.1, 5000).clip(0, 1)

    labels = np.concatenate([np.zeros(5000, int), np.ones(5000, int)])
    scores = np.concatenate([bonafide, spoof])

    threshold = threshold_at_false_alarm(labels, scores, 0.01)

    achieved = float((bonafide >= threshold).mean())

    assert achieved <= 0.011


def test_accuracy_alone_is_exposed_as_misleading():
    """
    A 90%-spoof set, model always predicts spoof.

    Accuracy is 90% and the model is worthless - EER 50%, and every genuine
    caller is flagged. This is exactly the failure the report is designed to
    make visible.
    """

    labels = [0] * 100 + [1] * 900
    scores = [0.99] * 1000

    report = calculate_metrics(labels, scores)

    assert report["accuracy"] == pytest.approx(0.9)
    assert report["at_threshold_0.5"]["false_alarm_rate"] == pytest.approx(1.0)
    assert report["eer"] > 0.4


def test_report_has_every_required_metric():
    rng = np.random.default_rng(5)

    labels = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate(
        [rng.normal(0.3, 0.15, 500), rng.normal(0.7, 0.15, 500)]
    ).clip(0, 1)

    report = calculate_metrics(labels, scores)

    for key in (
        "eer",
        "eer_percent",
        "min_dcf",
        "roc_auc",
        "at_threshold_0.5",
        "at_eer_threshold",
        "at_1pct_false_alarm",
    ):
        assert key in report, f"missing {key}"

    block = report["at_eer_threshold"]
    assert block["false_alarm_rate"] == pytest.approx(block["miss_rate"], abs=0.05)


def test_per_attack_breakdown_separates_attacks():
    labels = [0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.95, 0.1, 0.05]
    attacks = ["", "", "A07", "A07", "A19", "A19"]

    breakdown = per_attack_breakdown(labels, scores, attacks, threshold=0.5)

    assert breakdown["A07"]["miss_rate"] == pytest.approx(0.0)
    assert breakdown["A19"]["miss_rate"] == pytest.approx(1.0)
    assert breakdown["bonafide"]["false_alarm_rate"] == pytest.approx(0.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        calculate_metrics([0, 1, 1], [0.5, 0.5])


def test_single_class_does_not_crash():
    report = calculate_metrics([1, 1, 1], [0.9, 0.8, 0.7])

    assert report["eer"] is None
    assert "warning" in report
