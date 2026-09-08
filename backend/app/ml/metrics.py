"""
Evaluation metrics for the VoxShield anti-spoof detector.

Score / label convention used everywhere in this file
-----------------------------------------------------

    label 1 = spoof      (the attack, the positive class)
    label 0 = bonafide   (a genuine human caller)

    score   = P(spoof), so a HIGH score means "this is probably synthetic".

Two error rates matter and they are not interchangeable:

    false_alarm_rate  P(flag spoof | bonafide)
                      A real customer gets challenged. This is the usability
                      cost, and it is the one product owners care about.

    miss_rate         P(pass as bonafide | spoof)
                      An attack gets through. This is the security cost.

The EER is the operating point where those two are equal. It is the standard
single-number summary in the ASVspoof literature and is threshold-free, which
is why accuracy alone must never be reported for this task: on a set that is
90% spoof, a model that always says "spoof" scores 90% accuracy and is useless.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# core threshold-free metrics
# ---------------------------------------------------------------------------


@dataclass
class DetCurve:
    """Points of the detection-error-tradeoff curve."""

    thresholds: list[float] = field(default_factory=list)
    false_alarm_rates: list[float] = field(default_factory=list)
    miss_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def det_curve(labels, scores) -> DetCurve:
    """
    Sweep every threshold and record (false alarm, miss).

    Built on sklearn's roc_curve so the threshold grid is exactly the set of
    distinct scores - no arbitrary binning, no interpolation error.
    """

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    # roc_curve with pos_label=1 (spoof) gives fpr = P(score>=t | bonafide),
    # which is precisely the false alarm rate, and tpr = 1 - miss rate.
    false_alarm, true_positive, thresholds = roc_curve(labels, scores, pos_label=1)

    # sklearn prepends an infinite threshold so the curve starts at (0, 0).
    # That point is meaningful for the curve but poisons any arithmetic on the
    # threshold - interpolating between inf and a finite value yields NaN, and
    # a perfectly separable system crosses at exactly that first point. Replace
    # it with the finite threshold that has identical behaviour: just above the
    # largest observed score, so nothing is classified positive.
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.size and not np.isfinite(thresholds[0]):
        largest = float(np.max(scores)) if scores.size else 1.0
        thresholds[0] = np.nextafter(largest, np.inf)

    return DetCurve(
        thresholds=[float(t) for t in thresholds],
        false_alarm_rates=[float(x) for x in false_alarm],
        miss_rates=[float(1.0 - x) for x in true_positive],
    )


def equal_error_rate(labels, scores) -> tuple[float, float]:
    """
    Return ``(eer, threshold)``.

    The EER is where the false-alarm and miss curves cross. The crossing
    usually falls between two threshold samples, so we linearly interpolate
    rather than snapping to the nearest grid point.
    """

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    curve = det_curve(labels, scores)

    far = np.asarray(curve.false_alarm_rates)
    frr = np.asarray(curve.miss_rates)
    thresholds = np.asarray(curve.thresholds)

    difference = far - frr

    # Index of the last point where false alarms still exceed misses.
    crossing = np.nanargmin(np.abs(difference))

    # Linear interpolation between the bracketing points when one exists.
    sign_change = np.where(np.diff(np.sign(difference)) != 0)[0]

    if sign_change.size:
        i = int(sign_change[0])
        d0, d1 = difference[i], difference[i + 1]
        if d1 != d0:
            weight = d0 / (d0 - d1)
            eer = far[i] + weight * (far[i + 1] - far[i])
            threshold = thresholds[i] + weight * (thresholds[i + 1] - thresholds[i])
            return float(eer), float(threshold)

    return float((far[crossing] + frr[crossing]) / 2.0), float(thresholds[crossing])


def minimum_dcf(
    labels,
    scores,
    prior_spoof: float = 0.05,
    cost_miss: float = 1.0,
    cost_false_alarm: float = 1.0,
) -> tuple[float, float]:
    """
    Minimum normalised detection cost, ``(min_dcf, threshold)``.

    This is the NIST-style normalised DCF:

        DCF(t) = C_miss * P_spoof * miss(t)
               + C_fa   * (1 - P_spoof) * false_alarm(t)

    normalised by the cost of the best trivial system, so 1.0 means "no better
    than always guessing" and 0.0 is perfect.

    NOTE ON t-DCF: ASVspoof papers report *tandem* DCF (t-DCF), which folds in
    an ASV system's own scores on the same trials. VoxShield has no enrolled
    ASV subsystem yet, so t-DCF is not computable here and this normalised DCF
    is reported instead. Do not present this number as t-DCF - they are not
    the same quantity and reviewers will notice.

    The default ``prior_spoof`` of 0.05 encodes "spoofing attempts are rare",
    which is the realistic operating assumption for a fraud line. Change it
    deliberately, and always report the prior alongside the number.
    """

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    curve = det_curve(labels, scores)

    far = np.asarray(curve.false_alarm_rates)
    frr = np.asarray(curve.miss_rates)
    thresholds = np.asarray(curve.thresholds)

    costs = (
        cost_miss * prior_spoof * frr
        + cost_false_alarm * (1.0 - prior_spoof) * far
    )

    default_cost = min(
        cost_miss * prior_spoof,
        cost_false_alarm * (1.0 - prior_spoof),
    )

    normalised = costs / default_cost if default_cost > 0 else costs

    best = int(np.nanargmin(normalised))

    return float(normalised[best]), float(thresholds[best])


def threshold_at_false_alarm(labels, scores, target_false_alarm: float) -> float:
    """
    Lowest threshold whose false-alarm rate stays under ``target``.

    This is how a production threshold should actually be picked: decide how
    many genuine callers you are willing to inconvenience, then accept
    whatever miss rate that buys you.
    """

    curve = det_curve(labels, scores)

    far = np.asarray(curve.false_alarm_rates)
    thresholds = np.asarray(curve.thresholds)

    acceptable = np.where(far <= target_false_alarm)[0]

    if acceptable.size == 0:
        return float(thresholds[0])

    return float(thresholds[int(acceptable[0])])


# ---------------------------------------------------------------------------
# threshold-dependent metrics
# ---------------------------------------------------------------------------


def error_rates_at(labels, scores, threshold: float) -> dict:
    """Confusion-derived rates at one specific operating point."""

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    predictions = (scores >= threshold).astype(int)

    spoof = labels == 1
    bonafide = labels == 0

    miss = float((predictions[spoof] == 0).mean()) if spoof.any() else float("nan")
    false_alarm = (
        float((predictions[bonafide] == 1).mean()) if bonafide.any() else float("nan")
    )

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_alarm_rate": false_alarm,
        "miss_rate": miss,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_legend": {
            "rows": "true [bonafide, spoof]",
            "cols": "predicted [bonafide, spoof]",
        },
    }


# ---------------------------------------------------------------------------
# the full report
# ---------------------------------------------------------------------------


def calculate_metrics(
    labels,
    probabilities,
    attacks: list[str] | None = None,
    prior_spoof: float = 0.05,
) -> dict:
    """
    Full evaluation report.

    ``attacks`` is the per-sample attack id from the manifest (``A07``,
    ``A08``, ... for ASVspoof). When supplied, a per-attack miss rate is
    included - which is the breakdown that actually tells you whether the
    model generalises or has merely memorised two easy vocoders.
    """

    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(probabilities, dtype=np.float64)

    if labels_np.size == 0:
        raise ValueError("no samples to score")

    if labels_np.shape != scores_np.shape:
        raise ValueError(
            f"labels {labels_np.shape} and probabilities {scores_np.shape} differ"
        )

    both_classes = len(np.unique(labels_np)) == 2

    report: dict = {
        "samples": int(labels_np.size),
        "bonafide": int((labels_np == 0).sum()),
        "spoof": int((labels_np == 1).sum()),
        "prior_spoof_used": prior_spoof,
    }

    # --- threshold-free ---------------------------------------------------

    if both_classes:
        eer, eer_threshold = equal_error_rate(labels_np, scores_np)
        dcf, dcf_threshold = minimum_dcf(labels_np, scores_np, prior_spoof=prior_spoof)

        report["eer"] = eer
        report["eer_percent"] = eer * 100.0
        report["eer_threshold"] = eer_threshold
        report["min_dcf"] = dcf
        report["min_dcf_threshold"] = dcf_threshold
        report["roc_auc"] = float(roc_auc_score(labels_np, scores_np))
    else:
        report["eer"] = None
        report["eer_percent"] = None
        report["eer_threshold"] = None
        report["min_dcf"] = None
        report["min_dcf_threshold"] = None
        report["roc_auc"] = None
        report["warning"] = "only one class present - threshold-free metrics undefined"

    # --- operating points -------------------------------------------------

    report["at_threshold_0.5"] = error_rates_at(labels_np, scores_np, 0.5)

    if both_classes:
        report["at_eer_threshold"] = error_rates_at(
            labels_np, scores_np, report["eer_threshold"]
        )
        report["at_min_dcf_threshold"] = error_rates_at(
            labels_np, scores_np, report["min_dcf_threshold"]
        )

        # A usability-first operating point: at most 1% of genuine callers
        # get challenged. Report what that costs in missed attacks.
        low_friction = threshold_at_false_alarm(labels_np, scores_np, 0.01)
        report["at_1pct_false_alarm"] = error_rates_at(
            labels_np, scores_np, low_friction
        )

    # --- backwards-compatible flat keys -----------------------------------
    # Older code and the existing tests read these off the top level.

    flat = report["at_threshold_0.5"]
    report["accuracy"] = flat["accuracy"]
    report["f1"] = flat["f1"]
    report["confusion_matrix"] = flat["confusion_matrix"]

    # --- per-attack breakdown ---------------------------------------------

    if attacks is not None:
        report["per_attack"] = per_attack_breakdown(
            labels_np,
            scores_np,
            attacks,
            threshold=report.get("eer_threshold") or 0.5,
        )

    return report


def per_attack_breakdown(
    labels,
    scores,
    attacks: list[str],
    threshold: float,
) -> dict:
    """
    Miss rate for each spoofing attack id, at a fixed threshold.

    Bonafide samples are pooled into a single ``bonafide`` row carrying the
    false-alarm rate, because they have no attack id.
    """

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    if len(attacks) != labels.size:
        raise ValueError("attacks list length does not match number of samples")

    predictions = (scores >= threshold).astype(int)

    grouped: dict[str, list[int]] = defaultdict(list)

    for index, (label, attack) in enumerate(zip(labels, attacks)):
        key = "bonafide" if label == 0 else (attack or "unknown")
        grouped[key].append(index)

    breakdown: dict[str, dict] = {}

    for key in sorted(grouped):
        indices = np.asarray(grouped[key])
        group_labels = labels[indices]
        group_predictions = predictions[indices]
        group_scores = scores[indices]

        if key == "bonafide":
            rate = float((group_predictions == 1).mean())
            rate_name = "false_alarm_rate"
        else:
            rate = float((group_predictions == 0).mean())
            rate_name = "miss_rate"

        breakdown[key] = {
            "samples": int(indices.size),
            rate_name: rate,
            "mean_spoof_score": float(group_scores.mean()),
            "label": int(group_labels[0]) if group_labels.size else None,
        }

    return breakdown


# ---------------------------------------------------------------------------
# persistence and plotting
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    """Render the report as an aligned block for the terminal."""

    lines = []
    lines.append("=" * 70)
    lines.append("  VoxShield anti-spoof evaluation")
    lines.append("=" * 70)
    lines.append(
        f"  samples {report['samples']}"
        f"   bonafide {report['bonafide']}"
        f"   spoof {report['spoof']}"
    )
    lines.append("-" * 70)

    if report.get("eer") is not None and math.isfinite(report["eer"]):
        lines.append("  THRESHOLD-FREE  (the numbers to quote)")
        lines.append(f"    EER              {report['eer'] * 100:8.3f} %")
        lines.append(f"    ROC-AUC          {report['roc_auc']:8.4f}")
        lines.append(
            f"    min DCF          {report['min_dcf']:8.4f}"
            f"   (prior_spoof={report['prior_spoof_used']})"
        )
        lines.append("-" * 70)

    for title, key in (
        ("AT THRESHOLD 0.5", "at_threshold_0.5"),
        ("AT EER THRESHOLD", "at_eer_threshold"),
        ("AT 1% FALSE ALARM  (usability-first operating point)", "at_1pct_false_alarm"),
    ):
        block = report.get(key)
        if not block:
            continue
        lines.append(f"  {title}   t={block['threshold']:.4f}")
        lines.append(
            f"    accuracy {block['accuracy'] * 100:6.2f} %"
            f"    precision {block['precision'] * 100:6.2f} %"
            f"    recall {block['recall'] * 100:6.2f} %"
            f"    F1 {block['f1']:.4f}"
        )
        lines.append(
            f"    false alarm (genuine flagged) {block['false_alarm_rate'] * 100:6.2f} %"
            f"    miss (attack accepted) {block['miss_rate'] * 100:6.2f} %"
        )
        lines.append("")

    per_attack = report.get("per_attack")
    if per_attack:
        lines.append("-" * 70)
        lines.append("  PER-ATTACK BREAKDOWN")
        lines.append(f"    {'attack':<14}{'n':>8}{'rate':>10}{'mean score':>13}")
        for name, block in per_attack.items():
            rate = block.get("miss_rate", block.get("false_alarm_rate", float("nan")))
            lines.append(
                f"    {name:<14}{block['samples']:>8}"
                f"{rate * 100:>9.2f}%{block['mean_spoof_score']:>13.4f}"
            )

    lines.append("=" * 70)

    return "\n".join(lines)


def save_report(report: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def plot_curves(labels, scores, output_dir: Path | str, prefix: str = "eval") -> list[Path]:
    """
    Write ROC and DET plots. Returns the paths written.

    matplotlib is imported lazily so evaluation still works without it.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    from scipy.stats import norm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    curve = det_curve(labels, scores)
    far = np.asarray(curve.false_alarm_rates)
    frr = np.asarray(curve.miss_rates)

    written = []

    # ROC ------------------------------------------------------------------
    figure, axes = plt.subplots(figsize=(5.5, 5.5))
    axes.plot(far, 1 - frr, linewidth=2)
    axes.plot([0, 1], [0, 1], "--", linewidth=1, color="grey")
    axes.set_xlabel("false alarm rate  (genuine flagged as spoof)")
    axes.set_ylabel("true positive rate  (spoof detected)")
    axes.set_title(f"ROC - AUC {roc_auc_score(labels, scores):.4f}")
    axes.grid(alpha=0.3)
    roc_path = output_dir / f"{prefix}_roc.png"
    figure.tight_layout()
    figure.savefig(roc_path, dpi=140)
    plt.close(figure)
    written.append(roc_path)

    # DET on a normal-deviate scale, which is the conventional presentation.
    epsilon = 1e-6
    far_clipped = np.clip(far, epsilon, 1 - epsilon)
    frr_clipped = np.clip(frr, epsilon, 1 - epsilon)

    figure, axes = plt.subplots(figsize=(5.5, 5.5))
    axes.plot(norm.ppf(far_clipped), norm.ppf(frr_clipped), linewidth=2)

    ticks = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
    tick_locations = norm.ppf(ticks)
    tick_labels = [f"{t * 100:g}" for t in ticks]

    axes.set_xticks(tick_locations)
    axes.set_xticklabels(tick_labels)
    axes.set_yticks(tick_locations)
    axes.set_yticklabels(tick_labels)
    axes.set_xlim(norm.ppf(0.0005), norm.ppf(0.7))
    axes.set_ylim(norm.ppf(0.0005), norm.ppf(0.7))

    eer, _ = equal_error_rate(labels, scores)
    axes.plot(
        [norm.ppf(0.0005), norm.ppf(0.7)],
        [norm.ppf(0.0005), norm.ppf(0.7)],
        "--",
        linewidth=1,
        color="grey",
    )
    axes.set_xlabel("false alarm rate  (%)")
    axes.set_ylabel("miss rate  (%)")
    axes.set_title(f"DET - EER {eer * 100:.2f} %")
    axes.grid(alpha=0.3)

    det_path = output_dir / f"{prefix}_det.png"
    figure.tight_layout()
    figure.savefig(det_path, dpi=140)
    plt.close(figure)
    written.append(det_path)

    return written
