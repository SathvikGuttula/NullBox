"""
Fit the anti-spoof detector's probability calibration and its operating points.

    python scripts/calibrate_detector.py --scores kaggle_run/fixed/scores.csv

Why this script exists
----------------------
The trained detector ranks well - 2.95% EER on the ASVspoof 2019 LA evaluation
set - but its raw probabilities are not probabilities. Bonafide logits sit at
-3.44 +/- 0.31 and spoof logits at +2.67 +/- 1.95, so the equal-error point
falls at a raw probability of 0.044. Anything that treats the raw number as
"chance this is synthetic" (a dashboard, or the fusion layer weighting it
against other branches) is therefore wrong by a wide margin: judging at 0.5
misses 12% of attacks.

Method
------
Platt scaling - ``sigmoid(scale * logit(p) + bias)`` - fitted by maximum
likelihood. The evaluation set is split in half, **stratified by (label,
attack)** so both halves contain all thirteen attack types; the fit sees one
half and every number reported here comes from the other. Calibration is
monotone, so EER and ROC-AUC are unchanged by construction and are reported
only as a check that nothing was accidentally reordered.

The fitted bias encodes the prior of the data it was fitted on, and the
evaluation set is 89.7% spoof, which no real call stream is. ``--prior``
re-expresses the calibration at a chosen base rate; the default of 0.5 is the
prior-independent reading and puts the decision boundary back at 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.calibration import ProbabilityCalibrator  # noqa: E402
from app.ml.config import resolve_path  # noqa: E402
from app.ml.metrics import equal_error_rate  # noqa: E402


def stratified_halves(frame: pd.DataFrame, seed: int) -> tuple:
    """
    Split into two halves that each contain every (label, attack) group.

    A plain random split would usually be fine at this size, but "usually
    fine" is how an attack type ends up entirely on one side and the held-out
    numbers quietly stop meaning what they claim to.
    """

    rng = np.random.default_rng(seed)
    fit_rows: list[int] = []
    holdout_rows: list[int] = []

    for _, group in frame.groupby(["label", "attack"], sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)
        half = len(indices) // 2
        fit_rows.extend(indices[:half].tolist())
        holdout_rows.extend(indices[half:].tolist())

    return np.array(fit_rows), np.array(holdout_rows)


def operating_points(probabilities, labels, budgets) -> dict:
    """
    Threshold, false-alarm rate and miss rate for each false-alarm budget.

    Reported in log-odds as well as probability: above a calibrated log-odds of
    roughly 36 the probability saturates to exactly 1.0 in float64, and two
    genuinely different operating points then print as the same threshold.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    bonafide = np.sort(probabilities[labels == 0])[::-1]
    spoof = probabilities[labels == 1]

    points = {}
    for budget in budgets:
        # The largest threshold whose false-alarm rate is still within budget.
        allowed = int(np.floor(budget * bonafide.size))
        threshold = (
            float(bonafide[allowed]) if allowed < bonafide.size else float(bonafide[-1])
        )

        false_alarm = float((probabilities[labels == 0] >= threshold).mean())
        miss = float((spoof < threshold).mean())

        points[f"fa_{budget:g}"] = {
            "false_alarm_budget": budget,
            "threshold": threshold,
            "false_alarm_rate": false_alarm,
            "miss_rate": miss,
            "detection_rate": 1.0 - miss,
        }

    return points


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit Platt calibration for the anti-spoof detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores",
        default="kaggle_run/fixed/scores.csv",
        help="CSV with columns label, spoof_score and (optionally) attack",
    )
    parser.add_argument(
        "--out",
        default="results/calibration/antispoof_calibration.json",
        help="where to write the calibration artifact",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="operating prior to express the calibration at",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    scores_path = resolve_path(args.scores)
    if not scores_path.exists():
        print(f"no score file at {scores_path}", file=sys.stderr)
        print(
            "Produce one with scripts/evaluate_model.py --save-scores, or point "
            "--scores at an existing run.",
            file=sys.stderr,
        )
        return 1

    frame = pd.read_csv(scores_path)

    missing = {"label", "spoof_score"} - set(frame.columns)
    if missing:
        print(f"{scores_path} is missing column(s): {sorted(missing)}", file=sys.stderr)
        return 1

    if "attack" not in frame.columns:
        frame["attack"] = "-"
    frame["attack"] = frame["attack"].fillna("-")

    labels = frame["label"].to_numpy(int)
    raw = frame["spoof_score"].to_numpy(float)

    fit_rows, holdout_rows = stratified_halves(frame, args.seed)
    fitted_prior = float(labels[fit_rows].mean())

    print(f"scores            {scores_path}")
    print(f"  {len(frame):,} utterances - {labels.sum():,} spoof, "
          f"{(labels == 0).sum():,} bonafide ({fitted_prior * 100:.1f}% spoof)")
    print(f"  attacks: {', '.join(sorted(set(frame['attack'][labels == 1])))}")
    print(f"  fit half {len(fit_rows):,}   held-out half {len(holdout_rows):,}")

    calibrator = ProbabilityCalibrator()
    scale, bias = calibrator.fit_affine(raw[fit_rows], labels[fit_rows])
    print(f"\nfit (on the fit half only)")
    print(f"  scale {scale:.6f}   bias {bias:+.6f}")

    holdout_raw = raw[holdout_rows]
    holdout_labels = labels[holdout_rows]
    holdout_calibrated = calibrator.calibrate_array(holdout_raw)

    ece_before = ProbabilityCalibrator.expected_calibration_error(
        holdout_raw, holdout_labels
    )
    ece_after = ProbabilityCalibrator.expected_calibration_error(
        holdout_calibrated, holdout_labels
    )
    brier_before = ProbabilityCalibrator.brier_score(holdout_raw, holdout_labels)
    brier_after = ProbabilityCalibrator.brier_score(holdout_calibrated, holdout_labels)

    print(f"\nheld-out half - never seen by the fit")
    print(f"  expected calibration error   {ece_before:.4f}  ->  {ece_after:.4f}")
    print(f"  Brier score                  {brier_before:.4f}  ->  {brier_after:.4f}")
    print(
        f"  miss at p >= 0.5             "
        f"{(holdout_raw[holdout_labels == 1] < 0.5).mean() * 100:.2f}%  ->  "
        f"{(holdout_calibrated[holdout_labels == 1] < 0.5).mean() * 100:.2f}%"
    )
    print(
        f"  false alarm at p >= 0.5      "
        f"{(holdout_raw[holdout_labels == 0] >= 0.5).mean() * 100:.2f}%  ->  "
        f"{(holdout_calibrated[holdout_labels == 0] >= 0.5).mean() * 100:.2f}%"
    )

    eer_before, _ = equal_error_rate(holdout_labels, holdout_raw)
    eer_after, _ = equal_error_rate(holdout_labels, holdout_calibrated)
    print(
        f"  EER                          {eer_before * 100:.3f}%  ->  "
        f"{eer_after * 100:.3f}%   (monotone, must be unchanged)"
    )

    if abs(eer_before - eer_after) > 1e-6:
        print(
            "\nWARNING: calibration changed the EER. It is monotone and cannot; "
            "something reordered the scores.",
            file=sys.stderr,
        )

    deployment = calibrator.with_prior(args.prior, fitted_prior=fitted_prior)
    deployed = deployment.calibrate_array(holdout_raw)

    print(f"\nre-expressed at prior {args.prior}")
    print(f"  scale {deployment.scale:.6f}   bias {deployment.bias:+.6f}")
    print(
        f"  bonafide median p {np.median(deployed[holdout_labels == 0]):.5f}   "
        f"spoof median p {np.median(deployed[holdout_labels == 1]):.5f}"
    )

    budgets = [0.05, 0.02, 0.01, 0.005, 0.002]
    points = operating_points(deployed, holdout_labels, budgets)

    print(f"\n  {'budget':>8} {'threshold':>11} {'false alarm':>12} {'miss':>8} {'detects':>9}")
    for point in points.values():
        print(
            f"  {point['false_alarm_budget']:>7.1%} {point['threshold']:>11.4f} "
            f"{point['false_alarm_rate']:>11.2%} {point['miss_rate']:>7.2%} "
            f"{point['detection_rate']:>8.2%}"
        )

    at_half = {
        "threshold": 0.5,
        "false_alarm_rate": float((deployed[holdout_labels == 0] >= 0.5).mean()),
        "miss_rate": float((deployed[holdout_labels == 1] < 0.5).mean()),
    }
    at_half["detection_rate"] = 1.0 - at_half["miss_rate"]
    print(
        f"\n  at the natural 0.5 boundary: false alarm "
        f"{at_half['false_alarm_rate']:.2%}, detects "
        f"{at_half['detection_rate']:.2%} of attacks"
    )

    per_attack = {}
    for attack in sorted(set(frame["attack"].to_numpy()[holdout_rows][holdout_labels == 1])):
        mask = (holdout_labels == 1) & (
            frame["attack"].to_numpy()[holdout_rows] == attack
        )
        per_attack[attack] = {
            "n": int(mask.sum()),
            "miss_rate_at_0.5": float((deployed[mask] < 0.5).mean()),
        }

    print("\n  per-attack miss rate at p >= 0.5")
    for attack, stats in per_attack.items():
        print(f"    {attack}  {stats['miss_rate_at_0.5']:>7.2%}   (n={stats['n']})")

    deployment.metadata = {
        "source_scores": str(scores_path.name),
        "n_total": int(len(frame)),
        "n_fit": int(len(fit_rows)),
        "n_holdout": int(len(holdout_rows)),
        "fitted_prior": fitted_prior,
        "prior": float(args.prior),
        "seed": int(args.seed),
        "held_out": {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "eer": float(eer_after),
        },
        "operating_points": points,
        "at_half": at_half,
        "per_attack_miss_at_half": per_attack,
        "note": (
            "Fitted on half of the ASVspoof 2019 LA evaluation set, reported on "
            "the other half. Calibration is monotone so EER is unchanged. "
            "Thresholds apply to the CALIBRATED probability, not the raw score."
        ),
    }

    out_path = resolve_path(args.out)
    deployment.save(out_path)
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
