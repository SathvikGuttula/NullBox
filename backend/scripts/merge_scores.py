"""
Pool per-utterance scores from several evaluation runs into one report.

Why this exists
---------------
The ASVspoof 2021 DF eval set is 611,829 utterances shipped as four ~8 GB
archives. Extracted in full it is roughly 40 GB, which does not comfortably
co-exist with a training cache on a free Colab runtime's local disk. The
workable pattern is: extract one part, score it, save scores, delete the audio,
repeat.

But EER is not an average of per-part EERs - it is a property of the pooled
score distribution, and averaging four of them gives a number that is simply
wrong. This script concatenates the ``scores.csv`` files that
``evaluate_model.py --save-scores`` writes and computes the metrics once over
everything.

    python backend/scripts/merge_scores.py \
        --scores models/eval_df_part00/scores.csv \
                 models/eval_df_part01/scores.csv \
                 models/eval_df_part02/scores.csv \
                 models/eval_df_part03/scores.csv \
        --output models/eval_df_full

Duplicate utterances (the same path scored twice) are dropped, so re-running a
part after a crash is safe.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.ml.metrics import (  # noqa: E402
    calculate_metrics,
    format_report,
    plot_curves,
    save_report,
)
from app.ml.progress import format_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pool scores.csv files and compute one metrics report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scores",
        type=Path,
        nargs="+",
        required=True,
        help="scores.csv files from evaluate_model.py --save-scores.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-spoof", type=float, default=0.05)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--label",
        type=str,
        default="pooled",
        help="Name used for the report and plot filenames.",
    )

    return parser.parse_args()


def read_scores(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"scores file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"path", "label", "spoof_score"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(
            f"{path} is missing columns {required - set(rows[0])}. "
            "Was it written by evaluate_model.py --save-scores?"
        )

    return rows


def main() -> None:
    args = parse_args()

    seen: set[str] = set()
    merged: list[dict] = []
    duplicates = 0

    print()
    for path in args.scores:
        rows = read_scores(path)

        fresh = 0
        for row in rows:
            if row["path"] in seen:
                duplicates += 1
                continue
            seen.add(row["path"])
            merged.append(row)
            fresh += 1

        print(f"  {path}  {format_count(len(rows))} rows ({fresh} new)")

    if not merged:
        raise SystemExit("no rows to score")

    if duplicates:
        print(f"  dropped {format_count(duplicates)} duplicate utterance(s)")

    labels = np.asarray([int(row["label"]) for row in merged])
    scores = np.asarray([float(row["spoof_score"]) for row in merged])
    attacks = [row.get("attack", "") for row in merged]

    print(
        f"\n  pooled: {format_count(len(merged))} utterances   "
        f"bonafide {(labels == 0).sum()}   spoof {(labels == 1).sum()}"
    )

    report = calculate_metrics(
        labels, scores, attacks=attacks, prior_spoof=args.prior_spoof
    )
    report["pooled_from"] = [str(p) for p in args.scores]
    report["duplicates_dropped"] = duplicates

    print()
    print(format_report(report))
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"  report  {save_report(report, args.output / 'report.json')}")

    combined = args.output / "scores.csv"
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "attack", "spoof_score"]
        )
        writer.writeheader()
        for row in merged:
            writer.writerow(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "attack": row.get("attack", ""),
                    "spoof_score": row["spoof_score"],
                }
            )
    print(f"  scores  {combined}")

    if not args.no_plots:
        for plot in plot_curves(labels, scores, args.output, prefix=args.label):
            print(f"  plot    {plot}")

    print()
    print(
        "  This is the pooled EER over every utterance scored. Do not average\n"
        "  per-part EERs instead - EER is a property of the whole score\n"
        "  distribution, and the average of four of them is a different number."
    )
    print()


if __name__ == "__main__":
    main()
