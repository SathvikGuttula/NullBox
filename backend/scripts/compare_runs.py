"""
Put two or more evaluation reports side by side.

Built for the experiment this project is actually running: train once with the
voice-conversion attacks held out of training, once with them included, and show
what changes. A single EER hides that; a per-attack column pair makes it
obvious.

    python backend/scripts/compare_runs.py \
        --reports models/eval_heldout/report.json models/eval_full/report.json \
        --names "A05/A06 held out" "full train set"

Prints a headline table, a per-attack table, and - when the attack families are
recognised - a TTS vs VC summary, which is usually the sentence you actually
want on the slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ASVspoof 2019 LA synthesis families. A05/A06 are the only voice-conversion
# attacks in train/dev, and A17-A19 are the voice-conversion attacks in eval -
# which is why holding out A05/A06 removes the entire VC category from training
# and the model then fails on A17-A19 specifically.
ATTACK_FAMILY = {
    "A01": "TTS", "A02": "TTS", "A03": "TTS", "A04": "TTS",
    "A05": "VC", "A06": "VC",
    "A07": "TTS", "A08": "TTS", "A09": "TTS", "A10": "TTS",
    "A11": "TTS", "A12": "TTS",
    "A13": "TTS_VC", "A14": "TTS_VC", "A15": "TTS_VC",
    "A16": "TTS",
    "A17": "VC", "A18": "VC", "A19": "VC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluation reports side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Also write the comparison as markdown.")
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value) -> str:
    if value is None:
        return "    -"
    return f"{value * 100:6.2f}%"


def main() -> None:
    args = parse_args()

    reports = [load(p) for p in args.reports]
    names = args.names or [p.parent.name for p in args.reports]

    if len(names) != len(reports):
        raise SystemExit(f"{len(reports)} reports but {len(names)} names")

    width = max(len(n) for n in names) + 2
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    # -- headline ----------------------------------------------------------

    emit()
    emit("=" * (22 + width * len(names)))
    emit("  HEADLINE".ljust(22) + "".join(n.rjust(width) for n in names))
    emit("=" * (22 + width * len(names)))

    for label, key, fmt in [
        ("EER", "eer", lambda v: percent(v)),
        ("ROC-AUC", "roc_auc", lambda v: f"{v:6.4f}" if v is not None else "    -"),
        ("normalised minDCF", "min_dcf", lambda v: f"{v:6.4f}" if v is not None else "    -"),
    ]:
        emit(f"  {label:<20}" + "".join(fmt(r.get(key)).rjust(width) for r in reports))

    emit()
    emit("  at the EER threshold")
    for label, key in [
        ("false alarm", "false_alarm_rate"),
        ("miss", "miss_rate"),
    ]:
        emit(
            f"    {label:<18}"
            + "".join(
                percent((r.get("at_eer_threshold") or {}).get(key)).rjust(width)
                for r in reports
            )
        )

    # -- per attack --------------------------------------------------------

    breakdowns = [r.get("per_attack") or {} for r in reports]
    attacks = sorted({a for b in breakdowns for a in b if a != "bonafide"})

    if attacks:
        emit()
        emit("=" * (22 + width * len(names)))
        emit("  MISS RATE BY ATTACK".ljust(22) + "".join(n.rjust(width) for n in names))
        emit("=" * (22 + width * len(names)))

        for attack in attacks:
            family = ATTACK_FAMILY.get(attack, "")
            label = f"  {attack} {family:<7}"
            emit(
                label.ljust(22)
                + "".join(
                    percent(b.get(attack, {}).get("miss_rate")).rjust(width)
                    for b in breakdowns
                )
            )

        emit("  " + "-" * (20 + width * len(names)))
        emit(
            "  bonafide (FA)".ljust(22)
            + "".join(
                percent(b.get("bonafide", {}).get("false_alarm_rate")).rjust(width)
                for b in breakdowns
            )
        )

        # -- family summary ------------------------------------------------

        families = sorted(
            {ATTACK_FAMILY[a] for a in attacks if a in ATTACK_FAMILY}
        )

        if families:
            emit()
            emit("  BY SYNTHESIS FAMILY  (mean miss rate)")
            for family in families:
                members = [a for a in attacks if ATTACK_FAMILY.get(a) == family]
                cells = []
                for b in breakdowns:
                    rates = [
                        b[a]["miss_rate"] for a in members
                        if a in b and "miss_rate" in b[a]
                    ]
                    cells.append(
                        percent(sum(rates) / len(rates)) if rates else "    -"
                    )
                emit(f"    {family:<18}" + "".join(c.rjust(width) for c in cells))

            emit()
            emit(
                "  A05/A06 are the ONLY voice-conversion attacks in ASVspoof 2019 LA\n"
                "  train and dev; A17-A19 are the voice-conversion attacks in eval.\n"
                "  Holding A05/A06 out therefore removes the entire VC family from\n"
                "  training, and the VC row above is where that shows up."
            )

    emit()
    emit("=" * (22 + width * len(names)))
    emit()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8"
        )
        print(f"  written to {args.output}\n")


if __name__ == "__main__":
    main()
