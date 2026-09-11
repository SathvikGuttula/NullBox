"""
Inspect and verify VoxShield manifests before committing to a training run.

Run this after building manifests and before training. It catches the failures
that otherwise surface forty minutes into an epoch:

    * manifest rows pointing at files that no longer exist
    * unreadable or zero-length audio
    * a sample rate other than 16 kHz
    * a class balance that will dominate the loss
    * clips far shorter than the training window
    * the same audio file appearing in two different splits
    * the same speaker appearing in two different splits

The last two matter most. Speaker leakage between train and test inflates every
metric and is invisible in the numbers themselves - the model looks excellent
right up until it meets a voice it has not memorised.

    python backend/scripts/dataset_stats.py --manifest datasets/manifests/train.csv
    python backend/scripts/dataset_stats.py --all --check-audio
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.ml.dataset import AudioSample, load_manifest  # noqa: E402
from app.ml.progress import (  # noqa: E402
    IterationProgress,
    format_count,
    format_duration,
)


DEFAULT_MANIFESTS = [
    Path("datasets/manifests/train.csv"),
    Path("datasets/manifests/validation.csv"),
    Path("datasets/manifests/test.csv"),
]


# ---------------------------------------------------------------------------
# per-manifest inspection
# ---------------------------------------------------------------------------


def summarise(samples: list[AudioSample], name: str) -> dict:
    spoof = sum(1 for s in samples if s.label == 1)
    bonafide = len(samples) - spoof

    attacks = Counter(s.attack for s in samples if s.attack)
    speakers = {s.speaker for s in samples if s.speaker}

    print(f"\n[{name}]")
    print(f"  rows        {format_count(len(samples))}")
    print(
        f"  bonafide    {bonafide:>8}  ({bonafide / max(len(samples), 1) * 100:5.1f}%)"
    )
    print(f"  spoof       {spoof:>8}  ({spoof / max(len(samples), 1) * 100:5.1f}%)")

    if attacks:
        print(f"  attacks     {len(attacks)} types")
        for attack, count in sorted(attacks.items()):
            print(f"                {attack}  {count}")

    if speakers:
        print(f"  speakers    {len(speakers)}")

    ratio = spoof / max(len(samples), 1)
    if ratio > 0.8 or ratio < 0.2:
        print(
            f"  NOTE: {ratio * 100:.0f}% of this split is one class. Train with\n"
            f"        class-balanced sampling (on by default) or the model will\n"
            f"        learn the prior instead of the signal."
        )

    return {
        "rows": len(samples),
        "bonafide": bonafide,
        "spoof": spoof,
        "attacks": dict(attacks),
        "speakers": speakers,
    }


def check_audio(samples: list[AudioSample], name: str, limit: int | None) -> dict:
    """Open every file (or a sample of them) and report the real properties."""

    subset = samples if limit is None else samples[:limit]

    durations: list[float] = []
    rates: Counter = Counter()
    problems: Counter = Counter()
    bad_paths: list[str] = []

    progress = IterationProgress(
        total=len(subset),
        label=f"checking {name}",
        log_every=max(len(subset) // 10, 1),
    )

    for sample in subset:
        path = Path(sample.path)

        if not path.exists():
            problems["missing"] += 1
            if len(bad_paths) < 5:
                bad_paths.append(str(path))
            progress.update(1)
            continue

        try:
            info = sf.info(str(path))
        except Exception:
            problems["unreadable"] += 1
            if len(bad_paths) < 5:
                bad_paths.append(str(path))
            progress.update(1)
            continue

        if info.frames == 0:
            problems["empty"] += 1
        else:
            durations.append(info.duration)

        rates[info.samplerate] += 1
        progress.update(1)

    progress.close()

    print(f"  checked     {format_count(len(subset))} files")

    if durations:
        array = np.asarray(durations)
        print(
            f"  duration    mean {array.mean():.2f}s   "
            f"median {np.median(array):.2f}s   "
            f"min {array.min():.2f}s   max {array.max():.2f}s"
        )
        print(f"  total audio {format_duration(float(array.sum()))}")

        short = int((array < 4.0).sum())
        if short:
            print(
                f"  {short} clip(s) ({short / len(array) * 100:.1f}%) are under the\n"
                f"        4 s training window - these get tiled up to length."
            )

    for rate, count in rates.items():
        marker = "" if rate == 16000 else "   <-- not 16 kHz, will be resampled"
        print(f"  {rate} Hz   {count}{marker}")

    for problem, count in problems.items():
        print(f"  PROBLEM     {count} {problem}")

    for path in bad_paths:
        print(f"                {path}")

    return {"durations": durations, "rates": dict(rates), "problems": dict(problems)}


# ---------------------------------------------------------------------------
# cross-split leakage
# ---------------------------------------------------------------------------


def check_leakage(splits: dict[str, list[AudioSample]]) -> bool:
    """Return True when the splits are clean."""

    print("\n[leakage]")

    names = list(splits)
    clean = True

    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared_files = {s.path for s in splits[first]} & {
                s.path for s in splits[second]
            }

            if shared_files:
                clean = False
                print(
                    f"  FILE LEAK   {len(shared_files)} audio file(s) appear in both "
                    f"{first} and {second}"
                )
                for path in list(shared_files)[:3]:
                    print(f"                {path}")

            speakers_first = {s.speaker for s in splits[first] if s.speaker}
            speakers_second = {s.speaker for s in splits[second] if s.speaker}
            shared_speakers = speakers_first & speakers_second

            if shared_speakers:
                clean = False
                print(
                    f"  SPEAKER LEAK  {len(shared_speakers)} speaker(s) appear in both "
                    f"{first} and {second}"
                )
                print(f"                e.g. {sorted(shared_speakers)[:5]}")

    if clean:
        print("  no shared files or speakers between splits")

    # Attack overlap is not leakage - it is a design choice - but it decides
    # what the eval number actually measures.
    train_attacks = {s.attack for s in splits.get("train", []) if s.attack}
    test_attacks = {s.attack for s in splits.get("test", []) if s.attack}

    if train_attacks and test_attacks:
        unseen = test_attacks - train_attacks
        shared = test_attacks & train_attacks

        print(f"\n[attack coverage]")
        print(f"  train attacks   {sorted(train_attacks)}")
        print(f"  test attacks    {sorted(test_attacks)}")
        print(f"  unseen in test  {sorted(unseen) or 'NONE'}")

        if not unseen:
            print(
                "  WARNING: every test attack was seen in training. The eval EER\n"
                "           then measures memorisation, not generalisation. Report\n"
                "           it as a seen-attack number and get an unseen-attack set."
            )
        else:
            print(
                f"  {len(unseen)} of {len(test_attacks)} test attacks are unseen "
                f"({len(shared)} shared) - this is the number worth reporting."
            )

    return clean


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and verify VoxShield manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--all", action="store_true", help="Inspect all three manifests.")
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Open each file to verify it exists and read its real properties.",
    )
    parser.add_argument(
        "--check-limit",
        type=int,
        default=2000,
        help="How many files to open per split (0 = all). Opening 120k files "
             "takes a few minutes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifests = (
        DEFAULT_MANIFESTS
        if args.all or args.manifest is None
        else [args.manifest]
    )

    splits: dict[str, list[AudioSample]] = {}

    for manifest in manifests:
        if not manifest.exists() or manifest.stat().st_size == 0:
            print(f"\n[{manifest.stem}]  missing or empty - {manifest}")
            continue

        samples = load_manifest(manifest)
        splits[manifest.stem] = samples

        summarise(samples, manifest.stem)

        if args.check_audio:
            check_audio(
                samples,
                manifest.stem,
                None if args.check_limit == 0 else args.check_limit,
            )

    if len(splits) > 1:
        check_leakage(splits)

    if not splits:
        print(
            "\nNo manifests found. Build them first:\n"
            "  python backend/scripts/build_manifest.py "
            "--asvspoof2019-la datasets/raw/ASVspoof2019/LA\n"
        )

    print()


if __name__ == "__main__":
    main()
