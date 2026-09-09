"""
Decode a manifest once into a memmapped waveform cache.

Why bother
----------
On this hardware the GPU is not the bottleneck for the first training stage -
libsndfile is. Decoding ~25k FLAC files costs the same several minutes every
single epoch, and with a frozen encoder that decode time can exceed the
forward pass. Caching pays for itself after the second epoch and makes epoch
time predictable, which is the whole point of the ETA display.

Cost
----
ASVspoof 2019 LA at 16 kHz float16 is roughly 2 bytes per sample:

    train       ~25.4k clips   ~24 h audio    ~2.8 GB
    validation  ~24.8k clips   ~23 h audio    ~2.7 GB
    eval        ~71.2k clips   ~67 h audio    ~7.7 GB

Disk is cheap here (195 GB free on C:). Use ``--max-seconds`` to cap per-clip
length if that changes - 6.0 still leaves room for random 4 s crops.

    python backend/scripts/cache_dataset.py \
        --manifest datasets/manifests/train.csv \
        --output datasets/cache/train

    python backend/scripts/cache_dataset.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.dataset import WaveformStore, describe, load_manifest  # noqa: E402
from app.ml.progress import (  # noqa: E402
    IterationProgress,
    PhaseTimer,
    format_count,
    format_duration,
)


# --all deliberately does not cache the test split. Train and validation are
# re-read once per epoch, so caching them pays for itself twice over; the
# 71k-clip eval set is read once at the very end, where streaming FLAC beats
# spending 11 GB of disk and several minutes of decoding up front.
DEFAULT_SPLITS = [
    ("datasets/manifests/train.csv", "datasets/cache/train"),
    ("datasets/manifests/validation.csv", "datasets/cache/validation"),
]

HELDOUT_SPLITS = [
    ("datasets/manifests/train_heldout.csv", "datasets/cache/train_heldout"),
    ("datasets/manifests/val_unseen.csv", "datasets/cache/val_unseen"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a memmapped waveform cache from a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Cache train, validation and test from datasets/manifests/.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=6.0,
        help="Cap stored length per clip. Must exceed the training window so "
             "random cropping still has room. 0 stores the full clip.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Decode in parallel. FLAC decoding is pure CPU and scales "
             "almost linearly, so on a many-core machine this turns minutes "
             "into seconds. 0 or 1 stays single-process. Try os.cpu_count().",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the cache directory already looks complete.",
    )

    return parser.parse_args()


def cache_one(
    manifest: Path,
    output: Path,
    sample_rate: int,
    max_seconds: float | None,
    limit: int | None,
    force: bool,
    workers: int = 0,
) -> None:
    print(f"\n[{manifest.stem}]")

    if not manifest.exists() or manifest.stat().st_size == 0:
        print(f"  skipped - {manifest} is missing or empty")
        return

    index_path = output / WaveformStore.INDEX_NAME

    if index_path.exists() and not force:
        store = WaveformStore(output)
        print(
            f"  already cached: {format_count(len(store))} clips, "
            f"{store.size_bytes / (1024 ** 3):.2f} GB  (use --force to rebuild)"
        )
        return

    samples = load_manifest(manifest, limit=limit)
    stats = describe(samples)

    print(
        f"  {format_count(stats['total'])} clips   "
        f"bonafide {stats['bonafide']}   spoof {stats['spoof']}"
    )

    estimated_gb = stats["total"] * (max_seconds or 4.0) * sample_rate * 2 / (1024 ** 3)
    print(f"  estimated cache size: ~{estimated_gb:.2f} GB")

    progress = IterationProgress(
        total=len(samples),
        label="decoding",
        log_every=max(len(samples) // 20, 1),
    )

    with PhaseTimer(f"building cache at {output}") as timer:
        store = WaveformStore.build(
            samples=samples,
            root=output,
            sample_rate=sample_rate,
            max_seconds=max_seconds or None,
            progress=progress,
            workers=workers,
        )

    progress.close()

    total_audio_seconds = float(store.lengths.sum()) / sample_rate

    print(
        f"  cached {format_count(len(store))} clips   "
        f"{store.size_bytes / (1024 ** 3):.2f} GB on disk   "
        f"{format_duration(total_audio_seconds)} of audio"
    )
    print(
        f"  decode throughput: "
        f"{len(store) / max(timer.seconds, 1e-9):.1f} clips/s"
    )


def main() -> None:
    args = parse_args()

    max_seconds = args.max_seconds if args.max_seconds and args.max_seconds > 0 else None

    if args.all:
        splits = list(DEFAULT_SPLITS)

        # Cache the attack-held-out pair too, when build_manifest made them.
        splits += [
            (manifest, output)
            for manifest, output in HELDOUT_SPLITS
            if Path(manifest).exists() and Path(manifest).stat().st_size > 0
        ]

        for manifest, output in splits:
            cache_one(
                Path(manifest),
                Path(output),
                args.sample_rate,
                max_seconds,
                args.limit,
                args.force,
                args.workers,
            )

        print(
            "\nTest split not cached on purpose - it is read once, so streaming\n"
            "beats spending ~11 GB of disk on it. Cache it explicitly if you\n"
            "plan repeated evaluation runs:\n"
            "  python backend/scripts/cache_dataset.py "
            "--manifest datasets/manifests/test.csv --output datasets/cache/test\n"
        )
        print(
            "Train with the cache:\n"
            "  python backend/scripts/train_model.py --cache-dir datasets/cache\n"
        )
        return

    if not args.manifest or not args.output:
        raise SystemExit("provide --manifest and --output, or --all")

    cache_one(
        args.manifest,
        args.output,
        args.sample_rate,
        max_seconds,
        args.limit,
        args.force,
        args.workers,
    )


if __name__ == "__main__":
    main()
