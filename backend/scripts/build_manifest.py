"""Build CSV manifests for dataset splits.

This module creates placeholder manifest files that can later be populated with
file paths and labels for training and evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset manifest CSV files.")
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/processed"), help="Root directory for processed splits.")
    parser.add_argument("--manifest-root", type=Path, default=Path("datasets/manifests"), help="Directory to write manifest CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root
    manifest_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        rows = []
        for label in ("bonafide", "spoof"):
            split_dir = args.processed_root / split / label
            if not split_dir.exists():
                continue
            for audio_file in sorted(split_dir.glob("**/*.*")):
                rows.append({"path": str(audio_file.relative_to(args.processed_root.parent)), "label": label})

        manifest_path = manifest_root / f"{split}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote manifest: {manifest_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
