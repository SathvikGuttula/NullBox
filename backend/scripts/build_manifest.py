"""
Build a VoxShield manifest from ASVspoof 2021 DF metadata.

Expected metadata columns:
    0 = source
    1 = audio ID
    5 = label
    8 = attack

Labels:
    bonafide -> 0
    spoof    -> 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ASVspoof 2021 DF manifest."
    )

    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Directory containing DF .flac files.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to ASVspoof DF trial_metadata.txt.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio_root.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {args.audio_root}"
        )

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}"
        )

    # Find all available audio files.
    audio_files = {
        path.stem: path
        for path in args.audio_root.glob("*.flac")
    }

    print(f"Audio files found: {len(audio_files)}")

    entries = []

    with args.metadata.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:

        for line in metadata_file:

            line = line.strip()

            if not line:
                continue

            columns = line.split()

            if len(columns) < 9:
                continue

            audio_id = columns[1]
            label_text = columns[5]
            attack = columns[8]

            # Only include audio that actually exists.
            if audio_id not in audio_files:
                continue

            if label_text == "bonafide":
                label = 0
            elif label_text == "spoof":
                label = 1
            else:
                continue

            entries.append(
                {
                    "path": str(audio_files[audio_id]),
                    "label": label,
                    "split": "test",
                    "attack": attack,
                }
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "path",
                "label",
                "split",
                "attack",
            ],
        )

        writer.writeheader()
        writer.writerows(entries)

    print(f"Manifest written: {args.output}")
    print(f"Manifest rows: {len(entries)}")

    spoof = sum(
        entry["label"] == 1
        for entry in entries
    )

    bonafide = sum(
        entry["label"] == 0
        for entry in entries
    )

    print(f"Spoof: {spoof}")
    print(f"Bonafide: {bonafide}")


if __name__ == "__main__":
    main()