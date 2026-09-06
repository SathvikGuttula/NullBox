from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".ogg",
}


def find_audio_files(
    root: Path,
) -> list[Path]:

    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in AUDIO_EXTENSIONS
    ]


def classify_file(
    path: Path,
) -> str | None:

    name = path.name.lower()

    if "bonafide" in name:
        return "bonafide"

    if "spoof" in name:
        return "spoof"

    return None


def copy_dataset(
    source: Path,
    destination: Path,
):

    audio_files = find_audio_files(
        source
    )

    print(
        f"Found {len(audio_files)} audio files."
    )

    copied = {
        "bonafide": 0,
        "spoof": 0,
        "unknown": 0,
    }

    for audio_file in audio_files:

        label = classify_file(
            audio_file
        )

        if label is None:

            copied["unknown"] += 1

            continue

        target_dir = (
            destination / label
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = (
            target_dir
            / audio_file.name
        )

        shutil.copy2(
            audio_file,
            target,
        )

        copied[label] += 1

    print("\nDataset summary:")

    for key, value in copied.items():

        print(
            f"{key:10}: {value}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--destination",
        required=True,
    )

    args = parser.parse_args()

    source = Path(
        args.source
    ).resolve()

    destination = Path(
        args.destination
    ).resolve()

    if not source.exists():

        raise FileNotFoundError(
            f"Dataset not found: {source}"
        )

    copy_dataset(
        source,
        destination,
    )


if __name__ == "__main__":
    main()