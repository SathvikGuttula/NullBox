from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import soundfile as sf


def analyze(
    root: Path,
):

    files = list(
        root.rglob("*.flac")
    )

    files += list(
        root.rglob("*.wav")
    )

    labels = Counter()

    durations = []

    for path in files:

        label = (
            "bonafide"
            if "bonafide"
            in str(path).lower()
            else "spoof"
        )

        labels[label] += 1

        try:

            info = sf.info(
                str(path)
            )

            durations.append(
                info.duration
            )

        except Exception as exc:

            print(
                f"Could not read {path}: "
                f"{exc}"
            )

    print(
        "\n========== DATASET =========="
    )

    print(
        f"Total files: {len(files)}"
    )

    for label, count in labels.items():

        print(
            f"{label}: {count}"
        )

    if durations:

        total = sum(
            durations
        )

        average = (
            total / len(durations)
        )

        print(
            f"Total hours: "
            f"{total / 3600:.2f}"
        )

        print(
            f"Average duration: "
            f"{average:.2f}s"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    args = parser.parse_args()

    analyze(
        Path(
            args.root
        )
    )


if __name__ == "__main__":
    main()