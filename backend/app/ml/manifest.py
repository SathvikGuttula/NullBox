from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManifestEntry:

    path: str
    label: int
    split: str
    attack: str | None = None


class ManifestWriter:

    HEADER = [
        "path",
        "label",
        "split",
        "attack",
    ]

    @staticmethod
    def write(
        entries: list[ManifestEntry],
        output_path: str | Path,
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=ManifestWriter.HEADER,
            )

            writer.writeheader()

            for entry in entries:

                writer.writerow(
                    {
                        "path": entry.path,
                        "label": entry.label,
                        "split": entry.split,
                        "attack": (
                            entry.attack
                            or ""
                        ),
                    }
                )