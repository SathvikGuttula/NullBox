"""
Zip the VoxShield source for upload to Google Drive.

The Colab notebook prefers `git clone`, but the repository is private, so a
plain clone 404s and a token has to be pasted into a notebook cell. This is the
no-token alternative: run it, drop the resulting zip into your
`VoxShield_Dataset` folder on Drive, and the notebook finds it automatically.

    python backend/scripts/package_code.py

Only source is included - no datasets, no checkpoints, no virtualenv, no build
output. The result is well under a megabyte.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INCLUDE = [
    "backend/app",
    "backend/scripts",
    "backend/tests",
    "backend/requirements.txt",
    "backend/pytest.ini",
    "backend/.env.example",
    "notebooks",
    "TRAINING.md",
    "README.md",
    "docker-compose.yml",
]

EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
    ".next",
    ".git",
}

EXCLUDE_SUFFIXES = {".pt", ".pth", ".onnx", ".pkl", ".joblib", ".pyc", ".f16"}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "voxshield_code.zip",
        help="Where to write the zip.",
    )
    args = parser.parse_args()

    files: list[Path] = []

    for entry in INCLUDE:
        source = REPO_ROOT / entry

        if not source.exists():
            print(f"  skipped (missing): {entry}")
            continue

        if source.is_file():
            files.append(source)
        else:
            files.extend(p for p in source.rglob("*") if p.is_file())

    files = [f for f in files if should_include(f)]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(REPO_ROOT).as_posix())

    size = args.output.stat().st_size

    print(f"\n  wrote {args.output}")
    print(f"  {len(files)} files, {size / 1024:.0f} KB")
    print(
        "\n  Upload this to your VoxShield_Dataset folder on Google Drive.\n"
        "  The Colab notebook picks it up automatically when the clone fails.\n"
    )


if __name__ == "__main__":
    main()
