"""
Build VoxShield manifests from official ASVspoof protocol files.

Why the columns are auto-detected
---------------------------------
The protocol layouts differ between corpora:

    ASVspoof 2019 LA   LA_0079 LA_T_1138215 - - bonafide
                       speaker utt_id       - attack key

    ASVspoof 2021 DF   LA_0023 DF_E_2000011 nocodec asvspoof A11 spoof ...
                       (more columns, different order)

Hardcoding "label is column 5, attack is column 8" works for exactly one of
these and fails silently on the other - it produces a manifest with zero rows,
or worse, one where every label is wrong. So instead this script finds the
label column by looking for the one whose values are all bonafide/spoof, finds
the id column by matching against the filenames actually on disk, and finds the
attack column by its A-prefixed codes.

Labels follow the project convention throughout:  0 = bonafide, 1 = spoof.

Usage
-----
One split at a time::

    python backend/scripts/build_manifest.py \
        --audio-root datasets/raw/ASVspoof2019/LA/ASVspoof2019_LA_train/flac \
        --protocol   datasets/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
        --split train \
        --output datasets/manifests/train.csv

All three ASVspoof 2019 LA splits at once (this is the one you want)::

    python backend/scripts/build_manifest.py --asvspoof2019-la datasets/raw/ASVspoof2019/LA
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# Allow "python backend/scripts/build_manifest.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml import discovery  # noqa: E402
from app.ml.progress import PhaseTimer, format_count  # noqa: E402


AUDIO_EXTENSIONS = (".flac", ".wav", ".ogg", ".mp3")

LABEL_VALUES = {"bonafide", "spoof"}
ATTACK_PATTERN = re.compile(r"^(A\d{2}|-)$")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def index_audio(audio_root: Path) -> dict[str, Path]:
    """Map file stem -> path for every audio file under ``audio_root``."""

    index: dict[str, Path] = {}

    for extension in AUDIO_EXTENSIONS:
        for path in audio_root.rglob(f"*{extension}"):
            index[path.stem] = path

    return index


def read_protocol_rows(protocol: Path) -> list[list[str]]:
    rows = []

    with protocol.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(line.split())

    return rows


def _spread_sample(rows: list[list[str]], count: int) -> list[list[str]]:
    """Evenly strided sample across the whole list, so sorting cannot skew it."""

    if len(rows) <= count:
        return rows

    stride = len(rows) / count
    return [rows[int(i * stride)] for i in range(count)]


def detect_columns(rows: list[list[str]], audio_index: dict[str, Path]) -> dict:
    """
    Work out which column holds the id, the label and the attack code.

    Raises with a readable diagnosis rather than returning something wrong.
    """

    if not rows:
        raise ValueError("protocol file is empty")

    # Sample ACROSS the file, not the first N rows. ASVspoof protocols are
    # sorted with every bonafide utterance first, and train has 2,580 of them -
    # so a 2,000-row head slice sees nothing but attack="-" and concludes there
    # is no attack column at all. That silently emptied the attack field, which
    # in turn made the attack-held-out split impossible to build.
    sample = _spread_sample(rows, 2000)
    width = min(len(row) for row in sample)

    if width == 0:
        raise ValueError("protocol rows have no columns")

    label_column = None
    id_column = None
    attack_column = None

    for column in range(width):
        values = [row[column] for row in sample]

        if label_column is None and all(v in LABEL_VALUES for v in values):
            label_column = column
            continue

        if id_column is None and audio_index:
            hits = sum(1 for v in values if v in audio_index)
            if hits >= 0.5 * len(values):
                id_column = column
                continue

    # The attack column is picked after the label column is known, so a column
    # of "-" placeholders elsewhere in the row cannot be mistaken for it.
    for column in range(width):
        if column in (label_column, id_column):
            continue
        values = [row[column] for row in sample]
        if all(ATTACK_PATTERN.match(v) for v in values) and any(
            v != "-" for v in values
        ):
            attack_column = column
            break

    if label_column is None:
        raise ValueError(
            "could not find a bonafide/spoof column in the protocol file.\n"
            f"first row was: {rows[0]}\n"
            "Is this really an ASVspoof CM protocol / trial_metadata file?"
        )

    if id_column is None:
        raise ValueError(
            "no protocol column matched the audio filenames on disk.\n"
            f"first row was: {rows[0]}\n"
            f"example filenames: {list(audio_index)[:3]}\n"
            "The --audio-root probably points at the wrong directory, or the "
            "audio for this split has not been extracted yet."
        )

    return {"id": id_column, "label": label_column, "attack": attack_column}


# ---------------------------------------------------------------------------
# manifest construction
# ---------------------------------------------------------------------------


def build_rows(
    rows: list[list[str]],
    columns: dict,
    audio_index: dict[str, Path],
    split: str,
    audio_root: Path,
    relative_to: Path | None,
) -> tuple[list[dict], Counter]:
    entries: list[dict] = []
    skipped: Counter = Counter()

    id_column = columns["id"]
    label_column = columns["label"]
    attack_column = columns["attack"]

    # Speaker id is whichever column is neither id, label nor attack and looks
    # like a speaker code. Useful for checking splits are speaker-disjoint.
    speaker_column = None
    speaker_sample = _spread_sample(rows, 2000)
    for candidate in range(min(len(row) for row in speaker_sample)):
        if candidate in (id_column, label_column, attack_column):
            continue
        values = {row[candidate] for row in speaker_sample}
        if 1 < len(values) < len(speaker_sample) and all(
            v.startswith(("LA_", "PA_", "DF_", "spk", "SPK")) for v in values
        ):
            speaker_column = candidate
            break

    for row in rows:
        if len(row) <= max(id_column, label_column):
            skipped["short_row"] += 1
            continue

        audio_id = row[id_column]
        label_text = row[label_column]

        path = audio_index.get(audio_id)
        if path is None:
            skipped["audio_missing"] += 1
            continue

        if label_text == "bonafide":
            label = 0
        elif label_text == "spoof":
            label = 1
        else:
            skipped["bad_label"] += 1
            continue

        attack = ""
        if attack_column is not None and len(row) > attack_column:
            attack = row[attack_column]
            if attack == "-":
                attack = ""

        speaker = ""
        if speaker_column is not None and len(row) > speaker_column:
            speaker = row[speaker_column]

        stored = path.resolve()
        if relative_to is not None:
            try:
                stored = stored.relative_to(relative_to.resolve())
            except ValueError:
                pass

        entries.append(
            {
                "path": str(stored).replace("\\", "/"),
                "label": label,
                "split": split,
                "attack": attack,
                "speaker": speaker,
            }
        )

    return entries, skipped


def write_manifest(entries: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "split", "attack", "speaker"],
        )
        writer.writeheader()
        writer.writerows(entries)


def report(entries: list[dict], skipped: Counter, output: Path) -> None:
    spoof = sum(1 for e in entries if e["label"] == 1)
    bonafide = len(entries) - spoof

    print(f"  wrote      {output}")
    print(f"  rows       {format_count(len(entries))}")
    print(
        f"  bonafide   {bonafide:>8}   ({bonafide / max(len(entries), 1) * 100:5.1f}%)"
    )
    print(f"  spoof      {spoof:>8}   ({spoof / max(len(entries), 1) * 100:5.1f}%)")

    attacks = Counter(e["attack"] for e in entries if e["attack"])
    if attacks:
        rendered = "  ".join(f"{k}:{v}" for k, v in sorted(attacks.items()))
        print(f"  attacks    {rendered}")

    speakers = {e["speaker"] for e in entries if e["speaker"]}
    if speakers:
        print(f"  speakers   {len(speakers)}")

    for reason, count in skipped.items():
        print(f"  skipped    {count} ({reason})")

    if not entries:
        print("  !! manifest is EMPTY - check --audio-root and --protocol")


# ---------------------------------------------------------------------------
# one split
# ---------------------------------------------------------------------------


def build_split(
    audio_root: Path,
    protocol: Path,
    split: str,
    output: Path,
    relative_to: Path | None,
) -> int:
    if not audio_root.exists():
        raise FileNotFoundError(f"audio directory not found: {audio_root}")
    if not protocol.exists():
        raise FileNotFoundError(f"protocol file not found: {protocol}")

    print(f"\n[{split}]")

    with PhaseTimer(f"indexing audio under {audio_root}"):
        audio_index = index_audio(audio_root)

    print(f"  audio files found: {format_count(len(audio_index))}")

    rows = read_protocol_rows(protocol)
    print(f"  protocol rows:     {format_count(len(rows))}")

    columns = detect_columns(rows, audio_index)
    print(
        f"  detected columns:  id={columns['id']}  label={columns['label']}"
        f"  attack={columns['attack']}"
    )

    entries, skipped = build_rows(
        rows, columns, audio_index, split, audio_root, relative_to
    )

    write_manifest(entries, output)
    report(entries, skipped, output)

    return len(entries)


# ---------------------------------------------------------------------------
# ASVspoof 2019 LA convenience
# ---------------------------------------------------------------------------


LA_LAYOUT = [
    ("train", "ASVspoof2019_LA_train/flac", "ASVspoof2019.LA.cm.train.trn.txt", "train.csv"),
    ("validation", "ASVspoof2019_LA_dev/flac", "ASVspoof2019.LA.cm.dev.trl.txt", "validation.csv"),
    ("test", "ASVspoof2019_LA_eval/flac", "ASVspoof2019.LA.cm.eval.trl.txt", "test.csv"),
]


def read_manifest_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_attack_holdout(
    manifest_dir: Path,
    holdout: set[str],
) -> None:
    """
    Write ``train_heldout.csv`` and ``val_unseen.csv``.

    Why this exists
    ---------------
    ASVspoof 2019 LA train and dev share the same six attacks (A01-A06). So
    dev EER measures how well the model recognises attacks it trained on, and
    it saturates near zero within a couple of epochs while performance on the
    *eval* attacks (A07-A19) is still improving. Selecting the checkpoint on
    dev EER therefore picks a memorising model and reports a number that
    predicts nothing about unseen attacks.

    Holding a couple of attack families out of training and validating only on
    those gives a model-selection signal that behaves like the real test. It
    costs a few thousand training clips and buys a result you can defend.

    Use them as::

        --train-manifest      datasets/manifests/train_heldout.csv
        --validation-manifest datasets/manifests/val_unseen.csv
    """

    train_path = manifest_dir / "train.csv"
    validation_path = manifest_dir / "validation.csv"

    if not train_path.exists() or not validation_path.exists():
        print("\n[attack holdout]  skipped - train.csv/validation.csv not built")
        return

    train_rows = read_manifest_rows(train_path)
    validation_rows = read_manifest_rows(validation_path)

    kept = [r for r in train_rows if (r.get("attack") or "") not in holdout]
    dropped = len(train_rows) - len(kept)

    # Bonafide has no attack id, so it belongs in both - the unseen-attack
    # validation set still needs a negative class to compute an EER against.
    unseen = [
        r
        for r in validation_rows
        if (r.get("attack") or "") in holdout or r["label"] == "0"
    ]

    train_out = manifest_dir / "train_heldout.csv"
    validation_out = manifest_dir / "val_unseen.csv"

    write_manifest(kept, train_out)
    write_manifest(unseen, validation_out)

    unseen_spoof = sum(1 for r in unseen if r["label"] == "1")

    print(f"\n[attack holdout]  held out: {', '.join(sorted(holdout))}")
    print(
        f"  {train_out.name:<18} {len(kept)} rows "
        f"({dropped} dropped from train)"
    )
    print(
        f"  {validation_out.name:<18} {len(unseen)} rows "
        f"({len(unseen) - unseen_spoof} bonafide, {unseen_spoof} unseen-attack spoof)"
    )

    if unseen_spoof == 0:
        print(
            "  !! no spoof rows survived - the held-out attacks do not appear\n"
            "     in the validation split, so val_unseen.csv is unusable."
        )
    else:
        print(
            "  Select checkpoints on val_unseen EER, not on full-dev EER:\n"
            "    --train-manifest datasets/manifests/train_heldout.csv \\\n"
            "    --validation-manifest datasets/manifests/val_unseen.csv"
        )


def build_asvspoof2019_la(
    la_root: Path,
    manifest_dir: Path,
    relative_to: Path | None,
) -> None:
    protocols = la_root / "ASVspoof2019_LA_cm_protocols"

    if not protocols.exists():
        candidates = list(la_root.rglob("ASVspoof2019.LA.cm.*.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"could not find the CM protocol directory under {la_root}\n"
                "Expected ASVspoof2019_LA_cm_protocols/ containing "
                "ASVspoof2019.LA.cm.train.trn.txt"
            )
        protocols = candidates[0].parent

    total = 0

    for split, audio_subdir, protocol_name, manifest_name in LA_LAYOUT:
        audio_root = la_root / audio_subdir

        if not audio_root.exists():
            alternatives = [
                p for p in la_root.rglob("flac") if split.replace("validation", "dev") in str(p).lower()
            ]
            if alternatives:
                audio_root = alternatives[0]
            else:
                print(f"\n[{split}]  SKIPPED - {audio_root} not found")
                continue

        total += build_split(
            audio_root=audio_root,
            protocol=protocols / protocol_name,
            split=split,
            output=manifest_dir / manifest_name,
            relative_to=relative_to,
        )

    print(f"\nTotal rows across all splits: {format_count(total)}")
    print(
        "\nThese splits are speaker-disjoint by construction in ASVspoof 2019 LA,\n"
        "and the eval set uses attack types (A07-A19) that never appear in train\n"
        "(A01-A06). That is what makes the eval EER meaningful - do not reshuffle\n"
        "them into a random split."
    )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


SPLIT_TO_MANIFEST = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def build_discovered(
    root: Path,
    manifest_dir: Path,
    relative_to: Path | None,
    holdout_attacks: str | None,
) -> None:
    """
    Find whatever ASVspoof corpus is under ``root`` and build manifests for it.

    Used when the layout is unknown - a Kaggle dataset mount, a Drive folder, a
    zip someone made by hand. See app/ml/discovery.py for how splits are
    identified from utterance-id prefixes rather than directory names.
    """

    with PhaseTimer(f"scanning {root} for an ASVspoof corpus"):
        layout = discovery.discover(root)

    print()
    print(layout.describe())
    print()

    if not layout.audio:
        raise SystemExit(
            f"no ASVspoof audio found under {root}.\n"
            "Expected files named like LA_T_1138215.flac / DF_E_2000011.flac.\n"
            "Check the path, and that the archive was actually extracted."
        )

    missing = [s for s in layout.audio if s not in layout.protocols]
    if missing:
        print(
            f"  WARNING: no protocol found for split(s): {', '.join(missing)}\n"
            f"  Those splits cannot be labelled and will be skipped. For "
            f"ASVspoof 2021 DF the labels ship separately as "
            f"DF-keys-full.tar.gz - extract it under {root} and re-run.\n"
        )

    total = 0

    for split in sorted(layout.audio):
        if split not in layout.protocols:
            continue

        output = manifest_dir / SPLIT_TO_MANIFEST.get(split, f"{split}.csv")

        total += build_split(
            audio_root=layout.audio[split],
            protocol=layout.protocols[split],
            split=split,
            output=output,
            relative_to=relative_to,
        )

    print(f"\nTotal rows across all splits: {format_count(total)}")

    holdout = {
        token.strip().upper()
        for token in (holdout_attacks or "").split(",")
        if token.strip()
    }
    if holdout and "train" in layout.audio and "validation" in layout.audio:
        build_attack_holdout(manifest_dir, holdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VoxShield manifests from ASVspoof protocol files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--asvspoof2019-la",
        type=Path,
        help="Path to the extracted LA/ directory; builds all three manifests.",
    )

    parser.add_argument(
        "--discover",
        type=Path,
        help="Point at ANY directory containing an ASVspoof corpus and let the "
             "layout be found automatically. Splits are identified from the "
             "utterance-id prefixes (LA_T_/LA_D_/LA_E_/DF_E_) rather than folder "
             "names, so this works on a Kaggle dataset, a Drive folder or a "
             "hand-made zip without being told the structure. Use this instead "
             "of --asvspoof2019-la when you do not know the layout.",
    )

    parser.add_argument("--audio-root", type=Path, help="Directory containing the audio.")
    parser.add_argument("--protocol", type=Path, help="Protocol / trial_metadata file.")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        help="Split name written into the manifest.",
    )
    parser.add_argument("--output", type=Path, help="Output manifest CSV.")

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("datasets/manifests"),
        help="Where the --asvspoof2019-la mode writes its manifests.",
    )

    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute audio paths instead of paths relative to the repo root.",
    )

    parser.add_argument(
        "--holdout-attacks",
        type=str,
        default="A05,A06",
        help="Comma-separated attacks to hold out of training, producing "
             "train_heldout.csv and val_unseen.csv for honest checkpoint "
             "selection. Empty string disables. Default: A05,A06.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    relative_to = None if args.absolute_paths else repo_root

    if args.discover:
        build_discovered(
            root=args.discover.resolve(),
            manifest_dir=args.manifest_dir,
            relative_to=relative_to,
            holdout_attacks=args.holdout_attacks,
        )
        return

    if args.asvspoof2019_la:
        build_asvspoof2019_la(
            la_root=args.asvspoof2019_la.resolve(),
            manifest_dir=args.manifest_dir,
            relative_to=relative_to,
        )

        holdout = {
            token.strip().upper()
            for token in (args.holdout_attacks or "").split(",")
            if token.strip()
        }
        if holdout:
            build_attack_holdout(args.manifest_dir, holdout)

        return

    missing = [
        name
        for name, value in (
            ("--audio-root", args.audio_root),
            ("--protocol", args.protocol),
            ("--split", args.split),
            ("--output", args.output),
        )
        if value is None
    ]

    if missing:
        raise SystemExit(
            "Provide either --asvspoof2019-la, or all of: "
            + ", ".join(missing)
        )

    build_split(
        audio_root=args.audio_root.resolve(),
        protocol=args.protocol.resolve(),
        split=args.split,
        output=args.output,
        relative_to=relative_to,
    )


if __name__ == "__main__":
    main()
