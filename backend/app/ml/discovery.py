"""
Locate an ASVspoof corpus inside an arbitrary directory tree.

Why this is not just "join a few path strings"
----------------------------------------------
The official ASVspoof release has a fixed layout, but almost nobody hands it to
you that way. Kaggle datasets, Drive folders and hand-made zips each nest it
differently - ``LA/``, ``asvspoof2019/LA/``, ``data/LA/ASVspoof2019_LA_train/``,
or the three splits unpacked side by side with no ``LA/`` at all. Hardcoding
paths means the pipeline breaks on every new copy of the same corpus.

So the split is identified from the **filenames**, not the folders. ASVspoof
encodes it in every utterance id:

    LA_T_1138215   train        DF_E_2000011   2021 DF eval
    LA_D_1047731   dev
    LA_E_2482298   eval

A directory of ``LA_T_*.flac`` is the training audio no matter what anyone
called the folder above it. Protocol files are found the same way - by content,
not by name - so a renamed protocol still resolves.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_SUFFIXES = (".flac", ".wav", ".ogg")

# utterance-id prefix -> split name used throughout VoxShield
PREFIX_TO_SPLIT = {
    "LA_T": "train",
    "LA_D": "validation",
    "LA_E": "test",
    "DF_E": "test",
    "PA_T": "train",
    "PA_D": "validation",
    "PA_E": "test",
}

LABEL_VALUES = {"bonafide", "spoof"}


@dataclass
class Layout:
    """Where each split's audio and protocol actually live."""

    root: Path
    audio: dict[str, Path] = field(default_factory=dict)
    protocols: dict[str, Path] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def splits(self) -> list[str]:
        return sorted(set(self.audio) | set(self.protocols))

    def is_complete(self) -> bool:
        """Every split with audio also has a protocol to label it."""

        return bool(self.audio) and all(s in self.protocols for s in self.audio)

    def describe(self) -> str:
        lines = [f"discovered under {self.root}"]

        for split in self.splits:
            audio = self.audio.get(split)
            protocol = self.protocols.get(split)
            lines.append(f"  [{split}]")
            lines.append(
                f"    audio     {audio}  ({self.counts.get(split, 0)} files)"
                if audio
                else "    audio     MISSING"
            )
            lines.append(
                f"    protocol  {protocol}" if protocol else "    protocol  MISSING"
            )

        if not self.splits:
            lines.append("  nothing found")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def _split_of(stem: str) -> str | None:
    """Map an utterance id to a split via its prefix, e.g. LA_T_123 -> train."""

    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return PREFIX_TO_SPLIT.get(f"{parts[0]}_{parts[1]}")


def find_audio_directories(root: Path, max_files: int = 400_000) -> dict[str, Path]:
    """
    Group audio by split, returning the deepest common directory for each.

    A single walk over the tree; each file votes for a split based on its own
    id prefix. The directory reported per split is the common ancestor of that
    split's files, which is exactly what ``--audio-root`` wants - and it stays
    correct whether the splits sit in sibling folders or are all mixed into one.
    """

    root = Path(root)

    votes: dict[str, Counter] = {}
    seen = 0

    for path in root.rglob("*"):
        if seen >= max_files:
            break
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue

        seen += 1
        split = _split_of(path.stem)
        if split is None:
            continue

        votes.setdefault(split, Counter())[str(path.parent)] += 1

    directories: dict[str, Path] = {}

    for split, counter in votes.items():
        parents = [Path(p) for p in counter]
        common = parents[0]
        for parent in parents[1:]:
            common = _common_ancestor(common, parent)
        directories[split] = common

    return directories


def _common_ancestor(a: Path, b: Path) -> Path:
    a_parts, b_parts = a.parts, b.parts
    shared = []
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        shared.append(x)
    return Path(*shared) if shared else a


def count_audio(directory: Path, split: str) -> int:
    """How many files in ``directory`` actually belong to ``split``."""

    total = 0
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in AUDIO_SUFFIXES and _split_of(path.stem) == split:
            total += 1
    return total


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------


def looks_like_protocol(path: Path, probe_lines: int = 20) -> str | None:
    """
    Return the split this file labels, or None if it is not a protocol.

    Identified by content: a protocol has whitespace-separated columns, one of
    which is bonafide/spoof, and utterance ids whose prefix names the split.
    That survives any renaming of the file itself.
    """

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = []
            for index, line in enumerate(handle):
                if index >= probe_lines:
                    break
                columns = line.split()
                if columns:
                    rows.append(columns)
    except OSError:
        return None

    if not rows:
        return None

    if not any(any(c in LABEL_VALUES for c in row) for row in rows):
        return None

    splits: Counter = Counter()
    for row in rows:
        for column in row:
            split = _split_of(column)
            if split:
                splits[split] += 1

    if not splits:
        return None

    return splits.most_common(1)[0][0]


def find_protocols(root: Path, max_candidates: int = 4000) -> dict[str, Path]:
    """
    Find one protocol file per split.

    Where several candidates claim the same split, the largest wins - the full
    protocol rather than a truncated sample someone left alongside it.
    """

    root = Path(root)
    best: dict[str, tuple[int, Path]] = {}
    checked = 0

    for path in root.rglob("*"):
        if checked >= max_candidates:
            break
        if not path.is_file() or path.suffix.lower() not in (".txt", ".trl", ".trn", ".lst", ""):
            continue
        if path.stat().st_size == 0 or path.stat().st_size > 200 * 1024 * 1024:
            continue

        checked += 1
        split = looks_like_protocol(path)
        if split is None:
            continue

        size = path.stat().st_size
        if split not in best or size > best[split][0]:
            best[split] = (size, path)

    return {split: path for split, (_, path) in best.items()}


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def discover(root: Path | str, count_files: bool = True) -> Layout:
    """
    Find every ASVspoof split under ``root``.

        layout = discover("/kaggle/input")
        print(layout.describe())
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"not found: {root}")

    audio = find_audio_directories(root)
    protocols = find_protocols(root)

    layout = Layout(root=root, audio=audio, protocols=protocols)

    if count_files:
        layout.counts = {
            split: count_audio(directory, split) for split, directory in audio.items()
        }

    return layout
