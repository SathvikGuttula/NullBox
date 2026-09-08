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

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_SUFFIXES = (".flac", ".wav", ".ogg")

# Files worth opening to check whether they are a protocol. Deliberately narrow:
# on a 700k-file mount, every candidate costs a stat and a read.
PROTOCOL_SUFFIXES = (".txt", ".trl", ".trn", ".lst", ".csv", ".tsv")

MAX_CANDIDATES = 4000

# A01..A19 in ASVspoof 2019/2021 countermeasure protocols.
ATTACK_CODE = re.compile(r"^A\d{2}$")

# Present in ASV (speaker verification) protocols, absent from CM ones.
ASV_TRIAL_VALUES = {"target", "nontarget", "zerotarget"}

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


def scan(root: Path, progress_every: int = 100_000, stream=None) -> "Scan":
    """
    Walk the tree ONCE, collecting everything discovery needs.

    This is the whole performance story. ``/kaggle/input`` and Google Drive are
    network-backed FUSE mounts where every lookup is a round trip, and ASVspoof
    is enormous by file count - LA is ~121k files and DF eval is 611,829. The
    first version of this module walked the tree five times (audio, protocols,
    then once per split to count) and called ``Path.is_file()`` on every entry,
    which is a stat each. On a local disk that is merely wasteful; on a FUSE
    mount it takes tens of minutes.

    So: one ``os.walk``, no stat calls on audio at all. The split, the count and
    the directory all come from the filename, which we already have. Only
    protocol *candidates* - a handful of small text files - get stat'd later.
    """

    root = Path(root)
    stream = stream or sys.stdout

    audio_votes: dict[str, Counter] = {}
    counts: Counter = Counter()
    candidates: list[Path] = []
    scanned = 0

    for directory, _, filenames in os.walk(root):
        for name in filenames:
            scanned += 1

            if progress_every and scanned % progress_every == 0:
                stream.write(f"    scanned {scanned:,} files...\n")
                stream.flush()

            stem, _, extension = name.rpartition(".")
            extension = f".{extension.lower()}" if stem else ""

            if extension in AUDIO_SUFFIXES:
                split = _split_of(stem)
                if split is not None:
                    audio_votes.setdefault(split, Counter())[directory] += 1
                    counts[split] += 1
            elif extension in PROTOCOL_SUFFIXES and len(candidates) < MAX_CANDIDATES:
                candidates.append(Path(directory) / name)

    directories: dict[str, Path] = {}
    for split, counter in audio_votes.items():
        parents = [Path(p) for p in counter]
        common = parents[0]
        for parent in parents[1:]:
            common = _common_ancestor(common, parent)
        directories[split] = common

    return Scan(
        audio=directories,
        counts=dict(counts),
        protocol_candidates=candidates,
        files_scanned=scanned,
    )


@dataclass
class Scan:
    """Raw result of the single filesystem walk."""

    audio: dict[str, Path] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    protocol_candidates: list[Path] = field(default_factory=list)
    files_scanned: int = 0


def find_audio_directories(root: Path) -> dict[str, Path]:
    """Group audio by split, returning the common directory for each."""

    return scan(root, progress_every=0).audio


def _common_ancestor(a: Path, b: Path) -> Path:
    a_parts, b_parts = a.parts, b.parts
    shared = []
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        shared.append(x)
    return Path(*shared) if shared else a


def count_audio(directory: Path, split: str) -> int:
    """
    How many files in ``directory`` belong to ``split``.

    Kept for direct use, but ``discover`` no longer calls it - the counts come
    free from the single walk, and doing it here meant one extra full traversal
    per split.
    """

    total = 0
    for _, _, filenames in os.walk(directory):
        for name in filenames:
            stem, _, extension = name.rpartition(".")
            if stem and f".{extension.lower()}" in AUDIO_SUFFIXES:
                if _split_of(stem) == split:
                    total += 1
    return total


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------


def inspect_protocol(path: Path, probe_lines: int = 4000) -> tuple[str | None, bool]:
    """
    Return ``(split, is_countermeasure)`` for a candidate protocol file.

    ``is_countermeasure`` is the important half. ASVspoof ships two protocol
    families side by side and they are NOT interchangeable:

        ASVspoof2019_LA_cm_protocols/   LA_0079 LA_T_1138215 - A01 spoof
                                        countermeasure: bonafide vs spoof

        ASVspoof2019_LA_asv_protocols/  LA_0073 LA_D_4004968 bonafide target
                                        speaker verification: target vs nontarget

    Both mention bonafide and spoof, so "contains a label column" cannot tell
    them apart - and the ASV file is the *larger* of the two, so picking the
    biggest match silently chose the wrong one. Scoring an anti-spoof model
    against ASV trials produced a manifest that was 100% bonafide with 63,882
    rows discarded.

    The reliable discriminator is the attack column: a CM protocol carries
    A01..A19 codes, an ASV protocol carries target/nontarget instead.
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
        return None, False

    if not rows:
        return None, False

    if not any(any(c in LABEL_VALUES for c in row) for row in rows):
        return None, False

    splits: Counter = Counter()
    for row in rows:
        for column in row:
            split = _split_of(column)
            if split:
                splits[split] += 1

    if not splits:
        return None, False

    has_attack = any(
        ATTACK_CODE.match(column) for row in rows for column in row
    )
    has_asv_trial = any(
        column in ASV_TRIAL_VALUES for row in rows for column in row
    )

    return splits.most_common(1)[0][0], (has_attack and not has_asv_trial)


def looks_like_protocol(path: Path, probe_lines: int = 4000) -> str | None:
    """Split this file labels, or None. See :func:`inspect_protocol`."""

    return inspect_protocol(path, probe_lines)[0]


def select_protocols(candidates: list[Path]) -> dict[str, Path]:
    """
    Pick one protocol file per split from the candidates the walk collected.

    Where several claim the same split, the largest wins - the full protocol
    rather than a truncated sample someone left beside it.

    Each candidate is stat'd exactly once. The earlier version called
    ``.stat()`` three times per file and re-walked the whole tree to find them,
    which is what made this slow on a network mount.
    """

    best: dict[str, tuple[int, int, Path]] = {}

    for path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size == 0 or size > 200 * 1024 * 1024:
            continue

        split, is_cm = inspect_protocol(path)
        if split is None:
            continue

        # Rank on (is_countermeasure, size). A countermeasure protocol always
        # beats an ASV one regardless of size; size only breaks ties within a
        # family, where it correctly prefers the full protocol over a truncated
        # sample left beside it.
        rank = (1 if is_cm else 0, size)

        if split not in best or rank > best[split][:2]:
            best[split] = (rank[0], rank[1], path)

    return {split: path for split, (_, _, path) in best.items()}


def find_protocols(root: Path) -> dict[str, Path]:
    """Find one protocol file per split under ``root``."""

    return select_protocols(scan(root, progress_every=0).protocol_candidates)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def discover(
    root: Path | str,
    count_files: bool = True,
    verbose: bool = True,
) -> Layout:
    """
    Find every ASVspoof split under ``root``.

        layout = discover("/kaggle/input")
        print(layout.describe())

    One filesystem walk, then a stat and a short read on each protocol
    candidate. On a network mount, narrow ``root`` as much as you can - passing
    ``/kaggle/input`` with a 611k-file DF dataset attached means walking all of
    it, even when you only care about LA. Point at the specific dataset
    directory instead and it is seconds rather than minutes.
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"not found: {root}")

    result = scan(root, progress_every=100_000 if verbose else 0)

    if verbose:
        print(
            f"    walked {result.files_scanned:,} files, "
            f"{len(result.protocol_candidates)} protocol candidate(s)"
        )

    layout = Layout(
        root=root,
        audio=result.audio,
        protocols=select_protocols(result.protocol_candidates),
        counts=result.counts if count_files else {},
    )

    return layout
