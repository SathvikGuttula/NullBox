"""
Corpus discovery.

Every copy of ASVspoof is nested differently — a Kaggle dataset, a Drive
folder, a hand-made zip. These tests use deliberately awkward layouts, because
the layouts that matter are the ones nobody planned for.
"""

import pathlib

import numpy as np
import pytest
import soundfile as sf

from app.ml.discovery import (
    discover,
    find_audio_directories,
    find_protocols,
    looks_like_protocol,
)


def write_clip(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.3, 4800, endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000,
             format="FLAC")


def write_protocol(path, prefix, count, speaker="LA_0079"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(count):
        spoof = i % 3 != 0
        attack = f"A0{i % 6 + 1}" if spoof else "-"
        label = "spoof" if spoof else "bonafide"
        lines.append(f"{speaker} {prefix}_{1000000 + i} - {attack} {label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def official(tmp_path):
    """The layout as ASVspoof actually ships it."""

    la = tmp_path / "LA"
    for sub, prefix, n in [
        ("ASVspoof2019_LA_train", "LA_T", 10),
        ("ASVspoof2019_LA_dev", "LA_D", 8),
        ("ASVspoof2019_LA_eval", "LA_E", 6),
    ]:
        for i in range(n):
            write_clip(la / sub / "flac" / f"{prefix}_{1000000 + i}.flac")

    protocols = la / "ASVspoof2019_LA_cm_protocols"
    write_protocol(protocols / "ASVspoof2019.LA.cm.train.trn.txt", "LA_T", 10)
    write_protocol(protocols / "ASVspoof2019.LA.cm.dev.trl.txt", "LA_D", 8)
    write_protocol(protocols / "ASVspoof2019.LA.cm.eval.trl.txt", "LA_E", 6)

    return tmp_path


@pytest.fixture
def kaggle_style(tmp_path):
    """
    A Kaggle-style mount: extra nesting, renamed folders, protocols in an
    unrelated directory, and a stray README to ignore.
    """

    base = tmp_path / "kaggle" / "input" / "asvpoof-2019-dataset-la" / "data"

    for sub, prefix, n in [
        ("train_audio", "LA_T", 10),
        ("dev_audio", "LA_D", 8),
        ("eval_audio", "LA_E", 6),
    ]:
        for i in range(n):
            write_clip(base / sub / f"{prefix}_{1000000 + i}.flac")

    labels = tmp_path / "kaggle" / "input" / "asvpoof-2019-dataset-la" / "labels"
    write_protocol(labels / "train_keys.txt", "LA_T", 10)
    write_protocol(labels / "dev_keys.txt", "LA_D", 8)
    write_protocol(labels / "eval_keys.txt", "LA_E", 6)

    (labels / "README.txt").write_text(
        "This dataset contains speech.\nNothing structured here.\n", encoding="utf-8"
    )

    return tmp_path


# ---------------------------------------------------------------------------


def test_official_layout(official):
    layout = discover(official)

    assert set(layout.audio) == {"train", "validation", "test"}
    assert set(layout.protocols) == {"train", "validation", "test"}
    assert layout.is_complete()
    assert layout.counts["train"] == 10
    assert layout.counts["validation"] == 8
    assert layout.counts["test"] == 6


def test_renamed_folders_still_resolve(kaggle_style):
    """
    The point of the whole module: folders called `train_audio` and
    `train_keys.txt` carry no official naming, but the utterance ids do.
    """

    layout = discover(kaggle_style)

    assert layout.is_complete()
    assert set(layout.audio) == {"train", "validation", "test"}
    assert layout.audio["train"].name == "train_audio"
    assert layout.counts["validation"] == 8


def test_readme_is_not_mistaken_for_a_protocol(kaggle_style):
    protocols = find_protocols(kaggle_style)

    assert all(p.name != "README.txt" for p in protocols.values())


def test_all_splits_in_one_directory(tmp_path):
    """Some redistributions dump every split into a single folder."""

    flat = tmp_path / "audio"
    for prefix, n in [("LA_T", 6), ("LA_D", 4), ("LA_E", 5)]:
        for i in range(n):
            write_clip(flat / f"{prefix}_{1000000 + i}.flac")

    directories = find_audio_directories(tmp_path)

    assert set(directories) == {"train", "validation", "test"}
    # All three resolve to the same directory, which is correct here.
    assert len({str(p) for p in directories.values()}) == 1


def test_split_inferred_from_prefix_not_path(tmp_path):
    """A folder named 'train' holding eval clips must report test, not train."""

    misleading = tmp_path / "train"
    for i in range(5):
        write_clip(misleading / f"LA_E_{2000000 + i}.flac")

    directories = find_audio_directories(tmp_path)

    assert set(directories) == {"test"}


def test_df_prefix_recognised(tmp_path):
    for i in range(5):
        write_clip(tmp_path / "df" / f"DF_E_{2000000 + i}.flac")

    assert set(find_audio_directories(tmp_path)) == {"test"}


def test_protocol_detection_by_content():
    """A protocol is recognised by what is in it, not what it is called."""

    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp())

    good = directory / "anything.dat"
    write_protocol(good, "LA_T", 5)
    assert looks_like_protocol(good) == "train"

    prose = directory / "notes.txt"
    prose.write_text("bonafide speech is genuine\n", encoding="utf-8")
    # Mentions the word but has no utterance ids, so it is not a protocol.
    assert looks_like_protocol(prose) is None


def test_largest_protocol_wins(tmp_path):
    """A truncated sample alongside the real protocol must not win."""

    write_protocol(tmp_path / "full" / "keys.txt", "LA_T", 200)
    write_protocol(tmp_path / "sample" / "keys_sample.txt", "LA_T", 3)

    protocols = find_protocols(tmp_path)

    assert protocols["train"].parent.name == "full"


def test_missing_protocol_is_reported_not_guessed(tmp_path):
    for i in range(5):
        write_clip(tmp_path / "audio" / f"DF_E_{2000000 + i}.flac")

    layout = discover(tmp_path)

    assert "test" in layout.audio
    assert not layout.is_complete()


def test_empty_tree(tmp_path):
    layout = discover(tmp_path)

    assert layout.audio == {}
    assert not layout.is_complete()
    assert "nothing found" in layout.describe()


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")


def test_describe_is_readable(official):
    text = discover(official).describe()

    assert "[train]" in text
    assert "protocol" in text


# ---------------------------------------------------------------------------
# regressions from the Kaggle run
# ---------------------------------------------------------------------------


def write_cm_protocol(path, prefix, n):
    """Countermeasure protocol: speaker utt - attack label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # ASVspoof sorts every bonafide utterance FIRST. That ordering is the bug.
    for i in range(n // 10):
        rows.append(f"LA_{i%9:04d} {prefix}_{1000000+i} - - bonafide")
    for i in range(n // 10, n):
        rows.append(f"LA_{i%9:04d} {prefix}_{1000000+i} - A0{i%6+1} spoof")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_asv_protocol(path, prefix, n):
    """ASV protocol: speaker utt label trial. Larger, and NOT what we want."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        label = "bonafide" if i % 4 == 0 else "spoof"
        trial = "target" if i % 3 else "nontarget"
        rows.append(f"LA_{i%9:04d} {prefix}_{1000000+i} {label} {trial}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cm_protocol_beats_a_larger_asv_protocol(tmp_path):
    """
    The Kaggle failure. Both protocol families mention bonafide/spoof and the
    ASV one is bigger, so "largest wins" chose it - producing a test manifest
    that was 100% bonafide with 63,882 rows discarded.
    """

    for i in range(20):
        write_clip(tmp_path / "LA/eval/flac" / f"LA_E_{1000000+i}.flac")

    write_cm_protocol(tmp_path / "LA/cm_protocols/eval.trl.txt", "LA_E", 20)
    write_asv_protocol(tmp_path / "LA/asv_protocols/eval.gi.trl.txt", "LA_E", 200)

    layout = discover(tmp_path)

    chosen = layout.protocols["test"]
    assert "cm" in str(chosen), f"picked the ASV protocol: {chosen}"


def test_asv_protocol_alone_is_still_usable(tmp_path):
    """If only an ASV protocol exists, use it rather than finding nothing."""

    for i in range(20):
        write_clip(tmp_path / "eval" / f"LA_E_{1000000+i}.flac")
    write_asv_protocol(tmp_path / "asv/eval.trl.txt", "LA_E", 40)

    assert "test" in discover(tmp_path).protocols


def test_attack_column_survives_bonafide_first_ordering(tmp_path):
    """
    The second Kaggle failure. detect_columns sampled the first 2000 rows, and
    ASVspoof lists all bonafide first - so it saw only attack="-" and reported
    no attack column, which made the held-out-attack split impossible.
    """

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bm", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_manifest.py"
    )
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)

    protocol = tmp_path / "train.trn.txt"
    write_cm_protocol(protocol, "LA_T", 25000)   # 2500 bonafide first

    rows = bm.read_protocol_rows(protocol)
    audio_index = {r[1]: pathlib.Path(f"{r[1]}.flac") for r in rows}

    columns = bm.detect_columns(rows, audio_index)

    assert columns["attack"] is not None, "attack column missed again"
    assert columns["label"] == 4
    assert columns["id"] == 1
