"""
Corpus discovery.

Every copy of ASVspoof is nested differently — a Kaggle dataset, a Drive
folder, a hand-made zip. These tests use deliberately awkward layouts, because
the layouts that matter are the ones nobody planned for.
"""

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
