import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from app.ml.dataset import (
    AudioSample,
    VoiceSpoofDataset,
    WaveformStore,
    balanced_sample_weights,
    collate,
    describe,
    fit_length,
    load_manifest,
    peak_normalize,
    read_audio,
    stratified_subsample,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path):
    """A tiny on-disk corpus plus its manifest."""

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = []

    for index in range(6):
        label = index % 2
        seconds = 1.0 + index * 0.7
        t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
        signal = (0.3 * np.sin(2 * np.pi * (200 + 50 * index) * t)).astype(np.float32)

        path = audio_dir / f"clip_{index}.flac"
        sf.write(path, signal, 16000, format="FLAC")

        rows.append(
            {
                "path": str(path),
                "label": label,
                "split": "train" if index < 4 else "validation",
                "attack": "" if label == 0 else f"A0{index}",
                "speaker": f"SPK_{index % 3}",
            }
        )

    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "split", "attack", "speaker"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_load_manifest(corpus):
    samples = load_manifest(corpus)

    assert len(samples) == 6
    assert samples[0].label in (0, 1)
    assert samples[1].attack == "A01"
    assert samples[0].speaker == "SPK_0"


def test_load_manifest_filters_by_split(corpus):
    assert len(load_manifest(corpus, split="train")) == 4
    assert len(load_manifest(corpus, split="validation")) == 2


def test_load_manifest_limit(corpus):
    assert len(load_manifest(corpus, limit=3)) == 3


def test_limit_keeps_both_classes(corpus):
    """
    A head-limited subset of an ASVspoof protocol is often one speaker or even
    one class, which makes EER undefined and any --limit-train experiment
    meaningless.
    """

    samples = load_manifest(corpus, limit=4)

    assert len(samples) == 4
    assert len({s.label for s in samples}) == 2


def test_limit_larger_than_the_manifest_is_harmless(corpus):
    assert len(load_manifest(corpus, limit=999)) == 6


def test_subsample_keeps_every_attack_family(tmp_path):
    """
    The failure this guards against: a subset that drops whole attack families
    trains on a different problem than the full run, so its EER is not a
    preview of anything.
    """

    samples = []
    for attack_index, attack in enumerate(["A01", "A02", "A03", "A04", "A05", "A06"]):
        for i in range(50):
            samples.append(
                AudioSample(path=f"{attack}_{i}.flac", label=1, attack=attack)
            )
    for i in range(30):
        samples.append(AudioSample(path=f"bf_{i}.flac", label=0, attack=None))

    subset = stratified_subsample(samples, 60)

    assert len(subset) == 60
    assert {s.attack for s in subset if s.label == 1} == {
        "A01", "A02", "A03", "A04", "A05", "A06"
    }
    assert any(s.label == 0 for s in subset)


def test_subsample_preserves_class_ratio_approximately():
    samples = [AudioSample(path=f"s{i}.flac", label=1, attack="A01") for i in range(900)]
    samples += [AudioSample(path=f"b{i}.flac", label=0) for i in range(100)]

    subset = stratified_subsample(samples, 100)

    spoof_ratio = sum(1 for s in subset if s.label == 1) / len(subset)

    assert spoof_ratio == pytest.approx(0.9, abs=0.05)


def test_subsample_never_returns_duplicates():
    samples = [
        AudioSample(path=f"s{i}.flac", label=i % 2, attack=f"A0{i % 4}")
        for i in range(200)
    ]

    subset = stratified_subsample(samples, 37)

    assert len(subset) == 37
    assert len({s.path for s in subset}) == 37


def test_empty_manifest_raises_a_useful_error(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_manifest(empty)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.csv")


def test_describe_reports_balance(corpus):
    stats = describe(load_manifest(corpus))

    assert stats["total"] == 6
    assert stats["bonafide"] == 3
    assert stats["spoof"] == 3
    assert stats["spoof_ratio"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# audio handling
# ---------------------------------------------------------------------------


def test_read_audio_returns_mono_float32(corpus):
    sample = load_manifest(corpus)[0]
    audio = read_audio(sample.path)

    assert audio.ndim == 1
    assert audio.dtype == np.float32


def test_fit_length_crops_long_audio():
    audio = np.arange(1000, dtype=np.float32)
    assert fit_length(audio, 400).shape[0] == 400


def test_fit_length_tiles_rather_than_zero_pads():
    """
    Short clips must be tiled, not zero-padded.

    Zero padding makes "trailing silence" a usable feature, and because
    bonafide and spoof clips have different length distributions in ASVspoof,
    that feature leaks the label - the model scores well and has learned
    nothing about synthesis.
    """

    audio = np.ones(100, dtype=np.float32)
    fitted = fit_length(audio, 350)

    assert fitted.shape[0] == 350
    assert not np.any(fitted == 0.0)


def test_deterministic_crop_when_not_training():
    audio = np.arange(1000, dtype=np.float32)

    first = fit_length(audio, 400, random_crop=False)
    second = fit_length(audio, 400, random_crop=False)

    assert np.array_equal(first, second)
    assert first[0] == 0.0


def test_random_crop_actually_moves():
    import random

    audio = np.arange(100000, dtype=np.float32)

    offsets = {fit_length(audio, 400, random_crop=True, rng=random.Random(s))[0] for s in range(20)}

    assert len(offsets) > 1


def test_peak_normalize_scales_to_unit_peak():
    audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    assert np.max(np.abs(peak_normalize(audio))) == pytest.approx(1.0)


def test_peak_normalize_leaves_silence_alone():
    audio = np.zeros(100, dtype=np.float32)
    assert np.array_equal(peak_normalize(audio), audio)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def test_dataset_returns_fixed_length_tensors(corpus):
    dataset = VoiceSpoofDataset(samples=load_manifest(corpus), max_seconds=2.0)

    item = dataset[0]

    assert item["input_values"].shape == (32000,)
    assert item["input_values"].dtype == torch.float32
    assert item["label"].dtype == torch.long


def test_collate_stacks(corpus):
    dataset = VoiceSpoofDataset(samples=load_manifest(corpus), max_seconds=1.0)

    batch = collate([dataset[i] for i in range(4)])

    assert batch["input_values"].shape == (4, 16000)
    assert batch["label"].shape == (4,)


def test_augmentation_only_applies_in_training_mode(corpus):
    calls = []

    def augment(tensor):
        calls.append(1)
        return tensor

    samples = load_manifest(corpus)

    VoiceSpoofDataset(samples=samples, train=False, augment=augment)[0]
    assert not calls

    VoiceSpoofDataset(samples=samples, train=True, augment=augment)[0]
    assert calls


def test_balanced_sample_weights_equalise_class_mass():
    labels = np.array([0] * 10 + [1] * 90)

    weights = balanced_sample_weights(labels).numpy()

    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())


# ---------------------------------------------------------------------------
# waveform cache
# ---------------------------------------------------------------------------


def test_waveform_store_roundtrip(corpus, tmp_path):
    samples = load_manifest(corpus)

    store = WaveformStore.build(samples, tmp_path / "cache", sample_rate=16000)

    assert len(store) == len(samples)

    original = peak_normalize(read_audio(samples[0].path))
    cached = store.get(0)

    assert cached.shape == original.shape
    # float16 round trip: ~3 decimal digits is far below the recording noise floor.
    assert np.allclose(cached, original, atol=1e-3)


def test_dataset_from_store_matches_dataset_from_files(corpus, tmp_path):
    samples = load_manifest(corpus)
    store = WaveformStore.build(samples, tmp_path / "cache", sample_rate=16000)

    from_files = VoiceSpoofDataset(samples=samples, max_seconds=1.0, train=False)
    from_store = VoiceSpoofDataset(store=store, max_seconds=1.0, train=False)

    assert len(from_files) == len(from_store)

    for index in range(len(samples)):
        assert from_files[index]["label"] == from_store[index]["label"]
        assert torch.allclose(
            from_files[index]["input_values"],
            from_store[index]["input_values"],
            atol=1e-3,
        )


def test_store_rejects_a_sample_rate_mismatch(corpus, tmp_path):
    store = WaveformStore.build(
        load_manifest(corpus), tmp_path / "cache", sample_rate=16000
    )

    with pytest.raises(ValueError, match="cache was built"):
        VoiceSpoofDataset(store=store, target_sample_rate=8000)


def test_incomplete_cache_directory_raises(tmp_path):
    (tmp_path / "cache").mkdir()

    with pytest.raises(FileNotFoundError, match="cache incomplete"):
        WaveformStore(tmp_path / "cache")


# ---------------------------------------------------------------------------
# parallel cache building
# ---------------------------------------------------------------------------


def test_parallel_cache_is_identical_to_serial(corpus, tmp_path):
    """
    The only thing that matters here: a faster builder that produces different
    bytes is worse than a slow one. imap preserves submission order, so the
    flat array and its offset index must line up exactly either way.
    """

    samples = load_manifest(corpus)

    serial = WaveformStore.build(samples, tmp_path / "serial", workers=0)
    parallel = WaveformStore.build(samples, tmp_path / "parallel", workers=2)

    assert len(serial) == len(parallel)
    assert np.array_equal(serial.offsets, parallel.offsets)
    assert np.array_equal(serial.lengths, parallel.lengths)
    assert np.array_equal(serial.labels, parallel.labels)

    for index in range(len(samples)):
        assert np.array_equal(serial.get(index), parallel.get(index)), (
            f"clip {index} differs between serial and parallel builds"
        )

    # And the raw files themselves.
    a = (tmp_path / "serial" / WaveformStore.DATA_NAME).read_bytes()
    b = (tmp_path / "parallel" / WaveformStore.DATA_NAME).read_bytes()
    assert a == b, "cache files are not byte-identical"


def test_parallel_cache_preserves_order_with_varied_lengths(tmp_path):
    """
    Clips in the fixture have different durations, so a reordering would show
    up as mismatched offsets rather than merely shuffled content.
    """

    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    samples = []
    for index in range(12):
        seconds = 0.5 + index * 0.3
        t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
        path = audio_dir / f"clip_{index}.flac"
        sf.write(path, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
                 16000, format="FLAC")
        samples.append(AudioSample(path=str(path), label=index % 2))

    store = WaveformStore.build(samples, tmp_path / "cache", workers=3)

    # Lengths must increase monotonically, matching the order they were given.
    lengths = [int(store.lengths[i]) for i in range(len(samples))]
    assert lengths == sorted(lengths), "parallel build reordered the clips"
