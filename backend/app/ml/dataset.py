"""
Dataset plumbing for the VoxShield anti-spoof detector.

Two loading paths are provided and they share one interface:

    VoiceSpoofDataset(samples=..., ...)          decode FLAC/WAV on demand
    VoiceSpoofDataset(store=WaveformStore(...))  read from a memmapped cache

The cache exists because FLAC decoding, not the GPU, is the bottleneck on a
laptop: a frozen-encoder epoch over ASVspoof 2019 LA train spends most of its
wall clock inside libsndfile. ``scripts/cache_dataset.py`` decodes the corpus
once into a flat float16 array; after that an epoch is bounded by the GPU.

Why soundfile and not torchaudio.load
-------------------------------------
torchaudio >= 2.9 routes ``torchaudio.load`` through TorchCodec, which is a
separate install and is not present in this environment - the call raises
``ImportError: TorchCodec is required for load_with_torchcodec``. soundfile
reads FLAC natively through libsndfile, is already a dependency, is faster for
this workload, and cannot break when torchaudio reshuffles its I/O backends.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# manifest records
# ---------------------------------------------------------------------------


@dataclass
class AudioSample:
    """One row of a manifest CSV."""

    path: str
    label: int          # 0 = bonafide, 1 = spoof
    attack: str | None = None
    split: str | None = None
    speaker: str | None = None


def load_manifest(
    manifest_path: str | Path,
    split: str | None = None,
    limit: int | None = None,
) -> list[AudioSample]:
    """
    Read a manifest CSV into ``AudioSample`` records.

    Required columns: ``path``, ``label``. Optional: ``split``, ``attack``,
    ``speaker``. Rows whose ``split`` does not match are dropped when ``split``
    is given, which is how one combined manifest can serve all three sets.

    ``limit`` takes an evenly strided subsample across the whole file, not the
    first N rows. Protocol files are grouped by speaker and by attack, so the
    head of one is a badly skewed sample - often a single speaker, sometimes a
    single class, which makes EER undefined and a ``--limit-train`` experiment
    meaningless.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    if manifest_path.stat().st_size == 0:
        raise ValueError(
            f"manifest is empty: {manifest_path}\n"
            "Run backend/scripts/build_manifest.py first."
        )

    samples: list[AudioSample] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError(
                f"{manifest_path} has no 'path' column "
                f"(found: {reader.fieldnames})"
            )

        for row in reader:
            row_split = (row.get("split") or "").strip() or None

            if split is not None and row_split != split:
                continue

            samples.append(
                AudioSample(
                    path=row["path"],
                    label=int(row["label"]),
                    attack=(row.get("attack") or "").strip() or None,
                    split=row_split,
                    speaker=(row.get("speaker") or "").strip() or None,
                )
            )

    if not samples:
        raise ValueError(
            f"no rows in {manifest_path}"
            + (f" for split={split!r}" if split else "")
        )

    if limit is not None and 0 < limit < len(samples):
        samples = stratified_subsample(samples, limit)

    return samples


def stratified_subsample(samples: list[AudioSample], limit: int) -> list[AudioSample]:
    """
    Take ``limit`` samples preserving both the class ratio and attack coverage.

    Strata are ``(label, attack)``, so bonafide forms one group and each
    spoofing system forms its own.

    Neither a head slice nor a plain uniform stride is good enough. Protocol
    files are grouped by speaker and cycle through attacks A01..A06, so
    ``head(n)`` can return a single attack family and a uniform stride can
    alias with the cycle and return a single class. Either one silently turns
    a ``--limit-train`` experiment into a different problem: the subset no
    longer contains the attacks the full run would face, so its EER says
    nothing about the full run's.

    Striding *within* each stratum is immune to both, keeps every attack
    represented, and preserves the real class imbalance - so a small run
    behaves like a scaled-down version of the real one.
    """

    groups: dict[tuple[int, str], list[AudioSample]] = {}
    for sample in samples:
        key = (sample.label, sample.attack or "")
        groups.setdefault(key, []).append(sample)

    keys = sorted(groups)
    by_size = sorted(keys, key=lambda key: (-len(groups[key]), key))

    # When the limit is smaller than the number of strata, one-per-stratum
    # cannot fit and some strata must be dropped. Keep the largest, but take
    # the largest stratum of each label first - losing a whole class makes EER
    # undefined, which is worse than losing one attack family.
    if limit < len(keys):
        kept: list[tuple[int, str]] = []
        seen_labels: set[int] = set()

        for key in by_size:
            if key[0] not in seen_labels and len(kept) < limit:
                kept.append(key)
                seen_labels.add(key[0])

        for key in by_size:
            if len(kept) >= limit:
                break
            if key not in kept:
                kept.append(key)

        keys = sorted(kept)
        by_size = sorted(keys, key=lambda key: (-len(groups[key]), key))

    available = sum(len(groups[key]) for key in keys)

    # Proportional allocation, at least one from every stratum we kept.
    quotas = {
        key: max(1, int(limit * len(groups[key]) / available)) for key in keys
    }

    # Largest-remainder correction, biased towards the biggest strata so
    # rounding never starves a rare attack down to nothing. The loop stops
    # when a full pass changes nothing, which is the only correct termination
    # condition once quotas hit their per-stratum ceilings.
    drift = limit - sum(quotas.values())

    while drift != 0:
        progressed = False

        for key in by_size:
            if drift == 0:
                break
            if drift > 0 and quotas[key] < len(groups[key]):
                quotas[key] += 1
                drift -= 1
                progressed = True
            elif drift < 0 and quotas[key] > 1:
                quotas[key] -= 1
                drift += 1
                progressed = True

        if not progressed:
            break

    chosen: list[AudioSample] = []
    for key in keys:
        group = groups[key]
        take = min(quotas[key], len(group))
        stride = len(group) / take
        chosen.extend(group[int(i * stride)] for i in range(take))

    return chosen


def describe(samples: list[AudioSample]) -> dict:
    """Class balance and attack coverage, for printing before a run starts."""

    bonafide = sum(1 for s in samples if s.label == 0)
    spoof = len(samples) - bonafide

    attacks: dict[str, int] = {}
    for sample in samples:
        if sample.label == 1 and sample.attack:
            attacks[sample.attack] = attacks.get(sample.attack, 0) + 1

    return {
        "total": len(samples),
        "bonafide": bonafide,
        "spoof": spoof,
        "spoof_ratio": spoof / len(samples) if samples else 0.0,
        "attacks": dict(sorted(attacks.items())),
    }


# ---------------------------------------------------------------------------
# audio loading
# ---------------------------------------------------------------------------


def read_audio(path: str | Path, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Decode any libsndfile-supported file to mono float32 at the target rate.

    Resampling uses torchaudio's functional resampler (sinc interpolation),
    which is both accurate and fast enough that it never shows up in a profile
    for the handful of ASVspoof files that are not already 16 kHz.
    """

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        import torchaudio.functional as AF

        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        audio = AF.resample(tensor, sample_rate, target_sample_rate).numpy()

    return np.ascontiguousarray(audio, dtype=np.float32)


def fit_length(
    audio: np.ndarray,
    target_samples: int,
    random_crop: bool = False,
    rng: random.Random | None = None,
) -> np.ndarray:
    """
    Force ``audio`` to exactly ``target_samples``.

    Long clips are cropped. During training the crop offset is random, which
    is free augmentation and stops the model from keying on whatever happens
    to sit in the first four seconds of every file. During evaluation the crop
    is deterministic (from the start) so that a score is reproducible.

    Short clips are tiled rather than zero-padded. Zero padding teaches the
    model that "silence at the end" is a class cue, and because bonafide and
    spoof clips have different length distributions in ASVspoof, that cue
    leaks the label. Tiling keeps the signal statistics honest.
    """

    length = audio.shape[0]

    if length == target_samples:
        return audio

    if length > target_samples:
        if random_crop:
            picker = rng or random
            offset = picker.randint(0, length - target_samples)
        else:
            offset = 0
        return audio[offset : offset + target_samples]

    repeats = int(np.ceil(target_samples / max(length, 1)))
    tiled = np.tile(audio, repeats)[:target_samples]
    return np.ascontiguousarray(tiled)


def peak_normalize(audio: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > epsilon:
        audio = audio / peak
    return audio


# ---------------------------------------------------------------------------
# memmapped waveform cache
# ---------------------------------------------------------------------------


class WaveformStore:
    """
    A decoded corpus held as one flat float16 array plus an offset index.

    Layout on disk::

        <root>/waveforms.f16      flat samples, all files concatenated
        <root>/index.npz          offsets, lengths, labels
        <root>/meta.json          paths, attacks, splits, sample rate

    float16 costs ~2 bytes/sample: ASVspoof 2019 LA train (about 24 hours of
    audio) lands near 2.8 GB, which memmaps comfortably in 15 GB of RAM and
    lets the OS page cache do the rest. Precision loss is irrelevant here -
    the waveform is peak-normalised to [-1, 1] and fp16 keeps ~3 decimal
    digits across that range, far below the noise floor of the recordings.
    """

    DATA_NAME = "waveforms.f16"
    INDEX_NAME = "index.npz"
    META_NAME = "meta.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

        index_path = self.root / self.INDEX_NAME
        meta_path = self.root / self.META_NAME
        data_path = self.root / self.DATA_NAME

        for path in (index_path, meta_path, data_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"waveform cache incomplete, missing {path.name} in {self.root}\n"
                    "Build it with backend/scripts/cache_dataset.py"
                )

        index = np.load(index_path)
        self.offsets = index["offsets"]
        self.lengths = index["lengths"]
        self.labels = index["labels"]

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.sample_rate = int(self.meta.get("sample_rate", 16000))
        self.paths: list[str] = self.meta["paths"]
        self.attacks: list[str | None] = self.meta.get("attacks") or [None] * len(self.paths)
        self.splits: list[str | None] = self.meta.get("splits") or [None] * len(self.paths)

        self.total_samples = int(self.meta["total_samples"])
        self.data_path = data_path
        self._data: np.memmap | None = None

    # -- process-safe memmap handling --------------------------------------
    #
    # np.memmap subclasses ndarray, so pickling one copies the WHOLE buffer by
    # value. Windows has no fork: DataLoader pickles the dataset - and with it
    # this store - into every worker process. Holding the memmap as a plain
    # attribute therefore turned a 2.8 GB cache into 2.8 GB per worker, which
    # on 15 GB of RAM means swapping or an outright MemoryError, and made the
    # cache slower than decoding FLAC.
    #
    # Instead the handle is opened lazily, per process, and dropped from the
    # pickled state. Each worker maps the same file itself; the OS page cache
    # is shared, so the real memory cost is paid once for all workers.

    @property
    def data(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(
                self.data_path,
                dtype=np.float16,
                mode="r",
                shape=(self.total_samples,),
            )
        return self._data

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._data = None

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        return np.asarray(self.data[start : start + length], dtype=np.float32)

    def samples(self) -> list[AudioSample]:
        return [
            AudioSample(
                path=self.paths[i],
                label=int(self.labels[i]),
                attack=self.attacks[i],
                split=self.splits[i],
            )
            for i in range(len(self))
        ]

    @property
    def size_bytes(self) -> int:
        return (self.root / self.DATA_NAME).stat().st_size

    @staticmethod
    def build(
        samples: list[AudioSample],
        root: str | Path,
        sample_rate: int = 16000,
        max_seconds: float | None = None,
        progress=None,
    ) -> "WaveformStore":
        """
        Decode every sample once and write the cache.

        ``max_seconds`` caps how much of each file is stored. Leave it at None
        to keep the full clip so that random cropping still has room to move;
        set it when disk is tight.
        """

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        data_path = root / WaveformStore.DATA_NAME

        offsets = np.zeros(len(samples), dtype=np.int64)
        lengths = np.zeros(len(samples), dtype=np.int64)
        labels = np.zeros(len(samples), dtype=np.int64)

        max_samples = int(sample_rate * max_seconds) if max_seconds else None

        cursor = 0

        with data_path.open("wb") as handle:
            for index, sample in enumerate(samples):
                audio = read_audio(sample.path, sample_rate)
                audio = peak_normalize(audio)

                if max_samples is not None and audio.shape[0] > max_samples:
                    audio = audio[:max_samples]

                handle.write(audio.astype(np.float16).tobytes())

                offsets[index] = cursor
                lengths[index] = audio.shape[0]
                labels[index] = sample.label
                cursor += audio.shape[0]

                if progress is not None:
                    progress.update(1)

        np.savez(
            root / WaveformStore.INDEX_NAME,
            offsets=offsets,
            lengths=lengths,
            labels=labels,
        )

        (root / WaveformStore.META_NAME).write_text(
            json.dumps(
                {
                    "sample_rate": sample_rate,
                    "total_samples": int(cursor),
                    "count": len(samples),
                    "paths": [s.path for s in samples],
                    "attacks": [s.attack for s in samples],
                    "splits": [s.split for s in samples],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return WaveformStore(root)


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------


class VoiceSpoofDataset(Dataset):
    """
    Fixed-length mono 16 kHz waveforms with binary spoof labels.

    Parameters
    ----------
    samples
        Manifest records. Ignored when ``store`` is given.
    store
        A prebuilt :class:`WaveformStore`. Much faster; no FLAC decoding.
    max_seconds
        Window length fed to the model. 4.0 matches the ASVspoof convention.
    train
        Enables random cropping and augmentation. Off for validation/test so
        scores are deterministic.
    augment
        Callable applied to the tensor after cropping, e.g. ``AudioAugmenter``.
    seed
        Makes the random crops reproducible across runs.
    """

    def __init__(
        self,
        samples: list[AudioSample] | None = None,
        store: WaveformStore | None = None,
        target_sample_rate: int = 16000,
        max_seconds: float = 4.0,
        train: bool = False,
        augment=None,
        normalize: bool = True,
        seed: int = 1234,
    ) -> None:
        if store is None and samples is None:
            raise ValueError("provide either samples or store")

        self.store = store
        self.samples = store.samples() if store is not None else list(samples or [])
        self.target_sample_rate = target_sample_rate
        self.max_seconds = max_seconds
        self.max_samples = int(target_sample_rate * max_seconds)
        self.train = train
        self.augment = augment
        self.normalize = normalize
        self.seed = seed

        if store is not None and store.sample_rate != target_sample_rate:
            raise ValueError(
                f"cache was built at {store.sample_rate} Hz "
                f"but {target_sample_rate} Hz was requested"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([s.label for s in self.samples], dtype=np.int64)

    def class_counts(self) -> tuple[int, int]:
        labels = self.labels
        return int((labels == 0).sum()), int((labels == 1).sum())

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        if self.store is not None:
            audio = self.store.get(index)
        else:
            audio = read_audio(sample.path, self.target_sample_rate)

        if self.normalize:
            audio = peak_normalize(audio)

        # In training the crop offset comes from the process-global RNG, which
        # ``seed_dataloader_worker`` reseeds per worker and per epoch - so the
        # crops differ every epoch instead of being frozen at dataset build
        # time. In eval the offset is always 0, so no RNG is consulted at all
        # and a score is byte-for-byte reproducible.
        audio = fit_length(
            audio,
            self.max_samples,
            random_crop=self.train,
            rng=None,
        )

        tensor = torch.from_numpy(np.ascontiguousarray(audio))

        if self.train and self.augment is not None:
            tensor = self.augment(tensor)

        return {
            "input_values": tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "index": index,
        }


def collate(batch: list[dict]) -> dict:
    """Stack a batch. All items are already the same length by construction."""

    return {
        "input_values": torch.stack([item["input_values"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def seed_dataloader_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` for DataLoader.

    Two jobs:

    1. Seeding. Windows spawns fresh processes for workers, so each one
       re-imports this module with an unseeded ``random`` and would otherwise
       draw identical crop offsets. torch gives every worker a distinct
       ``initial_seed()`` that also advances each epoch; deriving from it
       makes crops vary across workers and epochs while staying reproducible
       for a fixed run seed.

    2. Thread capping. Each worker inherits torch's default intra-op thread
       pool, sized to all 16 logical cores. Four workers then contend for 64
       threads on 16 cores while the main process is also trying to feed the
       GPU. One thread per worker is strictly faster here - the per-item work
       (a memmap read, a resample, a few biquads) is far too small to
       parallelise usefully.
    """

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_num_threads(1)


def balanced_sample_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Per-sample weights for ``WeightedRandomSampler``.

    ASVspoof 2019 LA train is roughly 90% spoof. Left alone, the model learns
    the prior instead of the signal and every threshold-free metric suffers.
    Weighting each class to equal total mass fixes that without discarding data.
    """

    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[labels]
    return torch.from_numpy(weights).double()
