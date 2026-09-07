
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


@dataclass
class AudioSample:
    path: str
    label: int
    attack: str | None = None


class VoiceSpoofDataset(Dataset):
    def __init__(
        self,
        samples: list[AudioSample],
        target_sample_rate: int = 16000,
        max_seconds: float = 4.0,
    ):
        self.samples = samples
        self.target_sample_rate = target_sample_rate
        self.max_samples = int(target_sample_rate * max_seconds)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        # Robust FLAC loading
        audio, sample_rate = torchaudio.load(sample.path)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)

        # Resample if required
        if sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                sample_rate,
                self.target_sample_rate,
            )

        # Crop / pad to exactly 4 seconds
        if audio.numel() > self.max_samples:
            audio = audio[:self.max_samples]
        elif audio.numel() < self.max_samples:
            audio = F.pad(
                audio,
                (0, self.max_samples - audio.numel()),
            )

        return {
            "input_values": audio,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": sample.path,
            "attack": sample.attack,
        }
