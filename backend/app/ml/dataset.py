from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
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
        self.max_samples = int(
            target_sample_rate * max_seconds
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        audio, sample_rate = sf.read(
            sample.path,
            dtype="float32",
        )

        # Stereo / multi-channel -> mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        # Resample -> 16 kHz
        if sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        # Exactly 4 seconds
        audio = self._crop_or_pad(audio)

        return {
            "input_values": audio,
            "label": torch.tensor(
                sample.label,
                dtype=torch.long,
            ),
            "path": sample.path,
            "attack": sample.attack,
        }

    def _crop_or_pad(self, audio: torch.Tensor):

        # Crop
        if len(audio) > self.max_samples:
            return audio[: self.max_samples]

        # Zero-pad
        if len(audio) < self.max_samples:
            padding = self.max_samples - len(audio)

            audio = torch.nn.functional.pad(
                audio,
                (0, padding),
            )

        return audio