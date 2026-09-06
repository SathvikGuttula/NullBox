from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass
class AudioSample:

    path: str

    label: int

    attack: str | None = None


class VoiceSpoofDataset(
    Dataset
):

    def __init__(
        self,
        samples: list[AudioSample],
        target_sample_rate: int = 16000,
        max_seconds: float = 4.0,
    ):

        self.samples = samples

        self.target_sample_rate = (
            target_sample_rate
        )

        self.max_samples = int(
            target_sample_rate
            * max_seconds
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):

        sample = (
            self.samples[index]
        )

        audio, sample_rate = (
            sf.read(
                sample.path,
                dtype="float32",
            )
        )

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1,
            )

        audio = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        if sample_rate != (
            self.target_sample_rate
        ):

            raise ValueError(
                f"Unexpected sample rate "
                f"{sample_rate} for "
                f"{sample.path}"
            )

        audio = self._crop_or_pad(
            audio
        )

        return {
            "input_values": audio,
            "label": torch.tensor(
                sample.label,
                dtype=torch.long,
            ),
            "path": sample.path,
            "attack": sample.attack,
        }

    def _crop_or_pad(
        self,
        audio: torch.Tensor,
    ):

        if len(audio) > self.max_samples:

            return audio[
                : self.max_samples
            ]

        if len(audio) < self.max_samples:

            padding = (
                self.max_samples
                - len(audio)
            )

            audio = torch.nn.functional.pad(
                audio,
                (
                    0,
                    padding,
                ),
            )

        return audio