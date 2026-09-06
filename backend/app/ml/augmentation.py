from __future__ import annotations

import random

import torch


class AudioAugmenter:

    def __init__(
        self,
        noise_probability: float = 0.30,
        gain_probability: float = 0.30,
        noise_level: float = 0.005,
    ):

        self.noise_probability = (
            noise_probability
        )

        self.gain_probability = (
            gain_probability
        )

        self.noise_level = (
            noise_level
        )

    def __call__(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:

        audio = audio.clone()

        if random.random() < (
            self.noise_probability
        ):

            noise = torch.randn_like(
                audio
            ) * self.noise_level

            audio = audio + noise

        if random.random() < (
            self.gain_probability
        ):

            gain = random.uniform(
                0.7,
                1.3,
            )

            audio = audio * gain

        return torch.clamp(
            audio,
            -1.0,
            1.0,
        )