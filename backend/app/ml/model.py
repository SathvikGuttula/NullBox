
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class AntiSpoofOutput:
    spoof_probability: float
    bonafide_probability: float
    logits: list[float]


class AntiSpoofModel(nn.Module):
    """
    Neural anti-spoofing classifier.

    Architecture:

        waveform
           ↓
        pretrained speech encoder
           ↓
        mean pooling
           ↓
        dropout
           ↓
        binary classifier
           ↓
        bonafide / spoof
    """

    def __init__(
        self,
        encoder_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            encoder_name
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                2,
            ),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        if attention_mask is not None:

            mask = attention_mask.unsqueeze(-1)

            hidden = hidden * mask

            pooled = (
                hidden.sum(dim=1)
                /
                mask.sum(
                    dim=1
                ).clamp(min=1)
            )

        else:

            pooled = hidden.mean(
                dim=1
            )

        return self.classifier(
            pooled
        )


class AntiSpoofInference:

    def __init__(
        self,
        model: AntiSpoofModel,
        device: str = "cpu",
    ):

        self.model = model.to(device)

        self.device = device

        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        input_values: torch.Tensor,
    ) -> AntiSpoofOutput:

        input_values = (
            input_values.to(
                self.device
            )
        )

        logits = self.model(
            input_values
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        bonafide = (
            probabilities[0]
            .item()
        )

        spoof = (
            probabilities[1]
            .item()
        )

        return AntiSpoofOutput(
            spoof_probability=spoof,
            bonafide_probability=bonafide,
            logits=logits[0]
            .detach()
            .cpu()
            .tolist(),
        )