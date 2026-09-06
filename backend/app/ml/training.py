from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW

from app.ml.model import AntiSpoofModel


class AntiSpoofTrainer:

    def __init__(
        self,
        model: AntiSpoofModel,
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):

        self.device = (
            device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(
            self.device
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = (
            nn.CrossEntropyLoss()
        )

    def train_step(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor,
    ):

        self.model.train()

        input_values = (
            input_values.to(
                self.device
            )
        )

        labels = labels.to(
            self.device
        )

        self.optimizer.zero_grad()

        logits = self.model(
            input_values
        )

        loss = self.loss_fn(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        accuracy = (
            predictions == labels
        ).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }