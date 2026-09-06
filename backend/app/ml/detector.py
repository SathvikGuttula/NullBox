from __future__ import annotations

import numpy as np
import torch

from app.ml.model import (
    AntiSpoofInference,
    AntiSpoofModel,
)


class VoiceSpoofDetector:

    def __init__(
        self,
        encoder_name: str = (
            "facebook/wav2vec2-base"
        ),
        model_path: str | None = None,
        device: str = "cuda",
    ):

        self.device = (
            device
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            AntiSpoofModel(
                encoder_name=encoder_name
            )
        )

        if model_path:

            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )

            self.model.load_state_dict(
                checkpoint
            )

        self.inference = (
            AntiSpoofInference(
                self.model,
                device=self.device,
            )
        )

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:

        if audio.size == 0:

            return {
                "synthetic_probability": 0.0,
                "bonafide_probability": 0.0,
                "confidence": 0.0,
                "model_status": "empty",
            }

        tensor = torch.from_numpy(
            audio
        ).float()

        if tensor.ndim == 1:

            tensor = tensor.unsqueeze(0)

        result = (
            self.inference.predict(
                tensor
            )
        )

        confidence = max(
            result.spoof_probability,
            result.bonafide_probability,
        )

        return {
            "synthetic_probability":
                result.spoof_probability,

            "bonafide_probability":
                result.bonafide_probability,

            "confidence":
                confidence,

            "model_status":
                "neural",
        }