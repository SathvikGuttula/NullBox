import numpy as np
import torch
import torchaudio.functional as AF


class AudioPreprocessor:

    def __init__(
        self,
        target_sample_rate: int = 16000,
    ):
        self.target_sample_rate = (
            target_sample_rate
        )

    def normalize(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return audio

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        peak = np.max(
            np.abs(audio)
        )

        if peak > 1e-8:
            audio = audio / peak

        return np.clip(
            audio,
            -1.0,
            1.0,
        )

    def resample(
        self,
        audio: np.ndarray,
        source_sample_rate: int,
    ) -> np.ndarray:

        if (
            source_sample_rate
            == self.target_sample_rate
        ):
            return audio

        tensor = torch.from_numpy(
            audio
        ).float()

        resampled = AF.resample(
            tensor,
            source_sample_rate,
            self.target_sample_rate,
        )

        return resampled.numpy()

    def prepare(
        self,
        audio: np.ndarray,
        source_sample_rate: int,
    ) -> torch.Tensor:

        audio = self.normalize(
            audio
        )

        audio = self.resample(
            audio,
            source_sample_rate,
        )

        return torch.from_numpy(
            audio
        ).float()