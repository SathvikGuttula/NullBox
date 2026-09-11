"""
The serving-side wrapper around :class:`AntiSpoofModel`.

Three things this layer is responsible for and the raw model is not:

1. Rebuilding the right architecture from a checkpoint. The checkpoint records
   which encoder it was trained on and whether the spectral branch existed;
   guessing wrong loads a subset of tensors and produces confident nonsense.

2. Telling the truth about its own state. If no checkpoint is present the
   detector still answers - the API has to boot without a trained model - but
   it reports ``model_status: "untrained"`` and a probability of 0.5. A demo
   that shows plausible-looking numbers from randomly initialised weights is
   worse than one that shows none.

3. Not loading a 360 MB encoder once per websocket connection. ``get_shared``
   hands out a process-wide instance.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch

from app.ml.calibration import ProbabilityCalibrator
from app.ml.config import DetectorConfig, resolve_path
from app.ml.model import AntiSpoofInference, AntiSpoofModel

logger = logging.getLogger("voxshield.detector")


class VoiceSpoofDetector:
    """Loads a checkpoint (or not) and scores waveforms."""

    _shared: "VoiceSpoofDetector | None" = None
    _lock = threading.Lock()

    def __init__(
        self,
        encoder_name: str = "facebook/wav2vec2-base",
        model_path: str | Path | None = None,
        device: str = "cuda",
        use_spectral: bool = True,
        target_samples: int | None = None,
        sample_rate: int = 16000,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sample_rate = sample_rate

        # Calibration lives here and nowhere else. The raw model output is not
        # a probability - see app/ml/calibration.py - and every consumer
        # (the /analyze endpoint, the streaming engine, the fusion layer)
        # reads synthetic_probability. Applying the map at the single point
        # where the model is called is what keeps them consistent, and stops
        # a second layer from calibrating an already-calibrated number.
        self.calibrator = ProbabilityCalibrator()
        self.calibration_source: str | None = None

        if calibration_path is not None:
            path = Path(calibration_path)
            if path.exists():
                try:
                    self.calibrator = ProbabilityCalibrator.load(path)
                    self.calibration_source = str(path)
                except Exception as exc:
                    logger.warning("could not read calibration %s: %s", path, exc)
            else:
                logger.warning(
                    "no calibration at %s - scores will be raw model output, "
                    "which is NOT a probability. Run "
                    "scripts/calibrate_detector.py.",
                    path,
                )

        # The encoder was trained on fixed-length windows; feeding it a much
        # shorter clip changes the pooling statistics. Default to the 4 s
        # training window and tile shorter input up to it.
        self.target_samples = target_samples or int(sample_rate * 4.0)

        checkpoint = None
        architecture: dict = {}

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                architecture = checkpoint.get("architecture", {})
            else:
                # A bare state_dict from an older run. Fall back to the caller's
                # arguments and warn, because there is no way to verify them.
                architecture = {}

        self.encoder_name = architecture.get("encoder_name", encoder_name)
        self.use_spectral = architecture.get("use_spectral", use_spectral)

        self.model = AntiSpoofModel(
            encoder_name=self.encoder_name,
            use_spectral=self.use_spectral,
            spectral_dim=architecture.get("spectral_dim", 128),
            hidden_dim=architecture.get("hidden_dim", 256),
            sample_rate=architecture.get("sample_rate", sample_rate),
        )

        self.trained = False
        self.checkpoint_metadata: dict = {}

        if checkpoint is not None:
            state = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            missing, unexpected = self.model.load_state_dict(state, strict=False)

            if missing or unexpected:
                raise RuntimeError(
                    f"checkpoint does not match the model architecture\n"
                    f"  missing keys:    {len(missing)} (e.g. {list(missing)[:3]})\n"
                    f"  unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:3]})\n"
                    f"Rebuild the detector with the architecture the checkpoint records."
                )

            self.trained = True
            self.checkpoint_metadata = (
                checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
            )

        self.inference = AntiSpoofInference(self.model, device=self.device)

    # -- shared instance ---------------------------------------------------

    @classmethod
    def get_shared(cls, config: DetectorConfig | None = None) -> "VoiceSpoofDetector":
        """
        Process-wide detector.

        Constructing one per websocket connection would download and
        instantiate the encoder every time a call starts - hundreds of MB of
        RAM per concurrent call and several seconds of latency before the
        first window is scored.
        """

        if cls._shared is None:
            with cls._lock:
                if cls._shared is None:
                    config = config or DetectorConfig()
                    cls._shared = cls(
                        encoder_name=config.encoder_name,
                        model_path=resolve_path(config.model_path),
                        use_spectral=config.use_spectral,
                        sample_rate=config.sample_rate,
                        target_samples=int(
                            config.sample_rate * config.training_window_seconds
                        ),
                        calibration_path=resolve_path(config.calibration_path),
                    )
        return cls._shared

    @classmethod
    def reset_shared(cls) -> None:
        cls._shared = None

    # -- scoring -----------------------------------------------------------

    def _prepare(self, audio: np.ndarray) -> torch.Tensor:
        """Peak-normalise and force the training window length."""

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            audio = audio / peak

        length = audio.shape[0]

        if length > self.target_samples:
            offset = (length - self.target_samples) // 2
            audio = audio[offset : offset + self.target_samples]
        elif length < self.target_samples:
            repeats = int(np.ceil(self.target_samples / max(length, 1)))
            audio = np.tile(audio, repeats)[: self.target_samples]

        return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

    def predict(self, audio: np.ndarray, sample_rate: int | None = None) -> dict:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            # 0.5 / 0.5, not 0.0 / 0.0. The two probabilities are a softmax
            # over two classes and must sum to 1; a consumer that reads
            # bonafide_probability would otherwise see 0.0 and conclude the
            # caller is definitely synthetic.
            return {
                "synthetic_probability": 0.5,
                "bonafide_probability": 0.5,
                "confidence": 0.0,
                "model_status": "empty",
            }

        if not self.trained:
            # Random weights produce a meaningless number. Say so rather than
            # dressing it up as a prediction.
            return {
                "synthetic_probability": 0.5,
                "bonafide_probability": 0.5,
                "confidence": 0.0,
                "model_status": "untrained",
            }

        if sample_rate is not None and sample_rate != self.sample_rate:
            import torchaudio.functional as AF

            audio = AF.resample(
                torch.from_numpy(audio), sample_rate, self.sample_rate
            ).numpy()

        result = self.inference.predict(self._prepare(audio))

        raw = float(result.spoof_probability)
        calibrated = self.calibrator.calibrate(raw)

        return {
            # The calibrated number is the one that means "chance this is
            # synthetic". The raw score is kept alongside it because it is
            # what the checkpoint actually emitted, and a support question
            # about a decision is unanswerable without it.
            "synthetic_probability": calibrated,
            "bonafide_probability": 1.0 - calibrated,
            "raw_probability": raw,
            "confidence": max(calibrated, 1.0 - calibrated),
            "calibrated": not self.calibrator.is_identity,
            "model_status": "neural",
        }

    def predict_file(self, path: str | Path) -> dict:
        from app.ml.dataset import read_audio

        return self.predict(read_audio(path, self.sample_rate), self.sample_rate)

    def describe(self) -> dict:
        return {
            "encoder": self.encoder_name,
            "spectral_branch": self.use_spectral,
            "device": self.device,
            "trained": self.trained,
            "window_seconds": self.target_samples / self.sample_rate,
            "calibration": {
                **self.calibrator.to_dict(),
                "source": self.calibration_source,
            },
            **self.checkpoint_metadata,
        }


# The name used by app.audio.feature_pipeline and the inference tests. Kept as
# an alias so both spellings resolve to one implementation instead of one of
# them raising ImportError at startup, which is what was happening before.
BaselineNeuralDetector = VoiceSpoofDetector
