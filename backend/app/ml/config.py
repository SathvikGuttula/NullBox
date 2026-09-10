"""
Configuration for the VoxShield ML subsystem.

``DetectorConfig`` covers live inference. ``TrainingConfig`` covers offline
training and is written to ``experiments/<id>/config.json`` on every run, so a
checkpoint can always be traced back to the exact recipe that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


# backend/app/ml/config.py -> backend/app/ml -> backend/app -> backend -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """
    Interpret a relative path against the repository root, not the cwd.

    The API is normally started from ``backend/`` while checkpoints live in
    ``<repo>/models/``. Resolving against the cwd would look for
    ``backend/models/...`` and quietly find nothing, so the detector would
    report itself untrained even with a checkpoint sitting right there.
    """

    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class DetectorConfig:
    """Runtime settings for the streaming detector."""

    sample_rate: int = 16000

    minimum_audio_seconds: float = 1.0

    # The streaming window is deliberately shorter than the 4 s training
    # window: latency matters live. The model still sees 4 s of samples
    # because short windows are tiled up to the training length before
    # inference - see StreamingInferenceEngine. Do not silently change one of
    # these without the other.
    inference_window_seconds: float = 3.0
    inference_hop_seconds: float = 1.0
    training_window_seconds: float = 4.0

    smoothing_alpha: float = 0.35

    # -- decision thresholds ----------------------------------------------
    #
    # These are DERIVED, not chosen. scripts/calibrate_detector.py fits the
    # calibration below and reads the operating points off the held-out half
    # of the ASVspoof 2019 LA evaluation set (35,619 utterances, 13 attack
    # types the model never trained on). They apply to the CALIBRATED
    # probability - against the raw model output they mean nothing.
    #
    #   p >= 0.137   2.01% false alarm, detects 96.83% of attacks
    #   p >= 0.500   0.79% false alarm, detects 94.95% of attacks
    #
    # The previous values (0.60 / 0.85) were engineering guesses, and worse,
    # nothing read them: the live risk levels came from RiskThresholds'
    # own defaults. They are wired through StreamingInferenceEngine now.
    suspicious_threshold: float = 0.137
    high_risk_threshold: float = 0.50

    # The point at which a single window is called synthetic. Same basis.
    decision_threshold: float = 0.50

    encoder_name: str = "facebook/wav2vec2-base"
    use_spectral: bool = True

    model_path: str = "models/voxshield_antispoof.pt"

    # Committed under results/ rather than models/, which is gitignored - a
    # calibration that vanishes on clone is worse than none, because the
    # detector then silently reports raw scores as though they were
    # probabilities. resolve_path handles the repo-root lookup.
    calibration_path: str = "results/calibration/antispoof_calibration.json"


@dataclass
class TrainingConfig:
    """Everything that defines one training run."""

    # -- data --------------------------------------------------------------
    train_manifest: str = "datasets/manifests/train.csv"
    validation_manifest: str = "datasets/manifests/validation.csv"
    cache_dir: str | None = None
    sample_rate: int = 16000
    max_seconds: float = 4.0
    limit_train: int | None = None
    limit_validation: int | None = None
    balance_classes: bool = True

    # -- model -------------------------------------------------------------
    encoder_name: str = "facebook/wav2vec2-base"
    use_spectral: bool = True
    spectral_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2

    # -- optimisation ------------------------------------------------------
    epochs: int = 4
    batch_size: int = 8
    gradient_accumulation: int = 2
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05

    # -- staged fine-tuning ------------------------------------------------
    # Stage 1 trains only the head on frozen SSL features: fast, stable, and
    # it stops the randomly initialised classifier from wrecking the encoder
    # with large early gradients. Stage 2 then unfreezes the top layers.
    freeze_epochs: int = 1
    unfreeze_top_layers: int = 4
    freeze_feature_extractor: bool = True
    gradient_checkpointing: bool = False

    # -- augmentation ------------------------------------------------------
    augment: bool = True
    telephone_probability: float = 0.35
    noise_probability: float = 0.30
    gain_probability: float = 0.30
    reverb_probability: float = 0.15
    codec_probability: float = 0.20

    # -- runtime -----------------------------------------------------------
    device: str = "cuda"
    precision: str = "auto"      # auto | fp16 | bf16 | fp32
    num_workers: int = 4
    seed: int = 1234
    log_every: int = 25
    probe_steps: int = 12
    validate_every: int = 1
    early_stopping_patience: int = 2

    # -- outputs -----------------------------------------------------------
    experiment_id: str = "exp001"
    experiments_dir: str = "experiments"
    model_out: str = "models/voxshield_antispoof.pt"

    # -- provenance --------------------------------------------------------
    git_commit: str | None = None
    notes: str = ""

    extra: dict = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return Path(self.experiments_dir) / self.experiment_id

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * max(self.gradient_accumulation, 1)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_args(cls, args) -> "TrainingConfig":
        """Build from an argparse namespace, ignoring unset (None) values."""

        known = {f.name for f in fields(cls)}
        values = {
            key: value
            for key, value in vars(args).items()
            if key in known and value is not None
        }
        return cls(**values)

    def summary(self) -> dict:
        """The block printed at the top of a run."""

        return {
            "experiment": self.experiment_id,
            "encoder": self.encoder_name,
            "spectral branch": "on" if self.use_spectral else "off",
            "window": f"{self.max_seconds}s @ {self.sample_rate} Hz",
            "epochs": self.epochs,
            "batch size": f"{self.batch_size} x {self.gradient_accumulation} accum "
                          f"= {self.effective_batch_size} effective",
            "encoder lr": self.encoder_lr,
            "head lr": self.head_lr,
            "freeze epochs": self.freeze_epochs,
            "unfreeze top layers": self.unfreeze_top_layers,
            "augmentation": "on" if self.augment else "off",
            "class balancing": "on" if self.balance_classes else "off",
            "precision": self.precision,
            "workers": self.num_workers,
            "seed": self.seed,
            "checkpoint": self.model_out,
        }
