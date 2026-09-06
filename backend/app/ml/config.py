from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:

    sample_rate: int = 16000

    minimum_audio_seconds: float = 1.0

    inference_window_seconds: float = 3.0

    inference_hop_seconds: float = 1.0

    smoothing_alpha: float = 0.35

    suspicious_threshold: float = 0.60

    high_risk_threshold: float = 0.85

    encoder_name: str = (
        "facebook/wav2vec2-base"
    )

    model_path: str = (
        "models/antispoof.pt"
    )