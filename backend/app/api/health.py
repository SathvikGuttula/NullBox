"""
Liveness and a real system report.

    GET /api/v1/health          is the process up?
    GET /api/v1/status          is every part of it actually working?

``/health`` is the cheap one and answers instantly. ``/status`` is the one to
look at before demonstrating anything: it says whether the checkpoint loaded,
whether its scores are calibrated, which device inference is running on,
whether the speaker encoder is present, and which thresholds are in force.

Every one of those has been silently wrong at some point in this project - a
detector reporting itself untrained because a path resolved against the wrong
directory, a registry falling back to placeholder thresholds because the
calibration file was gitignored - and in each case the API kept answering
plausibly. A demo that cannot tell you which of its parts are real is worse
than one that is missing parts.

``/status`` deliberately does NOT construct anything. Loading the detector
takes about six seconds and 380 MB; a health endpoint that did that would
become a way to exhaust the box. It reports what is already loaded, and what
is on disk waiting to be.
"""

from __future__ import annotations

import importlib.util

from fastapi import APIRouter

from app.config import settings
from app.ml.calibration import ProbabilityCalibrator
from app.ml.config import DetectorConfig, resolve_path
from app.ml.detector import VoiceSpoofDetector
from app.ml.fusion import FusionWeights, PolicyThresholds

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


def _anti_spoof_status(config: DetectorConfig) -> dict:
    checkpoint = resolve_path(config.model_path)
    calibration = resolve_path(config.calibration_path)

    report: dict = {
        "checkpoint": str(checkpoint),
        "checkpoint_present": checkpoint.exists(),
        "checkpoint_mb": (
            round(checkpoint.stat().st_size / 1024**2, 1) if checkpoint.exists() else None
        ),
        "loaded": VoiceSpoofDetector._shared is not None,
        "thresholds": {
            "suspicious": config.suspicious_threshold,
            "high_risk": config.high_risk_threshold,
            "decision": config.decision_threshold,
            "basis": (
                "calibrated probability, read off the held-out half of the "
                "ASVspoof 2019 LA evaluation set"
            ),
        },
    }

    if calibration.exists():
        try:
            calibrator = ProbabilityCalibrator.load(calibration)
            metadata = calibrator.metadata
            report["calibration"] = {
                "present": True,
                "source": str(calibration),
                "scale": round(calibrator.scale, 4),
                "bias": round(calibrator.bias, 4),
                "prior": metadata.get("prior"),
                "held_out_utterances": metadata.get("n_holdout"),
                "detection_rate_at_0.5": metadata.get("at_half", {}).get(
                    "detection_rate"
                ),
                "false_alarm_at_0.5": metadata.get("at_half", {}).get(
                    "false_alarm_rate"
                ),
            }
        except Exception as exc:
            report["calibration"] = {"present": True, "error": str(exc)}
    else:
        report["calibration"] = {
            "present": False,
            "warning": (
                "No calibration found. Raw model scores are NOT probabilities - "
                "the equal-error point sits near 0.044, so judging at 0.5 misses "
                "roughly 12% of attacks. Run scripts/calibrate_detector.py."
            ),
        }

    # Only report the live device if something is already loaded; asking for it
    # would build the detector.
    if VoiceSpoofDetector._shared is not None:
        report["device"] = VoiceSpoofDetector._shared.device
        report["trained"] = VoiceSpoofDetector._shared.trained
        report["checkpoint_metadata"] = VoiceSpoofDetector._shared.checkpoint_metadata

    return report


def _speaker_status() -> dict:
    from app.api import speakers as speakers_module

    installed = importlib.util.find_spec("speechbrain") is not None

    # get_registry() is what populates the source, so read the attribute off
    # the module afterwards. Binding it with a from-import captures None on
    # the first call, before the registry has been built.
    registry = speakers_module.get_registry()
    calibration_source = speakers_module._calibration_source

    report: dict = {
        "speechbrain_installed": installed,
        "encoder": "speechbrain/spkrec-ecapa-voxceleb",
        "enrolled": len(registry),
        "identities": sorted(registry.profiles),
        "thresholds": {
            "match": round(registry.thresholds.match, 4),
            "no_match": round(registry.thresholds.no_match, 4),
            "version": registry.thresholds.version,
            "calibrated": registry.thresholds.calibrated,
            "source": calibration_source,
        },
    }

    if not installed:
        report["warning"] = (
            "speechbrain is not installed, so the speaker branch is off and "
            "every call degrades to anti-spoof only. pip install speechbrain"
        )

    rejected = getattr(registry, "rejected", None)
    if rejected:
        report["rejected_profiles"] = [
            {"identity": name, "reason": reason} for name, reason in rejected
        ]

    return report


@router.get("/status")
async def system_status() -> dict:
    """A full report on what is loaded, calibrated, and ready."""

    config = DetectorConfig()
    weights = FusionWeights()
    policy = PolicyThresholds()

    anti_spoof = _anti_spoof_status(config)

    try:
        speaker = _speaker_status()
    except Exception as exc:
        speaker = {"error": str(exc), "speechbrain_installed": False}

    warnings: list[str] = []

    if not anti_spoof["checkpoint_present"]:
        warnings.append(
            "No trained checkpoint. The detector will report model_status "
            "'untrained' and contribute nothing to the risk score."
        )
    if not anti_spoof["calibration"].get("present"):
        warnings.append(anti_spoof["calibration"]["warning"])
    if speaker.get("warning"):
        warnings.append(speaker["warning"])
    if not weights.calibrated:
        warnings.append(
            "Fusion weights are uncalibrated placeholders. The risk score is "
            "advisory - do not automate an irreversible action on it."
        )
    if speaker.get("enrolled") == 0:
        warnings.append(
            "No speakers enrolled, so identity verification cannot run. A low "
            "risk score therefore says nothing about who is speaking."
        )

    return {
        "service": settings.app_name,
        "version": "0.2.0",
        "ready": anti_spoof["checkpoint_present"],
        "anti_spoof": anti_spoof,
        "speaker": speaker,
        "fusion": {
            "weights_version": weights.version,
            "calibrated": weights.calibrated,
            "weights": {
                "synthetic_speech": weights.synthetic_speech,
                "identity_mismatch": weights.identity_mismatch,
                "context": weights.context,
                "audio_forensics": weights.audio_forensics,
                "prosody": weights.prosody,
                "nlp": weights.nlp,
            },
            "bands": {
                "warn": policy.warn,
                "verify": policy.verify,
                "escalate": policy.escalate,
                "version": policy.version,
            },
        },
        "audio": {
            "sample_rate": config.sample_rate,
            "inference_window_seconds": config.inference_window_seconds,
            "inference_hop_seconds": config.inference_hop_seconds,
            "training_window_seconds": config.training_window_seconds,
        },
        "warnings": warnings,
    }
