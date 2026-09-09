"""
Speaker enrollment and verification endpoints.

    POST /api/v1/speakers/enroll     identity + 3-5 audio files -> profile
    POST /api/v1/speakers/verify     identity + one audio file  -> decision
    GET  /api/v1/speakers            list enrolled identities
    DELETE /api/v1/speakers/{id}     remove a profile

Audio is accepted, embedded, and discarded. Only the embedding centroid is
persisted — see the handoff's privacy model. Embeddings remain biometric data
and the store should be encrypted at rest before this is exposed to real users;
that is one obvious place to change, in ``SpeakerRegistry``.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.ml.config import resolve_path
from app.ml.speaker import SpeakerEncoder, SpeakerRegistry, SpeakerThresholds

logger = logging.getLogger("voxshield.speakers")

router = APIRouter(prefix="/api/v1/speakers", tags=["Speakers"])


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MINIMUM_SECONDS = 1.0

_registry: SpeakerRegistry | None = None


def get_registry() -> SpeakerRegistry:
    """
    Process-wide registry, loading calibrated thresholds when present.

    Falls back to the uncalibrated placeholders rather than refusing to start,
    but every verification response carries ``calibrated`` so a caller can tell
    which it got.
    """

    global _registry

    if _registry is not None:
        return _registry

    thresholds = SpeakerThresholds.balanced()

    calibration = resolve_path("models/speaker_thresholds.json")
    if calibration.exists():
        try:
            data = json.loads(calibration.read_text(encoding="utf-8"))
            thresholds = SpeakerThresholds(**data["modes"]["balanced"])
            logger.info("loaded speaker thresholds %s", thresholds.version)
        except Exception as exc:
            logger.warning("could not read %s: %s", calibration, exc)

    _registry = SpeakerRegistry(
        thresholds=thresholds,
        store_path=resolve_path("models/speakers.json"),
    )
    return _registry


def reset_registry() -> None:
    """Test hook."""

    global _registry
    _registry = None


def decode_upload(data: bytes, filename: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Bytes from an upload -> mono float32 at the target rate.

    Everything is validated here rather than deeper in: this is the boundary
    where untrusted input arrives.
    """

    if not data:
        raise HTTPException(400, f"{filename} is empty")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"{filename} is {len(data) / 1024 ** 2:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1024 ** 2} MB",
        )

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(
            400,
            f"could not decode {filename} as audio ({exc}). WAV, FLAC and OGG "
            f"are supported; MP3 and M4A are not.",
        ) from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size / rate < MINIMUM_SECONDS:
        raise HTTPException(
            400,
            f"{filename} is {audio.size / rate:.2f}s; at least "
            f"{MINIMUM_SECONDS}s of speech is needed for a usable embedding",
        )

    if rate != sample_rate:
        import torch
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(audio), rate, sample_rate).numpy()

    return np.ascontiguousarray(audio, dtype=np.float32)


@router.post("/enroll")
async def enroll_speaker(
    identity: str = Form(...),
    samples: list[UploadFile] = File(...),
    allow_inconsistent: bool = Form(False),
) -> dict:
    """
    Create a voice profile from several recordings of one person.

    Three samples minimum, five recommended. The registry refuses a set whose
    samples do not agree with each other, because a blurred prototype is not a
    one-off error - it is an account that quietly matches several people for as
    long as it exists.
    """

    identity = identity.strip()
    if not identity:
        raise HTTPException(400, "identity must not be empty")

    registry = get_registry()

    try:
        encoder = SpeakerEncoder.get_shared()
    except ImportError as exc:
        raise HTTPException(503, str(exc)) from exc

    embeddings = []
    for upload in samples:
        audio = decode_upload(await upload.read(), upload.filename or "sample")
        embeddings.append(encoder.embed(audio))

    try:
        profile = registry.enroll(
            identity,
            embeddings,
            encoder=encoder.model_name,
            allow_inconsistent=allow_inconsistent,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "identity": profile.identity,
        "samples": profile.sample_count,
        "consistency": round(profile.consistency, 4),
        "encoder": profile.encoder,
        "created_at": profile.created_at,
        "threshold_version": profile.threshold_version,
        "note": "Audio was discarded; only the embedding centroid is stored.",
    }


@router.post("/verify")
async def verify_speaker(
    identity: str = Form(...),
    sample: UploadFile = File(...),
) -> dict:
    """Score one recording against a claimed identity."""

    registry = get_registry()

    try:
        encoder = SpeakerEncoder.get_shared()
    except ImportError as exc:
        raise HTTPException(503, str(exc)) from exc

    audio = decode_upload(await sample.read(), sample.filename or "sample")
    embedding = encoder.embed(audio)

    return registry.verify(identity.strip(), embedding).to_dict()


@router.get("")
async def list_speakers() -> dict:
    registry = get_registry()

    return {
        "count": len(registry),
        "thresholds": {
            "match": registry.thresholds.match,
            "no_match": registry.thresholds.no_match,
            "version": registry.thresholds.version,
            "calibrated": registry.thresholds.calibrated,
        },
        "speakers": [
            {
                "identity": p.identity,
                "samples": p.sample_count,
                "consistency": round(p.consistency, 4),
                "created_at": p.created_at,
            }
            for p in registry.profiles.values()
        ],
    }


@router.delete("/{identity}")
async def delete_speaker(identity: str) -> dict:
    registry = get_registry()

    if identity not in registry:
        raise HTTPException(404, f"no enrolled profile for '{identity}'")

    del registry.profiles[identity]

    if registry.store_path:
        registry.save()

    return {"identity": identity, "deleted": True}
