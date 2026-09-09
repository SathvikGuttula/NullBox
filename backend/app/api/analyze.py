"""
The endpoint the whole project builds up to.

    POST /api/v1/analyze     audio + optional claimed identity
                             -> spoof score, identity score, fused risk, reasons

This is "VoxShield ML Core v1" from §85 of the handoff: give it a wav and a
claimed identity, get back a decision with its reasoning attached.

Both branches are optional. With no identity claimed it degrades to anti-spoof
alone and says so in ``missing_signals`` — which matters, because a caller who
does not know the identity branch was skipped might read a low risk score as
"this person is who they say they are", and it means nothing of the sort.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.speakers import decode_upload, get_registry
from app.ml.config import DetectorConfig
from app.ml.detector import VoiceSpoofDetector
from app.ml.fusion import RiskFusion
from app.ml.speaker import SpeakerEncoder

logger = logging.getLogger("voxshield.analyze")

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

_fusion = RiskFusion()


@router.post("/analyze")
async def analyze(
    sample: UploadFile = File(...),
    claimed_identity: str | None = Form(None),
) -> dict:
    """
    Score one recording on every branch that is available.

    The heavy work runs in a threadpool: a transformer forward pass is tens of
    milliseconds of blocking CPU/GPU work, and holding the event loop for that
    would stall every other concurrent request.
    """

    audio = decode_upload(await sample.read(), sample.filename or "sample")
    config = DetectorConfig()

    # -- anti-spoof --------------------------------------------------------

    spoof_probability = None
    spoof_status = "unavailable"

    try:
        detector = VoiceSpoofDetector.get_shared(config)
        spoof = await run_in_threadpool(detector.predict, audio, config.sample_rate)
        spoof_status = spoof["model_status"]

        # An untrained detector returns 0.5 as a placeholder. Feeding that into
        # the fusion would manufacture a 50% risk contribution out of nothing.
        if spoof_status == "neural":
            spoof_probability = spoof["synthetic_probability"]
    except Exception as exc:
        logger.warning("anti-spoof branch unavailable: %s", exc)

    # -- speaker verification ----------------------------------------------

    identity = (claimed_identity or "").strip()
    verification = None

    if identity:
        try:
            encoder = SpeakerEncoder.get_shared()
            embedding = await run_in_threadpool(encoder.embed, audio)
            verification = get_registry().verify(identity, embedding).to_dict()
        except ImportError as exc:
            logger.warning("speaker branch unavailable: %s", exc)
        except Exception as exc:
            logger.warning("speaker verification failed: %s", exc)

    # -- fuse ---------------------------------------------------------------

    risk = _fusion.fuse(
        spoof_probability=spoof_probability,
        speaker_similarity=verification["similarity"] if verification else None,
        speaker_decision=verification["decision"] if verification else None,
    )

    notes = []
    if spoof_probability is None:
        notes.append(
            f"Anti-spoof branch did not contribute (model status: {spoof_status}). "
            "Train a checkpoint with scripts/train_model.py."
        )
    if not identity:
        notes.append(
            "No identity was claimed, so this is an anti-spoof result only. "
            "It says nothing about whether the speaker is who they claim to be."
        )
    elif verification is None:
        notes.append(
            "Speaker verification was requested but unavailable; install "
            "speechbrain and enroll the identity first."
        )

    return {
        "duration_seconds": round(len(audio) / config.sample_rate, 2),
        "anti_spoof": {
            "synthetic_probability": spoof_probability,
            "model_status": spoof_status,
        },
        "speaker": verification,
        "risk": risk.to_dict(),
        "notes": notes,
    }
