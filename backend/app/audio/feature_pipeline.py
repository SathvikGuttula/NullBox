"""
Per-connection audio pipeline: PCM bytes in, analysis dict out.

    VAD -> acoustic features
        -> streaming anti-spoof inference
        -> rolling speaker verification (when an identity is claimed)
        -> fused risk with reason codes

Three performance notes that matter for a live call:

* The detector and the speaker encoder are shared across connections.
  Constructing them per websocket would load hundreds of MB of weights every
  time a call starts.

* Pitch tracking (``librosa.pyin``) is roughly real-time on its own. Running it
  on every chunk leaves no headroom for the models, so it runs on a duty cycle
  and the last value is carried forward.

* Speaker embedding runs on its own, longer duty cycle. Identity does not
  change mid-call, and the estimate improves by accumulating speech rather than
  by being recomputed more often.
"""

from __future__ import annotations

import logging

import numpy as np

from app.audio.features import AcousticFeatureExtractor
from app.audio.vad import VoiceActivityDetector
from app.ml.config import DetectorConfig
from app.ml.context import CallContext, ContextAnalyzer
from app.ml.detector import VoiceSpoofDetector
from app.ml.fusion import RiskFusion
from app.ml.inference import StreamingInferenceEngine
from app.ml.policy import PolicyEngine
from app.ml.speaker import LiveSpeakerTracker, SpeakerEncoder

logger = logging.getLogger("voxshield.pipeline")


class AudioFeaturePipeline:
    """
    One instance per call.

    Parameters
    ----------
    claimed_identity
        Who the caller says they are. Without it the speaker branch is skipped
        and the response says so - a low risk score from anti-spoof alone must
        not be read as "this is who they claim to be".
    heavy_feature_interval
        Compute expensive prosody features once every N chunks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        config: DetectorConfig | None = None,
        claimed_identity: str | None = None,
        heavy_feature_interval: int = 3,
        call_context: CallContext | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.sample_rate = sample_rate
        self.claimed_identity = (claimed_identity or "").strip() or None

        # Context is fixed for the life of the call and cheap to score, so it
        # is assessed once here rather than on every chunk. Without it the
        # live path could only ever see two of the three branches, which is
        # what made a genuine-human impersonator invisible to it.
        self.call_context = call_context
        self.context = (
            ContextAnalyzer().assess(call_context) if call_context is not None else None
        )
        self.policy_engine = PolicyEngine()

        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

        self.inference = StreamingInferenceEngine(
            detector=VoiceSpoofDetector.get_shared(self.config),
            sample_rate=sample_rate,
            window_seconds=self.config.inference_window_seconds,
            hop_seconds=self.config.inference_hop_seconds,
            config=self.config,
        )

        self.fusion = RiskFusion()
        self.policy: dict | None = None
        self.speaker = self._build_speaker_tracker()

        self.heavy_feature_interval = max(int(heavy_feature_interval), 1)
        self.chunk_index = 0
        self.last_features: dict = {}

        self.total_audio_ms = 0.0
        self.speech_audio_ms = 0.0
        self.latest_risk: dict | None = None

    def _build_speaker_tracker(self) -> LiveSpeakerTracker | None:
        """
        Set up rolling verification, or return None with a logged reason.

        Never raises: a missing speechbrain install or an unenrolled identity
        must degrade the call to anti-spoof only, not drop it.
        """

        if not self.claimed_identity:
            return None

        try:
            from app.api.speakers import get_registry

            registry = get_registry()

            if self.claimed_identity not in registry:
                logger.info(
                    "no enrolled profile for %r - speaker branch disabled",
                    self.claimed_identity,
                )
                return None

            return LiveSpeakerTracker(
                registry=registry,
                encoder=SpeakerEncoder.get_shared(),
                identity=self.claimed_identity,
                sample_rate=self.sample_rate,
            )
        except ImportError as exc:
            logger.warning("speaker branch unavailable: %s", exc)
        except Exception as exc:
            logger.warning("could not start speaker tracking: %s", exc)

        return None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _decode_pcm16(audio_bytes: bytes) -> np.ndarray:
        """Signed 16-bit little-endian PCM -> float32 in [-1, 1]."""

        # An odd byte count means a chunk was split mid-sample; drop the
        # trailing byte rather than letting frombuffer raise and kill the call.
        if len(audio_bytes) % 2:
            audio_bytes = audio_bytes[:-1]

        if not audio_bytes:
            return np.zeros(0, dtype=np.float32)

        return np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0

    def _empty_result(self) -> dict:
        return {"vad": {}, "features": {}, "deepfake": {}, "speaker": None,
                "context": self.context.to_dict() if self.context else None,
                "risk": None, "policy": None, "stream": {}}

    # -- main entry point --------------------------------------------------

    def process(self, audio_bytes: bytes) -> dict:
        if not audio_bytes:
            return self._empty_result()

        audio = self._decode_pcm16(audio_bytes)

        if audio.size == 0:
            return self._empty_result()

        self.chunk_index += 1

        duration_ms = audio.size / self.sample_rate * 1000.0
        self.total_audio_ms += duration_ms

        vad_result = self.vad.process(audio)
        is_speech = bool(vad_result["speech"])

        if is_speech:
            self.speech_audio_ms += duration_ms

        # Prosody is expensive; recompute on a duty cycle and reuse otherwise.
        if self.chunk_index % self.heavy_feature_interval == 1 or not self.last_features:
            self.last_features = self.extractor.extract(audio)
            self.last_features["stale"] = False
        else:
            self.last_features = {**self.last_features, "stale": True}

        deepfake_results = []
        speaker_updated = False

        if is_speech:
            deepfake_results = self.inference.add_audio(audio)

            if self.speaker is not None:
                speaker_updated = self.speaker.add_speech(audio) is not None

        latest = deepfake_results[-1] if deepfake_results else None
        speaker_state = self.speaker.to_dict() if self.speaker else None

        # Re-fuse whenever either branch produced something new.
        if latest is not None or speaker_updated:
            fused = self.fusion.fuse(
                spoof_probability=(
                    latest["smoothed_probability"]
                    if latest and latest["model_status"] == "neural"
                    else None
                ),
                speaker_similarity=(
                    speaker_state["similarity"] if speaker_state else None
                ),
                speaker_decision=(
                    speaker_state["decision"] if speaker_state else None
                ),
                context_score=self.context.risk if self.context else None,
            )

            self.latest_risk = fused.to_dict()

            # A score with no action attached moves the decision onto whoever
            # is reading it, under time pressure, while someone talks at them
            # urgently - which is the condition social engineering exploits.
            self.policy = self.policy_engine.decide(
                risk_decision=fused.decision,
                risk_score=fused.risk_score,
                transaction_amount=(
                    self.call_context.transaction_amount if self.call_context else None
                ),
                identity_decision=(
                    speaker_state["decision"] if speaker_state else None
                ),
                calibrated_inputs=self.latest_risk["calibrated"],
            ).to_dict()

        speech_ratio = self.speech_audio_ms / max(self.total_audio_ms, 1e-8)

        return {
            "vad": vad_result,
            "features": self.last_features,
            "deepfake": {
                "available": latest is not None,
                "result": latest,
                "windows_scored": self.inference.windows_scored,
                "buffered_seconds": round(self.inference.buffered_seconds, 2),
            },
            "speaker": speaker_state,
            "context": self.context.to_dict() if self.context else None,
            "risk": self.latest_risk,
            "policy": self.policy,
            "stream": {
                "total_audio_ms": round(self.total_audio_ms, 2),
                "speech_audio_ms": round(self.speech_audio_ms, 2),
                "speech_ratio": round(speech_ratio, 4),
                "chunks": self.chunk_index,
                "claimed_identity": self.claimed_identity,
                "speaker_branch": self.speaker is not None,
            },
        }

    def reset(self) -> None:
        self.inference.reset()
        self.speaker = self._build_speaker_tracker()
        self.chunk_index = 0
        self.last_features = {}
        self.total_audio_ms = 0.0
        self.speech_audio_ms = 0.0
        self.latest_risk = None
        self.policy = None
