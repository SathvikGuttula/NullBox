"""
Per-connection audio pipeline: PCM bytes in, analysis dict out.

VAD -> acoustic features -> streaming anti-spoof inference -> risk update.

Two performance notes that matter for a live call:

* The detector is shared across connections. Constructing one per websocket
  would download and instantiate a ~360 MB speech encoder every time a call
  starts.

* Pitch tracking (``librosa.pyin``) is roughly real-time on its own - running
  it on every incoming chunk leaves no headroom for the model. It runs on a
  duty cycle instead, and the last value is carried forward between updates.
"""

from __future__ import annotations

import numpy as np

from app.audio.features import AcousticFeatureExtractor
from app.audio.vad import VoiceActivityDetector
from app.ml.config import DetectorConfig
from app.ml.detector import VoiceSpoofDetector
from app.ml.inference import StreamingInferenceEngine


class AudioFeaturePipeline:
    """
    One instance per call.

    Parameters
    ----------
    heavy_feature_interval
        Compute the expensive prosody features once every N chunks. At the
        default 2 s chunk size, 3 means roughly every six seconds, which is
        ample for a slowly varying speaker profile.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        config: DetectorConfig | None = None,
        heavy_feature_interval: int = 3,
    ) -> None:
        self.config = config or DetectorConfig()
        self.sample_rate = sample_rate

        self.vad = VoiceActivityDetector(sample_rate=sample_rate)

        self.extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

        self.inference = StreamingInferenceEngine(
            detector=VoiceSpoofDetector.get_shared(self.config),
            sample_rate=sample_rate,
            window_seconds=self.config.inference_window_seconds,
            hop_seconds=self.config.inference_hop_seconds,
            config=self.config,
        )

        self.heavy_feature_interval = max(int(heavy_feature_interval), 1)
        self.chunk_index = 0
        self.last_features: dict = {}

        self.total_audio_ms = 0.0
        self.speech_audio_ms = 0.0

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
        return {"vad": {}, "features": {}, "deepfake": {}, "stream": {}}

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

        if vad_result["speech"]:
            self.speech_audio_ms += duration_ms

        # Prosody is expensive; recompute on a duty cycle and reuse otherwise.
        if self.chunk_index % self.heavy_feature_interval == 1 or not self.last_features:
            self.last_features = self.extractor.extract(audio)
            self.last_features["stale"] = False
        else:
            self.last_features = {**self.last_features, "stale": True}

        deepfake_results = []

        if vad_result["speech"]:
            deepfake_results = self.inference.add_audio(audio)

        speech_ratio = self.speech_audio_ms / max(self.total_audio_ms, 1e-8)

        latest = deepfake_results[-1] if deepfake_results else None

        return {
            "vad": vad_result,
            "features": self.last_features,
            "deepfake": {
                "available": latest is not None,
                "result": latest,
                "windows_scored": self.inference.windows_scored,
                "buffered_seconds": round(self.inference.buffered_seconds, 2),
            },
            "stream": {
                "total_audio_ms": round(self.total_audio_ms, 2),
                "speech_audio_ms": round(self.speech_audio_ms, 2),
                "speech_ratio": round(speech_ratio, 4),
                "chunks": self.chunk_index,
            },
        }

    def reset(self) -> None:
        self.inference.reset()
        self.chunk_index = 0
        self.last_features = {}
        self.total_audio_ms = 0.0
        self.speech_audio_ms = 0.0
