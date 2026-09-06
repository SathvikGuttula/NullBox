import numpy as np

from app.audio.features import (
    AcousticFeatureExtractor,
)

from app.audio.vad import (
    VoiceActivityDetector,
)

from app.ml.detector import (
    BaselineNeuralDetector,
)

from app.ml.inference import (
    StreamingInferenceEngine,
)


class AudioFeaturePipeline:

    def __init__(
        self,
        sample_rate: int = 16000,
    ):

        self.sample_rate = sample_rate

        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
        )

        self.extractor = (
            AcousticFeatureExtractor(
                sample_rate=sample_rate,
            )
        )

        detector = (
            BaselineNeuralDetector()
        )

        self.inference = (
            StreamingInferenceEngine(
                detector=detector,
                sample_rate=sample_rate,
                window_seconds=3.0,
                hop_seconds=1.0,
            )
        )

        self.total_audio_ms = 0.0

        self.speech_audio_ms = 0.0

    def process(
        self,
        audio_bytes: bytes,
    ) -> dict:

        if not audio_bytes:

            return {
                "vad": {},
                "features": {},
                "deepfake": {},
                "stream": {},
            }

        audio = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        ).astype(
            np.float32
        )

        audio /= 32768.0

        duration_ms = (
            len(audio)
            / self.sample_rate
            * 1000
        )

        self.total_audio_ms += (
            duration_ms
        )

        vad_result = (
            self.vad.process(
                audio
            )
        )

        if vad_result["speech"]:

            self.speech_audio_ms += (
                duration_ms
            )

        features = (
            self.extractor.extract(
                audio
            )
        )

        deepfake_results = []

        if vad_result["speech"]:

            deepfake_results = (
                self.inference.add_audio(
                    audio
                )
            )

        speech_ratio = (
            self.speech_audio_ms
            / max(
                self.total_audio_ms,
                1e-8,
            )
        )

        latest_deepfake = (
            deepfake_results[-1]
            if deepfake_results
            else None
        )

        return {

            "vad": vad_result,

            "features": features,

            "deepfake": {
                "available":
                    latest_deepfake
                    is not None,

                "result":
                    latest_deepfake,
            },

            "stream": {

                "total_audio_ms":
                    round(
                        self.total_audio_ms,
                        2,
                    ),

                "speech_audio_ms":
                    round(
                        self.speech_audio_ms,
                        2,
                    ),

                "speech_ratio":
                    round(
                        speech_ratio,
                        4,
                    ),
            },
        }