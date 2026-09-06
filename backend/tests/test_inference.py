import numpy as np

from app.ml.detector import (
    BaselineNeuralDetector,
)

from app.ml.inference import (
    StreamingInferenceEngine,
)


def test_baseline_detector():

    detector = (
        BaselineNeuralDetector()
    )

    audio = np.zeros(
        16000,
        dtype=np.float32,
    )

    result = detector.predict(
        audio,
        16000,
    )

    assert (
        "synthetic_probability"
        in result
    )

    assert (
        0.0
        <= result[
            "synthetic_probability"
        ]
        <= 1.0
    )


def test_streaming_window():

    detector = (
        BaselineNeuralDetector()
    )

    engine = (
        StreamingInferenceEngine(
            detector=detector,
            sample_rate=16000,
            window_seconds=3.0,
            hop_seconds=1.0,
        )
    )

    audio = np.zeros(
        16000,
        dtype=np.float32,
    )

    results = engine.add_audio(
        audio
    )

    assert len(results) == 0

    results = engine.add_audio(
        np.zeros(
            16000,
            dtype=np.float32,
        )
    )

    assert len(results) == 0

    results = engine.add_audio(
        np.zeros(
            16000,
            dtype=np.float32,
        )
    )

    assert len(results) >= 1