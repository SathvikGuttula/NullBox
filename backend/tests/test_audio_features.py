import numpy as np

from app.audio.features import (
    AcousticFeatureExtractor,
)


def test_feature_extraction():

    sample_rate = 16000

    duration = 1.0

    t = np.linspace(
        0,
        duration,
        int(
            sample_rate *
            duration
        ),
        endpoint=False,
    )

    audio = (
        0.2 *
        np.sin(
            2 *
            np.pi *
            220 *
            t
        )
    ).astype(
        np.float32
    )

    extractor = (
        AcousticFeatureExtractor(
            sample_rate=sample_rate
        )
    )

    features = extractor.extract(
        audio
    )

    assert features[
        "duration_ms"
    ] > 900

    assert features[
        "pitch_mean_hz"
    ] > 0

    assert len(
        features[
            "mfcc_mean"
        ]
    ) == 13

    assert len(
        features[
            "mfcc_std"
        ]
    ) == 13


def test_empty_audio():

    extractor = (
        AcousticFeatureExtractor()
    )

    features = extractor.extract(
        np.array(
            [],
            dtype=np.float32,
        )
    )

    assert features[
        "duration_ms"
    ] == 0.0