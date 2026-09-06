import numpy as np
import librosa


class AcousticFeatureExtractor:
    """
    Extracts acoustic and spectral features from
    normalized mono audio.

    Output is intentionally represented as a
    serializable dictionary so it can later be
    stored, streamed, or passed into ML models.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_lfcc: int = 20,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc

    def _safe_mean(self, values) -> float:

        values = np.asarray(values)

        if values.size == 0:
            return 0.0

        return float(
            np.nan_to_num(values).mean()
        )

    def _safe_std(self, values) -> float:

        values = np.asarray(values)

        if values.size == 0:
            return 0.0

        return float(
            np.nan_to_num(values).std()
        )

    def extract(
        self,
        audio: np.ndarray,
    ) -> dict:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:

            return {
                "duration_ms": 0.0,
                "energy": 0.0,
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_rolloff": 0.0,
                "spectral_flatness": 0.0,
                "zero_crossing_rate": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "mfcc_mean": [],
                "mfcc_std": [],
            }

        duration_ms = (
            len(audio)
            / self.sample_rate
            * 1000
        )

        # -------------------------------------------------
        # Time-domain energy
        # -------------------------------------------------

        rms = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=256,
        )[0]

        energy = self._safe_mean(rms)

        # -------------------------------------------------
        # Spectral representation
        # -------------------------------------------------

        stft = librosa.stft(
            audio,
            n_fft=1024,
            hop_length=256,
        )

        magnitude = np.abs(stft)

        # -------------------------------------------------
        # Spectral centroid
        # -------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            S=magnitude,
            sr=self.sample_rate,
        )[0]

        # -------------------------------------------------
        # Spectral bandwidth
        # -------------------------------------------------

        bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude,
            sr=self.sample_rate,
        )[0]

        # -------------------------------------------------
        # Spectral rolloff
        # -------------------------------------------------

        rolloff = librosa.feature.spectral_rolloff(
            S=magnitude,
            sr=self.sample_rate,
            roll_percent=0.85,
        )[0]

        # -------------------------------------------------
        # Spectral flatness
        # -------------------------------------------------

        flatness = librosa.feature.spectral_flatness(
            S=magnitude,
        )[0]

        # -------------------------------------------------
        # Zero crossing
        # -------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=1024,
            hop_length=256,
        )[0]

        # -------------------------------------------------
        # MFCC
        # -------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=1024,
            hop_length=256,
        )

        mfcc_mean = [
            float(
                np.nan_to_num(
                    coefficient
                ).mean()
            )
            for coefficient in mfcc
        ]

        mfcc_std = [
            float(
                np.nan_to_num(
                    coefficient
                ).std()
            )
            for coefficient in mfcc
        ]

        # -------------------------------------------------
        # Pitch / fundamental frequency
        # -------------------------------------------------

        try:

            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=65,
                fmax=500,
                sr=self.sample_rate,
                frame_length=1024,
            )

            valid_pitch = f0[
                ~np.isnan(f0)
            ]

            pitch_mean = (
                float(valid_pitch.mean())
                if valid_pitch.size
                else 0.0
            )

            pitch_std = (
                float(valid_pitch.std())
                if valid_pitch.size
                else 0.0
            )

            voiced_ratio = (
                float(valid_pitch.size / len(f0))
                if len(f0)
                else 0.0
            )

        except Exception:

            pitch_mean = 0.0
            pitch_std = 0.0
            voiced_ratio = 0.0

        return {
            "duration_ms": round(
                duration_ms,
                2,
            ),

            "energy": round(
                energy,
                6,
            ),

            "spectral_centroid": round(
                self._safe_mean(centroid),
                3,
            ),

            "spectral_bandwidth": round(
                self._safe_mean(bandwidth),
                3,
            ),

            "spectral_rolloff": round(
                self._safe_mean(rolloff),
                3,
            ),

            "spectral_flatness": round(
                self._safe_mean(flatness),
                6,
            ),

            "zero_crossing_rate": round(
                self._safe_mean(zcr),
                6,
            ),

            "pitch_mean_hz": round(
                pitch_mean,
                3,
            ),

            "pitch_std_hz": round(
                pitch_std,
                3,
            ),

            "voiced_ratio": round(
                voiced_ratio,
                4,
            ),

            "mfcc_mean": [
                round(x, 4)
                for x in mfcc_mean
            ],

            "mfcc_std": [
                round(x, 4)
                for x in mfcc_std
            ],
        }