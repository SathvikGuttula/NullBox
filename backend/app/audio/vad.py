from collections import deque

import numpy as np


class VoiceActivityDetector:
    """
    Lightweight streaming voice activity detector.

    Energy-based, with a percentile noise-floor tracker and a hangover so a
    single quiet frame mid-word does not end the utterance.

    Why the noise floor is tracked from a window rather than seeded
    --------------------------------------------------------------
    The previous version set ``noise_floor = rms`` of the very first chunk. If
    the caller was already speaking when the stream opened - which is the
    normal case, since the connection opens when the call connects - the floor
    was seeded to *speech* level. The threshold then sat above every subsequent
    speech frame, and because adaptation only ran while the signal looked
    quiet, it could never come back down. The result was a VAD that reported
    silence for the entire call, and therefore an anti-spoof model that was
    never fed a single window.

    Tracking a low percentile of a rolling window fixes the ordinary case, but
    not the degenerate one: if *every* frame in the window is speech, the 20th
    percentile is still speech level. An energy VAD cannot separate speech from
    noise when it has never heard the noise.

    So the floor is additionally capped at ``maximum_noise_floor``. Real
    background noise on a microphone or phone line sits well below -30 dBFS;
    a measured "floor" above that means the window contains no silence, not
    that the room is that loud. Capping makes the mid-speech case behave.

    The cap chooses a failure direction deliberately. In genuinely loud
    surroundings the VAD will over-report speech, so the anti-spoof model gets
    fed windows it did not need to see - wasted compute, and the model's own
    score is what decides the outcome. Under-reporting is the dangerous
    direction: the pipeline silently analyses nothing and the call goes
    unmonitored while the dashboard looks healthy.

    This interface can later be backed by a neural VAD (Silero, WebRTC) without
    changing anything else in VoxShield, and should be for production.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        initial_noise_floor: float = 0.003,
        speech_multiplier: float = 2.5,
        window_frames: int = 50,
        floor_percentile: float = 20.0,
        hangover_frames: int = 3,
        minimum_threshold: float = 0.008,
        maximum_noise_floor: float = 0.02,
    ) -> None:
        self.sample_rate = sample_rate

        self.initial_noise_floor = initial_noise_floor
        self.noise_floor = initial_noise_floor
        self.speech_multiplier = speech_multiplier
        self.minimum_threshold = minimum_threshold
        self.maximum_noise_floor = maximum_noise_floor

        self.history: deque = deque(maxlen=window_frames)
        self.floor_percentile = floor_percentile

        self.hangover_frames = hangover_frames
        self.hangover_remaining = 0

        self.speech_frames = 0
        self.silence_frames = 0

    def reset(self) -> None:
        self.noise_floor = self.initial_noise_floor
        self.history.clear()
        self.hangover_remaining = 0
        self.speech_frames = 0
        self.silence_frames = 0

    def _rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def process(self, audio: np.ndarray) -> dict:
        if audio.size == 0:
            return {
                "speech": False,
                "confidence": 0.0,
                "rms": 0.0,
                "noise_floor": round(self.noise_floor, 6),
            }

        audio = audio.astype(np.float32, copy=False)

        rms = self._rms(audio)

        self.history.append(rms)

        # A low percentile of the recent window is the quiet part of the
        # signal, whether or not the stream opened on speech - then capped, so
        # a window containing no silence at all cannot lock the detector out.
        if len(self.history) >= 5:
            measured = float(
                np.percentile(np.asarray(self.history), self.floor_percentile)
            )
            self.noise_floor = min(
                max(measured, 1e-5),
                self.maximum_noise_floor,
            )

        threshold = max(
            self.noise_floor * self.speech_multiplier,
            self.minimum_threshold,
        )

        above = rms > threshold

        if above:
            self.hangover_remaining = self.hangover_frames
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            if self.hangover_remaining > 0:
                self.hangover_remaining -= 1

        speech = above or self.hangover_remaining > 0

        ratio = rms / max(threshold, 1e-8)
        confidence = min(max((ratio - 1.0) / 3.0, 0.0), 1.0)

        return {
            "speech": bool(speech),
            "confidence": round(confidence, 4),
            "rms": round(rms, 6),
            "noise_floor": round(self.noise_floor, 6),
            "threshold": round(threshold, 6),
            "hangover": self.hangover_remaining,
        }
