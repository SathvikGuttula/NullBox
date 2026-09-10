"""
Is there actually speech here?

Why this exists
---------------
The anti-spoof model was trained on one thing: speech, either bonafide or
synthesised. It has never seen silence, hiss, hold music or a dial tone, and a
classifier asked about a class it was never trained on does not answer "I don't
know" - it answers confidently and wrongly.

Measured on the shipped checkpoint, with calibration applied:

    four seconds of digital silence   ->  0.999 synthetic
    four seconds of white noise       ->  0.997 synthetic

Both are nonsense, and both are exactly what a tester tries in the first two
minutes. Worse, they are what a real call produces during hold music or while
nobody is talking. Feeding them to the model and reporting the answer is how a
demo produces a confident lie.

What separates speech from the things that fool it
--------------------------------------------------
Not energy. White noise at -20 dBFS has plenty of energy, and an energy-based
VAD (``app.audio.vad``) passes it straight through.

Speech is *intermittent*. People stop between words, and they stop between
sentences, so the quiet parts of a speech signal are far quieter than the loud
parts. Steady signals - noise, tones, hum, a DC offset - have no such gap.

Measured across a 4-second window, the ratio of the 95th to the 10th percentile
of frame energy. Every number below was measured, not assumed:

    NON-SPEECH                          SPEECH (measured on TTS)
    silence                 0.00        clean            > 170,000,000
    dc offset               1.00        + noise at 20 dB          19-21
    square wave             1.00        + noise at 15 dB          10-12
    pure tone               1.01        + noise at 10 dB         6.2-6.9
    hum, 50 Hz + harmonics  1.07        + noise at  5 dB         3.6-4.0
    white noise             1.09
    filtered noise          1.37
    amplitude-modulated     1.90
    noise under a slow fade 2.71

The threshold is 3.0: a factor of 2.2 above the loudest naturally occurring
non-speech, and still below speech buried in noise at 5 dB SNR - which is
barely intelligible to a person. An earlier value of 5.0 was too tight and
rejected speech at 10 dB SNR in some envelopes.

There is genuine overlap between the two columns, and no single threshold
removes it. Deliberately modulating white noise at 4 Hz with 90% depth - a
synthetic syllable pattern - reaches 7.4 and passes. That is a constructed
adversarial input rather than a thing a phone line produces.

What this is not
----------------
Not a speech/music classifier: music, ring tones and anything else with real
on/off structure will pass and be scored. It is a guard against the degenerate
*steady* inputs that make the detector confidently wrong, not a content filter.

How the result is used
----------------------
A failed check does **not** mean "bonafide". It means the anti-spoof branch has
nothing to say, so the branch is reported as unavailable and dropped from the
fusion, the same way a missing speaker profile is. Scoring it as zero risk
would be the "unknown is innocent" mistake the rest of this codebase is careful
to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frame size for the energy envelope. 32 ms at 16 kHz - long enough to be
# stable, short enough to land inside the gaps between words.
FRAME_SAMPLES = 512
HOP_SAMPLES = 160

# Below this RMS the signal is silence for any practical purpose. -60 dBFS.
SILENCE_RMS = 1e-3

# See the measured table in the module docstring. 2.2x above the loudest
# naturally occurring non-speech, and below speech at 5 dB SNR.
MINIMUM_DYNAMIC_RANGE = 3.0

# Shorter than this and the percentile estimates are not meaningful - there
# may simply not be a gap between words inside the window.
MINIMUM_SECONDS = 0.5


@dataclass
class SpeechPresence:
    """Whether a clip carries speech, and the numbers behind the verdict."""

    has_speech: bool
    reason: str
    rms: float
    dynamic_range: float
    seconds: float

    def to_dict(self) -> dict:
        return {
            "has_speech": self.has_speech,
            "reason": self.reason,
            "rms": round(self.rms, 6),
            "dynamic_range": round(self.dynamic_range, 2),
            "seconds": round(self.seconds, 2),
        }


def frame_energies(audio: np.ndarray) -> np.ndarray:
    """RMS of each overlapping frame."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size < FRAME_SAMPLES:
        return np.array([float(np.sqrt(np.mean(np.square(audio))))]) if audio.size else np.zeros(0)

    count = 1 + (audio.size - FRAME_SAMPLES) // HOP_SAMPLES

    # A strided view rather than a copy: this runs on every window of every
    # live call, and materialising ~400 overlapping frames per second is
    # avoidable work.
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, FRAME_SAMPLES),
        strides=(audio.strides[0] * HOP_SAMPLES, audio.strides[0]),
        writeable=False,
    )

    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def assess_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    minimum_dynamic_range: float = MINIMUM_DYNAMIC_RANGE,
) -> SpeechPresence:
    """
    Decide whether ``audio`` is speech-like enough to score.

    Returns the verdict with its measurements attached, so a caller can show
    an operator *why* a clip was not assessed rather than silently dropping a
    branch.
    """

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    seconds = audio.size / float(sample_rate) if sample_rate else 0.0

    if audio.size == 0:
        return SpeechPresence(False, "the clip is empty", 0.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    if rms < SILENCE_RMS:
        return SpeechPresence(
            False,
            f"the clip is silent (level {rms:.2e}, below {SILENCE_RMS:.0e})",
            rms,
            0.0,
            seconds,
        )

    if seconds < MINIMUM_SECONDS:
        # Too short to judge. Let it through rather than blocking the branch:
        # a caller that hands over 200 ms has other problems, and the length
        # limits on the API endpoints already reject anything this short.
        return SpeechPresence(
            True,
            f"clip is only {seconds:.2f}s - too short to check for speech structure",
            rms,
            0.0,
            seconds,
        )

    energies = frame_energies(audio)

    if energies.size < 4:
        return SpeechPresence(True, "too few frames to check", rms, 0.0, seconds)

    loud = float(np.percentile(energies, 95))
    quiet = float(np.percentile(energies, 10))

    # Floor the denominator so true digital silence between words gives a
    # large finite ratio instead of an infinity.
    dynamic_range = loud / max(quiet, 1e-9)

    if dynamic_range < minimum_dynamic_range:
        return SpeechPresence(
            False,
            (
                f"the signal is steady, not speech (dynamic range "
                f"{dynamic_range:.1f}, speech is typically above 10). "
                f"Tones, hum, hold music and noise look like this."
            ),
            rms,
            dynamic_range,
            seconds,
        )

    return SpeechPresence(True, "speech detected", rms, dynamic_range, seconds)
