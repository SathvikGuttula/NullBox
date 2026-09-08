"""
Channel augmentation for the anti-spoof detector.

Why this matters more than it looks
-----------------------------------
ASVspoof is studio-clean. A model trained on it alone learns "clean + slightly
odd spectrum = spoof" and then collapses the first time a judge holds a phone
to a laptop microphone, because *every* sample it now sees is band-limited and
compressed. The classifier has no way to tell "degraded by a phone line" apart
from "degraded by a vocoder".

The fix is to make channel degradation uninformative: apply it to bonafide and
spoof alike, at training time, so the only thing left that separates the
classes is the synthesis artifact itself.

Every transform here is pure torch and runs on the CPU inside DataLoader
workers in well under a millisecond per clip, so augmentation never becomes
the bottleneck. No ffmpeg, no sox, no external binaries - which also means it
works identically on Windows.
"""

from __future__ import annotations

import math
import random

import torch
import torchaudio.functional as AF


def _rms(audio: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(audio.pow(2)).clamp(min=1e-12))


class AudioAugmenter:
    """
    Randomised channel simulation.

    Each effect fires independently with its own probability, so a clip may
    pick up several at once - which is realistic, since a real call is
    band-limited *and* compressed *and* noisy.

    Parameters are probabilities in [0, 1]. Set any to 0 to disable that
    effect; construct with ``AudioAugmenter.disabled()`` to turn everything off
    without changing call sites.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_probability: float = 0.30,
        gain_probability: float = 0.30,
        telephone_probability: float = 0.35,
        codec_probability: float = 0.20,
        reverb_probability: float = 0.15,
        clipping_probability: float = 0.05,
        dropout_probability: float = 0.10,
        snr_range: tuple[float, float] = (10.0, 30.0),
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.noise_probability = noise_probability
        self.gain_probability = gain_probability
        self.telephone_probability = telephone_probability
        self.codec_probability = codec_probability
        self.reverb_probability = reverb_probability
        self.clipping_probability = clipping_probability
        self.dropout_probability = dropout_probability
        self.snr_range = snr_range

        self.rng = random.Random(seed) if seed is not None else random

    @classmethod
    def disabled(cls, sample_rate: int = 16000) -> "AudioAugmenter":
        return cls(
            sample_rate=sample_rate,
            noise_probability=0.0,
            gain_probability=0.0,
            telephone_probability=0.0,
            codec_probability=0.0,
            reverb_probability=0.0,
            clipping_probability=0.0,
            dropout_probability=0.0,
        )

    # -- individual effects ------------------------------------------------

    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Additive white noise at a randomly chosen SNR."""

        snr_db = self.rng.uniform(*self.snr_range)

        signal_rms = _rms(audio)
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))

        return audio + torch.randn_like(audio) * noise_rms

    def random_gain(self, audio: torch.Tensor) -> torch.Tensor:
        return audio * self.rng.uniform(0.5, 1.4)

    def telephone(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Narrowband telephony: 300-3400 Hz passband, then an 8 kHz round trip.

        The resample down and back is what actually destroys the top octave -
        exactly the band where vocoder artifacts are easiest to spot - so this
        is the single most important augmentation for a phone-call demo.
        """

        audio = AF.highpass_biquad(audio, self.sample_rate, cutoff_freq=300.0)
        audio = AF.lowpass_biquad(audio, self.sample_rate, cutoff_freq=3400.0)

        narrow = AF.resample(audio, self.sample_rate, 8000)
        return AF.resample(narrow, 8000, self.sample_rate)

    def codec(self, audio: torch.Tensor) -> torch.Tensor:
        """
        mu-law companding, i.e. G.711 quantisation.

        This is the quantisation noise floor every PSTN and most VoIP calls
        carry. torchaudio's mu-law codec expects the signal inside [-1, 1], so
        clamp first or the encoder wraps and produces garbage.
        """

        peak = audio.abs().max().clamp(min=1e-8)
        scaled = (audio / peak).clamp(-1.0, 1.0)

        encoded = AF.mu_law_encoding(scaled, quantization_channels=256)
        decoded = AF.mu_law_decoding(encoded, quantization_channels=256)

        return decoded * peak

    def reverb(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convolution with a synthetic exponentially decaying impulse response.

        A real RIR corpus would be better, but this captures the part that
        matters - smearing across time - with no download and no extra
        dependency.
        """

        rt60 = self.rng.uniform(0.1, 0.5)
        length = int(self.sample_rate * rt60)

        if length < 8:
            return audio

        time = torch.arange(length, dtype=audio.dtype, device=audio.device)
        decay = torch.exp(-6.9 * time / length)          # -60 dB over the tail
        impulse = torch.randn(length, dtype=audio.dtype, device=audio.device) * decay
        impulse[0] = 1.0
        impulse = impulse / impulse.abs().max().clamp(min=1e-8)

        wet = AF.fftconvolve(audio, impulse, mode="full")[: audio.shape[-1]]

        mix = self.rng.uniform(0.15, 0.45)
        return (1.0 - mix) * audio + mix * wet

    def clip(self, audio: torch.Tensor) -> torch.Tensor:
        """Drive the signal into a hard limiter, as a hot microphone would."""

        threshold = self.rng.uniform(0.3, 0.8) * audio.abs().max().clamp(min=1e-8)
        return torch.clamp(audio, -threshold, threshold)

    def packet_dropout(self, audio: torch.Tensor) -> torch.Tensor:
        """Zero a few short spans, the way VoIP packet loss does."""

        audio = audio.clone()
        length = audio.shape[-1]

        for _ in range(self.rng.randint(1, 3)):
            span = int(self.sample_rate * self.rng.uniform(0.01, 0.04))
            if span >= length:
                continue
            start = self.rng.randint(0, length - span)
            audio[..., start : start + span] = 0.0

        return audio

    # -- pipeline ----------------------------------------------------------

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.clone().float()

        if self.rng.random() < self.reverb_probability:
            audio = self.reverb(audio)

        if self.rng.random() < self.telephone_probability:
            audio = self.telephone(audio)

        if self.rng.random() < self.codec_probability:
            audio = self.codec(audio)

        if self.rng.random() < self.noise_probability:
            audio = self.add_noise(audio)

        if self.rng.random() < self.clipping_probability:
            audio = self.clip(audio)

        if self.rng.random() < self.dropout_probability:
            audio = self.packet_dropout(audio)

        if self.rng.random() < self.gain_probability:
            audio = self.random_gain(audio)

        # Renormalise so augmentation never changes the overall level, which
        # would otherwise become a shortcut feature of its own.
        peak = audio.abs().max()
        if peak > 1e-8:
            audio = audio / peak

        return torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    def describe(self) -> dict:
        return {
            "reverb": self.reverb_probability,
            "telephone": self.telephone_probability,
            "codec_mulaw": self.codec_probability,
            "noise": self.noise_probability,
            "clipping": self.clipping_probability,
            "packet_dropout": self.dropout_probability,
            "gain": self.gain_probability,
            "snr_range_db": self.snr_range,
        }


def build_augmenter(config, sample_rate: int = 16000) -> AudioAugmenter | None:
    """Construct from a :class:`TrainingConfig`, or None when disabled."""

    if not getattr(config, "augment", False):
        return None

    return AudioAugmenter(
        sample_rate=sample_rate,
        noise_probability=config.noise_probability,
        gain_probability=config.gain_probability,
        telephone_probability=config.telephone_probability,
        codec_probability=config.codec_probability,
        reverb_probability=config.reverb_probability,
    )
