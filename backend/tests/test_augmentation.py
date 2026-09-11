import numpy as np
import pytest
import torch

from app.ml.augmentation import AudioAugmenter


@pytest.fixture
def speech_like():
    """
    A harmonic-rich signal with real energy all the way to Nyquist.

    50 harmonics of 150 Hz reaches 7.5 kHz. The band above 4.5 kHz has to
    contain actual signal, otherwise the telephone-filter test below measures
    numerical leakage against numerical leakage and proves nothing.
    """

    t = torch.linspace(0, 2.0, 32000)
    signal = sum(
        (1.0 / (k + 1)) * torch.sin(2 * torch.pi * 150 * (k + 1) * t)
        for k in range(50)
    )
    return (signal / signal.abs().max()).float()


def test_output_shape_and_range_are_preserved(speech_like):
    augmenter = AudioAugmenter(seed=0)

    for _ in range(30):
        out = augmenter(speech_like)
        assert out.shape == speech_like.shape
        assert out.abs().max() <= 1.0 + 1e-5
        assert torch.isfinite(out).all()


def test_disabled_augmenter_only_normalises(speech_like):
    out = AudioAugmenter.disabled()(speech_like)
    assert torch.allclose(out, speech_like, atol=1e-5)


def test_telephone_filter_removes_high_frequencies(speech_like):
    """
    The narrowband round trip must actually destroy the top octave - that is
    the whole point of the augmentation.
    """

    augmenter = AudioAugmenter(seed=1)
    filtered = augmenter.telephone(speech_like)

    def band_energy(signal, low_bin, high_bin):
        spectrum = torch.fft.rfft(signal).abs()
        return float(spectrum[low_bin:high_bin].pow(2).sum())

    # 32000 samples at 16 kHz -> 0.5 Hz per bin.
    high_before = band_energy(speech_like, 9000, 16000)   # 4.5-8 kHz
    high_after = band_energy(filtered, 9000, 16000)

    # Measured attenuation is around -36 dB; 1% leaves ample headroom.
    assert high_after < high_before * 0.01

    # The passband must survive - an augmentation that destroys the speech
    # along with the high band would just be teaching the model to score noise.
    band_before = band_energy(speech_like, 600, 6800)      # 300-3400 Hz
    band_after = band_energy(filtered, 600, 6800)

    assert band_after > band_before * 0.25


def test_mu_law_codec_is_lossy_but_close(speech_like):
    augmenter = AudioAugmenter(seed=2)
    coded = augmenter.codec(speech_like)

    assert not torch.allclose(coded, speech_like)
    assert (coded - speech_like).abs().mean() < 0.05


def test_noise_respects_the_requested_snr():
    augmenter = AudioAugmenter(seed=3, snr_range=(20.0, 20.0))

    signal = torch.ones(16000) * 0.5
    noisy = augmenter.add_noise(signal)

    noise = noisy - signal
    measured_snr = 20 * np.log10(
        float(signal.pow(2).mean().sqrt()) / float(noise.pow(2).mean().sqrt())
    )

    assert measured_snr == pytest.approx(20.0, abs=1.5)


def test_packet_dropout_zeroes_some_samples(speech_like):
    augmenter = AudioAugmenter(seed=4)
    dropped = augmenter.packet_dropout(speech_like)

    assert (dropped == 0.0).sum() > 0
    assert (dropped == 0.0).float().mean() < 0.2


def test_reverb_keeps_the_length(speech_like):
    augmenter = AudioAugmenter(seed=5)
    assert augmenter.reverb(speech_like).shape == speech_like.shape


def test_augmenter_does_not_mutate_its_input(speech_like):
    original = speech_like.clone()
    AudioAugmenter(seed=6)(speech_like)
    assert torch.equal(speech_like, original)


def test_silence_does_not_produce_nan():
    augmenter = AudioAugmenter(seed=7)
    out = augmenter(torch.zeros(16000))
    assert torch.isfinite(out).all()
