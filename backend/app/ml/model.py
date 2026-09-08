"""
The VoxShield anti-spoof classifier.

    waveform  (B, 64000)  16 kHz mono, 4 s
        |
        +-- SSL branch ------ wav2vec2 / WavLM ---> (B, T, H)
        |                     attentive stat pooling -> (B, 2H)
        |
        +-- Spectral branch - log-Mel -> CNN ------> (B, spectral_dim)
        |
        v
      concat -> MLP -> 2 logits  [bonafide, spoof]

Two things here are deliberate and are worth not undoing:

1. The spectral branch is computed *inside* ``forward`` on the same device as
   the waveform. The previous handoff flagged that the spectral tensor was
   never actually produced, so the "multi-branch" model was in truth a bare
   wav2vec2 classifier. Building the log-Mel on the GPU also keeps the
   DataLoader free of feature extraction, which matters on a laptop CPU.

2. Pooling is attentive statistics pooling, not a plain mean. Vocoder
   artifacts are bursty - they live in a few frames, not spread evenly across
   the clip - and a mean dilutes exactly the evidence the model needs. Adding
   the weighted standard deviation costs nothing and captures frame-to-frame
   inconsistency, which is itself a synthesis giveaway.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel


@dataclass
class AntiSpoofOutput:
    """Per-utterance prediction."""

    spoof_probability: float
    bonafide_probability: float
    logits: list[float]


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------


class AttentiveStatsPooling(nn.Module):
    """
    Learned attention over frames, returning weighted mean and std.

    Output dimension is ``2 * input_dim``.
    """

    def __init__(self, input_dim: int, attention_dim: int = 128) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden: (B, T, D)
        scores = self.attention(hidden)  # (B, T, 1)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))

        weights = torch.softmax(scores, dim=1)

        mean = torch.sum(weights * hidden, dim=1)

        variance = torch.sum(weights * hidden.pow(2), dim=1) - mean.pow(2)
        std = torch.sqrt(variance.clamp(min=1e-8))

        return torch.cat([mean, std], dim=-1)


# ---------------------------------------------------------------------------
# spectral branch
# ---------------------------------------------------------------------------


class SpectralBranch(nn.Module):
    """
    Log-Mel spectrogram -> small CNN -> fixed-length embedding.

    The mel filterbank spans the full 0-8 kHz Nyquist band on purpose. Neural
    vocoders leave their most reliable fingerprints in the top octave, and a
    filterbank capped at 4 kHz (a common default copied from ASR recipes)
    throws that evidence away.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate / 2,
            power=2.0,
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # The filterbank is numerically fragile in fp16 (squared magnitudes
        # underflow), so it always runs in fp32 even under autocast. The CNN
        # that follows is happy in whatever precision autocast picks.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrogram = self.mel(waveform.float())
            spectrogram = self.to_db(spectrogram)

            # Per-utterance standardisation, so loudness is not a feature.
            mean = spectrogram.mean(dim=(-2, -1), keepdim=True)
            std = spectrogram.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
            spectrogram = (spectrogram - mean) / std

        spectrogram = spectrogram.unsqueeze(1)  # (B, 1, n_mels, T)

        return self.project(self.features(spectrogram))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class AntiSpoofModel(nn.Module):
    """
    SSL encoder + optional spectral branch + fusion classifier.

    Parameters
    ----------
    encoder_name
        Any HuggingFace speech encoder: ``facebook/wav2vec2-base``,
        ``microsoft/wavlm-base-plus``, ``facebook/wav2vec2-xls-r-300m``.
    use_spectral
        Include the log-Mel CNN branch.
    freeze_encoder
        Freeze the whole SSL encoder (stage 1 of the training schedule).
    freeze_feature_extractor
        Freeze only the convolutional front end. Recommended always: those
        layers learn generic waveform filters, they are unstable to fine-tune,
        and freezing them saves both memory and time.
    unfreeze_top_layers
        Train only the last N transformer layers, freezing the rest. This is
        the memory lever that makes fine-tuning fit on a 6 GB card.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.2,
        use_spectral: bool = True,
        spectral_dim: int = 128,
        hidden_dim: int = 256,
        sample_rate: int = 16000,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = True,
        unfreeze_top_layers: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.encoder_name = encoder_name
        self.sample_rate = sample_rate

        self.encoder = AutoModel.from_pretrained(encoder_name)

        # Whether this checkpoint expects zero-mean unit-variance input is a
        # property of the checkpoint, not a universal rule: wav2vec2-base was
        # pretrained on raw waveforms (do_normalize=False) while most large /
        # XLS-R checkpoints expect normalised input. Getting this backwards
        # silently costs several points of EER, so read it rather than assume.
        try:
            extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
            self.do_normalize = bool(getattr(extractor, "do_normalize", False))
        except Exception:
            config = AutoConfig.from_pretrained(encoder_name)
            self.do_normalize = bool(getattr(config, "do_normalize", False))

        encoder_dim = self.encoder.config.hidden_size

        self.pooling = AttentiveStatsPooling(encoder_dim)

        fusion_dim = encoder_dim * 2

        self.spectral: SpectralBranch | None = None
        if use_spectral:
            self.spectral = SpectralBranch(
                sample_rate=sample_rate,
                output_dim=spectral_dim,
                dropout=dropout / 2,
            )
            fusion_dim += spectral_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.configure_freezing(
            freeze_encoder=freeze_encoder,
            freeze_feature_extractor=freeze_feature_extractor,
            unfreeze_top_layers=unfreeze_top_layers,
        )

    # -- freezing ----------------------------------------------------------

    def configure_freezing(
        self,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = True,
        unfreeze_top_layers: int | None = None,
    ) -> None:
        """(Re)apply a freezing policy. Safe to call between training stages."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze_encoder

        if not freeze_encoder and unfreeze_top_layers is not None:
            layers = self._transformer_layers()
            if layers is not None:
                cutoff = max(len(layers) - unfreeze_top_layers, 0)
                for index, layer in enumerate(layers):
                    for parameter in layer.parameters():
                        parameter.requires_grad = index >= cutoff

        if freeze_feature_extractor:
            self._freeze_feature_extractor()

    def _transformer_layers(self):
        encoder = getattr(self.encoder, "encoder", None)
        return getattr(encoder, "layers", None) if encoder is not None else None

    def _freeze_feature_extractor(self) -> None:
        extractor = getattr(self.encoder, "feature_extractor", None)
        if extractor is not None:
            for parameter in extractor.parameters():
                parameter.requires_grad = False

        projection = getattr(self.encoder, "feature_projection", None)
        if projection is not None and getattr(self, "_freeze_projection", False):
            for parameter in projection.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _is_decayed(name: str, parameter: torch.nn.Parameter) -> bool:
        """
        Whether weight decay should apply to this parameter.

        Biases and all normalisation scales/offsets are excluded. Decaying a
        LayerNorm gain towards zero shrinks the activations it is supposed to
        keep unit-scaled, and decaying a bias just adds noise - neither
        parameter is a capacity knob, so regularising them costs accuracy for
        nothing. Excluding them is standard for every transformer recipe.
        """

        if parameter.ndim <= 1:
            return False
        return not any(token in name.lower() for token in ("norm", ".bias"))

    def trainable_parameter_groups(
        self,
        encoder_lr: float,
        head_lr: float,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        Four param groups: {encoder, head} x {decayed, not decayed}.

        The LR split is the important one. The head starts from random weights
        and needs a large LR; the encoder is pretrained and needs a small one.
        A single LR for both either destroys the pretrained features or leaves
        the head undertrained.
        """

        head_modules = [("pooling", self.pooling), ("classifier", self.classifier)]
        if self.spectral is not None:
            head_modules.append(("spectral", self.spectral))

        buckets: dict[tuple[str, bool], list] = {
            ("encoder", True): [],
            ("encoder", False): [],
            ("head", True): [],
            ("head", False): [],
        }

        for name, parameter in self.encoder.named_parameters():
            if parameter.requires_grad:
                buckets[("encoder", self._is_decayed(name, parameter))].append(parameter)

        for prefix, module in head_modules:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    key = ("head", self._is_decayed(f"{prefix}.{name}", parameter))
                    buckets[key].append(parameter)

        learning_rates = {"encoder": encoder_lr, "head": head_lr}

        groups = []
        for (part, decayed), params in buckets.items():
            if not params:
                continue
            groups.append(
                {
                    "params": params,
                    "lr": learning_rates[part],
                    "weight_decay": weight_decay if decayed else 0.0,
                    "name": f"{part}{'' if decayed else '_nodecay'}",
                }
            )

        return groups

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_fraction": trainable / total if total else 0.0,
        }

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_values.ndim == 1:
            input_values = input_values.unsqueeze(0)

        waveform = input_values

        if self.do_normalize:
            mean = waveform.mean(dim=-1, keepdim=True)
            std = waveform.std(dim=-1, keepdim=True).clamp(min=1e-7)
            encoder_input = (waveform - mean) / std
        else:
            encoder_input = waveform

        outputs = self.encoder(
            input_values=encoder_input,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state  # (B, T, H)

        frame_mask = None
        if attention_mask is not None and hasattr(
            self.encoder, "_get_feature_vector_attention_mask"
        ):
            frame_mask = self.encoder._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )

        pooled = self.pooling(hidden, frame_mask)

        if self.spectral is not None:
            pooled = torch.cat([pooled, self.spectral(waveform)], dim=-1)

        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


class AntiSpoofInference:
    """
    Batch-safe scoring wrapper.

    ``predict`` returns one :class:`AntiSpoofOutput` per row, and
    ``predict_batch`` returns the raw spoof probabilities as a tensor. The
    earlier version indexed ``probabilities[0]`` unconditionally, so a batch of
    32 silently produced the first row's answer 32 times - which looks like a
    working evaluation and is not.
    """

    def __init__(self, model: AntiSpoofModel, device: str = "cpu") -> None:
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def logits(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim == 1:
            input_values = input_values.unsqueeze(0)
        return self.model(input_values.to(self.device))

    @torch.inference_mode()
    def predict_batch(self, input_values: torch.Tensor) -> torch.Tensor:
        """Spoof probability per row, shape (B,)."""

        probabilities = torch.softmax(self.logits(input_values).float(), dim=-1)
        return probabilities[:, 1]

    @torch.inference_mode()
    def predict(self, input_values: torch.Tensor) -> AntiSpoofOutput:
        """Score a single utterance. Raises on a batch, rather than lying."""

        if input_values.ndim > 1 and input_values.shape[0] != 1:
            raise ValueError(
                f"predict() scores one utterance, got a batch of "
                f"{input_values.shape[0]}. Use predict_batch()."
            )

        logits = self.logits(input_values)
        probabilities = torch.softmax(logits.float(), dim=-1)[0]

        return AntiSpoofOutput(
            bonafide_probability=float(probabilities[0]),
            spoof_probability=float(probabilities[1]),
            logits=logits[0].detach().cpu().tolist(),
        )

    @torch.inference_mode()
    def predict_many(self, input_values: torch.Tensor) -> list[AntiSpoofOutput]:
        logits = self.logits(input_values)
        probabilities = torch.softmax(logits.float(), dim=-1)

        return [
            AntiSpoofOutput(
                bonafide_probability=float(row[0]),
                spoof_probability=float(row[1]),
                logits=logit_row.detach().cpu().tolist(),
            )
            for row, logit_row in zip(probabilities, logits)
        ]
