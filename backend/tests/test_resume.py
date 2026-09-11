"""
Checkpoint resume.

Free Colab caps sessions around 12 hours and drops idle runtimes sooner, so a
resume that silently loses the optimiser trajectory is the difference between
"the run continued" and "the run restarted with warm weights and cold Adam
moments". These tests check the parts that are easy to get wrong and hard to
notice.

The model here is a tiny stand-in rather than wav2vec2 - the trainer only ever
touches ``state_dict``, ``parameters`` and ``trainable_parameter_groups``, so
the resume logic is exercised without a 360 MB download.
"""

import torch
import torch.nn as nn

from app.ml.config import TrainingConfig
from app.ml.training import AntiSpoofTrainer


class TinyModel(nn.Module):
    """Duck-types the parts of AntiSpoofModel the trainer uses."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 16)
        self.pooling = nn.Identity()
        self.spectral = None
        self.classifier = nn.Sequential(nn.Linear(16, 2))

    def forward(self, input_values, attention_mask=None):
        return self.classifier(torch.tanh(self.encoder(input_values)))

    def trainable_parameter_groups(self, encoder_lr, head_lr, weight_decay=0.01):
        return [
            {
                "params": [p for p in self.encoder.parameters() if p.requires_grad],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": [p for p in self.classifier.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def configure_freezing(self, **kwargs):
        pass

    def parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "frozen": 0, "trainable_fraction": 1.0}


def make_trainer(tmp_path, seed=0):
    torch.manual_seed(seed)

    config = TrainingConfig()
    config.device = "cpu"
    config.precision = "fp32"
    config.experiments_dir = str(tmp_path)
    config.experiment_id = "resume-test"

    return AntiSpoofTrainer(model=TinyModel(), config=config, device="cpu")


def batch(seed=0):
    torch.manual_seed(seed)
    return torch.randn(4, 16), torch.tensor([0, 1, 0, 1])


def test_resume_state_roundtrips(tmp_path):
    trainer = make_trainer(tmp_path)

    for _ in range(3):
        trainer.train_step(*batch())

    path = trainer.save_resume_state(
        tmp_path / "last.pt",
        {
            "epoch": 2,
            "best_eer": 0.041,
            "best_epoch": 2,
            "history": [{"epoch": 1}, {"epoch": 2}],
            "epochs_without_improvement": 0,
        },
    )

    restored = make_trainer(tmp_path, seed=99)
    state = restored.load_resume_state(path)

    assert state["epoch"] == 2
    assert state["best_eer"] == 0.041
    assert state["best_epoch"] == 2
    assert len(state["history"]) == 2


def test_resume_restores_weights_exactly(tmp_path):
    trainer = make_trainer(tmp_path)

    for _ in range(3):
        trainer.train_step(*batch())

    path = trainer.save_resume_state(tmp_path / "last.pt", {"epoch": 1})

    # A differently seeded trainer starts from different weights.
    restored = make_trainer(tmp_path, seed=99)
    assert not torch.allclose(
        restored.model.encoder.weight, trainer.model.encoder.weight
    )

    restored.load_resume_state(path)

    assert torch.allclose(restored.model.encoder.weight, trainer.model.encoder.weight)


def test_resume_restores_optimizer_moments(tmp_path):
    """
    The part that is easy to miss.

    Reloading weights alone gives a model that looks right and then takes a
    different optimisation path, because AdamW's exp_avg / exp_avg_sq buffers
    restart at zero. The next few steps behave like the start of training.
    """

    trainer = make_trainer(tmp_path)

    for _ in range(5):
        trainer.train_step(*batch())

    path = trainer.save_resume_state(tmp_path / "last.pt", {"epoch": 1})

    original = trainer.optimizer.state_dict()["state"]
    assert original, "optimizer should have accumulated moments"

    restored = make_trainer(tmp_path, seed=99)
    restored.load_resume_state(path)

    reloaded = restored.optimizer.state_dict()["state"]

    assert set(reloaded) == set(original)

    for key in original:
        assert torch.allclose(reloaded[key]["exp_avg"], original[key]["exp_avg"])
        assert torch.allclose(reloaded[key]["exp_avg_sq"], original[key]["exp_avg_sq"])
        assert reloaded[key]["step"] == original[key]["step"]


def test_resumed_trainer_continues_identically(tmp_path):
    """
    The behavioural check: a step taken after resuming must match the step the
    original trainer would have taken.
    """

    trainer = make_trainer(tmp_path)
    for _ in range(4):
        trainer.train_step(*batch())

    path = trainer.save_resume_state(tmp_path / "last.pt", {"epoch": 1})

    # Original takes one more step.
    trainer.train_step(*batch(seed=7))
    expected = trainer.model.encoder.weight.detach().clone()

    # Resumed trainer takes the same step.
    restored = make_trainer(tmp_path, seed=99)
    restored.load_resume_state(path)
    restored.train_step(*batch(seed=7))

    assert torch.allclose(restored.model.encoder.weight, expected, atol=1e-6)


def test_save_is_atomic(tmp_path):
    """
    A session killed mid-write must not leave a truncated checkpoint where the
    previous good one was.
    """

    trainer = make_trainer(tmp_path)
    path = tmp_path / "last.pt"

    trainer.save_resume_state(path, {"epoch": 1})
    trainer.save_resume_state(path, {"epoch": 2})

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert trainer.load_resume_state(path)["epoch"] == 2


def test_stage_is_carried_across_resume(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.stage = "finetune"

    path = trainer.save_resume_state(tmp_path / "last.pt", {"epoch": 2})

    restored = make_trainer(tmp_path, seed=99)
    assert restored.stage == "frozen"

    restored.load_resume_state(path)
    assert restored.stage == "finetune"


# ---------------------------------------------------------------------------
# precision selection
# ---------------------------------------------------------------------------


def test_bf16_only_on_ampere_and_newer(monkeypatch):
    """
    torch.cuda.is_bf16_supported() returns True on Turing (sm_75) because
    PyTorch counts *emulated* bf16 as supported. A Kaggle T4 reports bf16 True
    and then runs it far slower than fp16, because emulation bypasses the fp16
    tensor cores. Native bf16 starts at Ampere (sm_80).
    """

    from app.ml import training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # T4 / Turing
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    enabled, dtype = training.resolve_precision("auto", "cuda")
    assert enabled and dtype is torch.float16, "T4 must use fp16, not emulated bf16"

    # P100 / Pascal
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (6, 0))
    assert training.resolve_precision("auto", "cuda")[1] is torch.float16

    # RTX 3050 / Ampere
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    assert training.resolve_precision("auto", "cuda")[1] is torch.bfloat16

    # H100 / Hopper
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert training.resolve_precision("auto", "cuda")[1] is torch.bfloat16


def test_explicit_precision_is_respected(monkeypatch):
    from app.ml import training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))

    assert training.resolve_precision("fp16", "cuda")[1] is torch.float16
    assert training.resolve_precision("bf16", "cuda")[1] is torch.bfloat16
    assert training.resolve_precision("fp32", "cuda") == (False, None)


def test_cpu_never_uses_amp():
    from app.ml import training

    assert training.resolve_precision("auto", "cpu") == (False, None)
