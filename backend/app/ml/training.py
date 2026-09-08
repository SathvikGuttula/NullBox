"""
Training loop for the VoxShield anti-spoof detector.

The loop is deliberately boring; the interesting decisions are in the schedule:

Stage 1 - frozen encoder (``freeze_epochs``)
    Only the pooling, spectral branch and classifier train. The head starts
    from random weights, so its first gradients are large and noisy; letting
    them flow into a pretrained encoder is the classic way to destroy the
    features you paid for. It is also fast, because no gradient is computed
    through 12 transformer layers.

Stage 2 - partial fine-tune
    The top ``unfreeze_top_layers`` transformer layers unfreeze and train at a
    much smaller LR than the head. The convolutional front end stays frozen
    permanently: it learns generic waveform filters, it is numerically
    unstable to fine-tune, and freezing it is what keeps peak VRAM inside
    6 GB on a laptop card.

Everything that affects the result - seed, config, git commit, per-epoch
metrics - is written to ``experiments/<id>/`` alongside the checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from app.ml.config import TrainingConfig
from app.ml.dataset import (
    VoiceSpoofDataset,
    balanced_sample_weights,
    collate,
    seed_dataloader_worker,
)
from app.ml.metrics import calculate_metrics
from app.ml.model import AntiSpoofModel
from app.ml.progress import (
    IterationProgress,
    TrainingProgress,
    format_duration,
    gpu_memory_summary,
)


# ---------------------------------------------------------------------------
# environment helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def tune_cuda_backend() -> None:
    """
    Enable the throughput switches that are safe for this workload.

    ``cudnn.benchmark`` autotunes each convolution shape on first sight and
    caches the winner. It is normally a bad trade when input shapes vary,
    because every new shape pays the search again - but here every batch is
    exactly (B, 64000) and ``build_loader`` sets ``drop_last=True`` for
    training, so there is one shape and the search is paid once.

    TF32 costs a few mantissa bits on fp32 matmuls, which is irrelevant next
    to the bf16 autocast already in use, and speeds up the layers that stay in
    fp32 - notably the mel filterbank in the spectral branch.
    """

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_precision(requested: str, device: str) -> tuple[bool, torch.dtype | None]:
    """
    Return ``(amp_enabled, dtype)``.

    On Ampere and newer, bf16 is preferred over fp16: it has the same speed on
    tensor cores but the dynamic range of fp32, so there is no loss scaler to
    tune and no silent NaN when a gradient overflows.
    """

    if device == "cpu" or requested == "fp32":
        return False, None

    if requested == "fp16":
        return True, torch.float16

    if requested == "bf16":
        return True, torch.bfloat16

    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16

    return True, torch.float16


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def describe_device(device: str) -> str:
    if device == "cpu":
        return f"cpu ({os.cpu_count()} logical cores)"

    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    return f"{name}  {total:.1f} GB VRAM  (torch {torch.__version__})"


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    loss: float
    eer: float
    roc_auc: float
    accuracy: float
    report: dict

    def as_progress_extras(self) -> dict:
        return {
            "val_loss": self.loss,
            "val_eer%": self.eer * 100.0,
            "val_auc": self.roc_auc,
            "val_acc%": self.accuracy * 100.0,
        }


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class AntiSpoofTrainer:
    """
    Owns the model, optimiser, schedule and checkpointing for one run.

    ``fit`` runs the whole schedule. ``train_step`` remains available for the
    single-batch smoke test that the tests use.
    """

    def __init__(
        self,
        model: AntiSpoofModel,
        config: TrainingConfig | None = None,
        learning_rate: float | None = None,
        device: str = "cuda",
    ) -> None:
        self.config = config or TrainingConfig()

        if learning_rate is not None:
            self.config.head_lr = learning_rate

        self.device = resolve_device(device or self.config.device)
        self.amp_enabled, self.amp_dtype = resolve_precision(
            self.config.precision, self.device
        )

        tune_cuda_backend()

        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        # fp16 needs a loss scaler; bf16 does not.
        self.scaler = torch.amp.GradScaler(
            self.device.split(":")[0],
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.stage = "frozen" if self.config.freeze_epochs > 0 else "finetune"

    # -- optimiser / schedule ---------------------------------------------

    def _build_optimizer(self) -> AdamW:
        groups = self.model.trainable_parameter_groups(
            encoder_lr=self.config.encoder_lr,
            head_lr=self.config.head_lr,
            weight_decay=self.config.weight_decay,
        )

        if not groups:
            raise RuntimeError(
                "no trainable parameters - every module is frozen"
            )

        # weight_decay is set per group by the model (biases and norms get
        # zero), so it must not be passed here as a global override.
        return AdamW(groups)

    def _head_lr(self) -> float:
        """
        The head's current LR, for display.

        There are four param groups (encoder/head x decayed/not), so indexing
        by position would report whichever happened to land last. Name lookup
        keeps the logged number meaningful when the groups change.
        """

        for group in self.optimizer.param_groups:
            if str(group.get("name", "")).startswith("head"):
                return float(group["lr"])

        return float(self.optimizer.param_groups[0]["lr"])

    def _build_scheduler(self, total_steps: int):
        warmup = max(int(total_steps * self.config.warmup_ratio), 1)

        def curve(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, curve)

    def enter_finetune_stage(self) -> dict:
        """
        Unfreeze the top transformer layers and rebuild the optimiser.

        The optimiser must be rebuilt, not merely extended: AdamW keeps
        per-parameter moment buffers, and parameters that were frozen have
        none. Adding them to an existing optimiser silently gives them a
        stale or absent state.
        """

        self.model.configure_freezing(
            freeze_encoder=False,
            freeze_feature_extractor=self.config.freeze_feature_extractor,
            unfreeze_top_layers=self.config.unfreeze_top_layers,
        )

        self.optimizer = self._build_optimizer()
        self.stage = "finetune"

        return self.model.parameter_summary()

    # -- single step -------------------------------------------------------

    def train_step(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """One optimiser step on one batch. Used by the smoke test."""

        self.model.train()

        input_values = input_values.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            logits = self.model(input_values)
            loss = self.loss_fn(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    # -- speed probe -------------------------------------------------------

    def probe_speed(self, loader: DataLoader, steps: int = 12) -> float:
        """
        Time a handful of real training steps and return seconds per step.

        The first few steps are excluded: they pay for cuDNN autotuning, lazy
        CUDA context creation and the first DataLoader worker spin-up, and
        including them inflates the projection by several times.
        """

        self.model.train()

        iterator = iter(loader)
        warmup = min(3, max(steps // 3, 1))
        timings: list[float] = []

        for index in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            start = time.perf_counter()
            self.train_step(batch["input_values"], batch["label"])

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            if index >= warmup:
                timings.append(elapsed)

        if not timings:
            return 0.0

        # Median, not mean: one stalled step from a Windows worker respawn
        # should not dominate the estimate.
        return float(np.median(timings))

    # -- validation --------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        label: str = "validating",
        show_progress: bool = True,
    ) -> ValidationResult:
        self.model.eval()

        losses: list[float] = []
        all_scores: list[float] = []
        all_labels: list[int] = []

        progress = (
            IterationProgress(
                total=len(loader),
                label=label,
                log_every=max(len(loader) // 4, 1),
            )
            if show_progress
            else None
        )

        for batch in loader:
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.split(":")[0],
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(input_values)
                loss = self.loss_fn(logits, labels)

            losses.append(loss.item())

            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]

            all_scores.extend(probabilities.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()

        report = calculate_metrics(all_labels, all_scores)

        return ValidationResult(
            loss=float(np.mean(losses)) if losses else float("nan"),
            eer=report.get("eer") or float("nan"),
            roc_auc=report.get("roc_auc") or float("nan"),
            accuracy=report["accuracy"],
            report=report,
        )

    # -- checkpointing -----------------------------------------------------

    def save_checkpoint(self, path: str | Path, metadata: dict) -> Path:
        """
        Save weights *and* the recipe that produced them.

        A bare ``state_dict`` cannot be reloaded safely: nothing records which
        encoder it was built on or whether the spectral branch was present, so
        loading it into a mismatched architecture either explodes or, worse,
        silently loads a subset of the tensors.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "format_version": 2,
                "state_dict": self.model.state_dict(),
                "architecture": {
                    "encoder_name": self.config.encoder_name,
                    "use_spectral": self.config.use_spectral,
                    "spectral_dim": self.config.spectral_dim,
                    "hidden_dim": self.config.hidden_dim,
                    "dropout": self.config.dropout,
                    "sample_rate": self.config.sample_rate,
                },
                "config": self.config.to_dict(),
                "metadata": metadata,
            },
            path,
        )

        return path

    # -- resume ------------------------------------------------------------

    def save_resume_state(self, path: str | Path, state: dict) -> Path:
        """
        Everything needed to continue this run in a fresh process.

        Weights alone are not enough. AdamW's moment buffers and the LR
        schedule's step count are part of the optimisation trajectory: reload
        without them and the first steps after a resume take a different path
        than they would have, which quietly changes the result. On a platform
        that disconnects sessions - free Colab caps at ~12 hours and drops idle
        runtimes sooner - that difference lands in the middle of every run.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "stage": self.stage,
            "config": self.config.to_dict(),
            "architecture": {
                "encoder_name": self.config.encoder_name,
                "use_spectral": self.config.use_spectral,
                "spectral_dim": self.config.spectral_dim,
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
                "sample_rate": self.config.sample_rate,
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            **state,
        }

        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()

        # Write to a temporary file and move it into place, so a session killed
        # mid-write leaves the previous good checkpoint intact rather than a
        # truncated one that fails to load.
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

        return path

    def load_resume_state(self, path: str | Path) -> dict:
        """
        Restore a run. Returns the bookkeeping fields for ``fit`` to continue.

        The freezing policy is re-applied *before* the optimizer state loads.
        AdamW's state dict is keyed by parameter position within its groups, so
        loading a fine-tuning-stage optimizer into a still-frozen model raises
        a size mismatch - or worse, silently maps moments onto the wrong
        tensors.
        """

        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint.get("stage") == "finetune" and self.stage != "finetune":
            self.enter_finetune_stage()

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.stage = checkpoint.get("stage", self.stage)

        rng = checkpoint.get("rng") or {}
        try:
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        except Exception:
            # A mismatched RNG format must not block a resume; the run stays
            # correct, it just diverges from the original random stream.
            pass

        return {
            "epoch": int(checkpoint.get("epoch", 0)),
            "best_eer": float(checkpoint.get("best_eer", float("inf"))),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "history": checkpoint.get("history", []),
            "epochs_without_improvement": int(
                checkpoint.get("epochs_without_improvement", 0)
            ),
            "scheduler": checkpoint.get("scheduler"),
        }

    # -- the full run ------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        progress: TrainingProgress | None = None,
        resume_from: str | Path | None = None,
    ) -> dict:
        config = self.config

        steps_per_epoch = max(
            len(train_loader) // max(config.gradient_accumulation, 1), 1
        )

        progress = progress or TrainingProgress(
            total_epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=config.effective_batch_size,
            run_dir=config.run_dir,
            log_every=config.log_every,
        )

        self.scheduler = self._build_scheduler(steps_per_epoch * config.epochs)

        best_eer = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        history: list[dict] = []
        start_epoch = 1

        resume_path = Path(resume_from) if resume_from else None

        if resume_path is not None and resume_path.exists():
            restored = self.load_resume_state(resume_path)

            start_epoch = restored["epoch"] + 1
            best_eer = restored["best_eer"]
            best_epoch = restored["best_epoch"]
            history = restored["history"]
            epochs_without_improvement = restored["epochs_without_improvement"]

            if restored["scheduler"] is not None:
                try:
                    self.scheduler.load_state_dict(restored["scheduler"])
                except Exception:
                    pass

            progress.note(
                f"resumed from {resume_path} - completed {restored['epoch']} "
                f"epoch(s), best EER "
                + (
                    f"{best_eer * 100:.3f}% at epoch {best_epoch}"
                    if math.isfinite(best_eer)
                    else "n/a"
                )
            )

            if start_epoch > config.epochs:
                progress.note(
                    f"all {config.epochs} epochs already completed - nothing to do. "
                    f"Raise --epochs to continue training."
                )
                return {
                    "best_eer": best_eer if math.isfinite(best_eer) else None,
                    "best_epoch": best_epoch if best_epoch > 0 else None,
                    "epochs_run": len(history),
                    "history": history,
                    "checkpoint": str(config.model_out),
                    "resumed": True,
                }

        resume_state_path = config.run_dir / "last.pt"

        for epoch in range(start_epoch, config.epochs + 1):
            if epoch == config.freeze_epochs + 1 and config.freeze_epochs > 0:
                summary = self.enter_finetune_stage()
                self.scheduler = self._build_scheduler(
                    steps_per_epoch * (config.epochs - epoch + 1)
                )
                progress.note(
                    f"  stage change -> fine-tuning top {config.unfreeze_top_layers} "
                    f"encoder layers  "
                    f"({summary['trainable']:,} of {summary['total']:,} params trainable)"
                )

            progress.start_epoch(epoch)
            self._train_one_epoch(train_loader, progress)

            extras: dict = {"stage": self.stage}
            validation: ValidationResult | None = None

            should_validate = (
                validation_loader is not None
                and (epoch % max(config.validate_every, 1) == 0 or epoch == config.epochs)
            )

            if should_validate:
                validation = self.evaluate(validation_loader)
                extras.update(validation.as_progress_extras())

            result = progress.end_epoch(**extras)

            record = {
                "epoch": epoch,
                "stage": self.stage,
                "train_loss": result.loss,
                "train_accuracy": result.accuracy,
                "seconds": result.seconds,
            }

            if validation is not None:
                record.update(
                    {
                        "val_loss": validation.loss,
                        "val_eer": validation.eer,
                        "val_roc_auc": validation.roc_auc,
                        "val_accuracy": validation.accuracy,
                    }
                )

                improved = validation.eer < best_eer - 1e-6

                if improved:
                    best_eer = validation.eer
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    self.save_checkpoint(
                        config.model_out,
                        metadata={
                            "epoch": epoch,
                            "val_eer": validation.eer,
                            "val_roc_auc": validation.roc_auc,
                            "val_accuracy": validation.accuracy,
                            "git_commit": config.git_commit,
                            "experiment_id": config.experiment_id,
                        },
                    )

                    progress.note(
                        f"  new best EER {validation.eer * 100:.3f}% "
                        f"-> saved {config.model_out}"
                    )
                else:
                    epochs_without_improvement += 1
                    progress.note(
                        f"  no improvement ({epochs_without_improvement}"
                        f"/{config.early_stopping_patience})"
                        f"  best {best_eer * 100:.3f}% at epoch {best_epoch}"
                    )

            history.append(record)

            (config.run_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

            # Written every epoch, not only on improvement: the point is to
            # survive a killed session, and a session dies just as readily
            # during an epoch that made the model slightly worse.
            self.save_resume_state(
                resume_state_path,
                {
                    "epoch": epoch,
                    "best_eer": best_eer,
                    "best_epoch": best_epoch,
                    "history": history,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )

            if (
                config.early_stopping_patience > 0
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                progress.note(
                    f"  early stopping: no EER improvement for "
                    f"{epochs_without_improvement} epochs"
                )
                break

        # A run with no validation set still needs a checkpoint.
        if best_epoch < 0:
            self.save_checkpoint(
                config.model_out,
                metadata={
                    "epoch": config.epochs,
                    "note": "no validation set - final epoch saved",
                    "git_commit": config.git_commit,
                    "experiment_id": config.experiment_id,
                },
            )

        summary = {
            "best_eer": best_eer if math.isfinite(best_eer) else None,
            "best_epoch": best_epoch if best_epoch > 0 else None,
            "epochs_run": len(history),
            "history": history,
            "checkpoint": str(config.model_out),
        }

        progress.run_footer(
            **{
                "best val EER": (
                    f"{best_eer * 100:.3f}% (epoch {best_epoch})"
                    if math.isfinite(best_eer)
                    else "n/a"
                ),
                "checkpoint": config.model_out,
                "experiment": str(config.run_dir),
            }
        )

        (config.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        return summary

    def _train_one_epoch(
        self,
        loader: DataLoader,
        progress: TrainingProgress,
    ) -> None:
        config = self.config
        accumulation = max(config.gradient_accumulation, 1)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        window_loss = 0.0
        window_correct = 0
        window_total = 0

        for index, batch in enumerate(loader):
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.split(":")[0],
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(input_values)
                loss = self.loss_fn(logits, labels)

            # Scale so that accumulated gradients average rather than sum.
            self.scaler.scale(loss / accumulation).backward()

            window_loss += loss.item()
            window_correct += (logits.argmax(dim=-1) == labels).sum().item()
            window_total += labels.numel()

            if (index + 1) % accumulation != 0:
                continue

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            progress.step(
                loss=window_loss / accumulation,
                accuracy=window_correct / max(window_total, 1),
                lr=self._head_lr(),
            )

            window_loss = 0.0
            window_correct = 0
            window_total = 0

            if progress.epoch_step >= progress.steps_per_epoch:
                break


# ---------------------------------------------------------------------------
# loader construction
# ---------------------------------------------------------------------------


def build_loader(
    dataset: VoiceSpoofDataset,
    config: TrainingConfig,
    train: bool,
) -> DataLoader:
    """
    Build a DataLoader with Windows-appropriate defaults.

    ``persistent_workers`` matters a lot here: Windows has no fork, so every
    epoch would otherwise re-spawn and re-import the workers, which on this
    project costs several seconds per epoch before a single batch arrives.
    """

    sampler = None
    shuffle = train

    if train and config.balance_classes:
        weights = balanced_sample_weights(dataset.labels)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    workers = config.num_workers if train else max(config.num_workers // 2, 0)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=seed_dataloader_worker if workers > 0 else None,
    )
