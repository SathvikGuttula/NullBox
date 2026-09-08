"""
Train the VoxShield anti-spoof detector.

The run always tells you what it is about to cost before it costs it: a short
speed probe measures real training steps, prints projected per-epoch and total
wall-clock, and only then starts. ``--dry-run`` stops after the projection.

Quick reference
---------------
Smoke test (a few dozen files, one epoch, seconds)::

    python backend/scripts/train_model.py --smoke-test

Size the real run without committing to it::

    python backend/scripts/train_model.py --dry-run

The baseline run::

    python backend/scripts/train_model.py --experiment-id exp001

Resume-friendly variants::

    --limit-train 5000        train on a subset
    --epochs 3                shorter schedule
    --batch-size 4            if VRAM is tight
    --no-augment              clean-audio ablation
    --encoder microsoft/wavlm-base-plus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.ml.augmentation import build_augmenter  # noqa: E402
from app.ml.config import TrainingConfig  # noqa: E402
from app.ml.dataset import (  # noqa: E402
    VoiceSpoofDataset,
    WaveformStore,
    describe,
    load_manifest,
)
from app.ml.model import AntiSpoofModel  # noqa: E402
from app.ml.progress import PhaseTimer, TrainingProgress, format_count  # noqa: E402
from app.ml.training import (  # noqa: E402
    AntiSpoofTrainer,
    build_loader,
    describe_device,
    git_commit,
    resolve_device,
    resolve_precision,
    set_seed,
)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the VoxShield anti-spoof detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train-manifest", type=str)
    data.add_argument("--validation-manifest", type=str)
    data.add_argument(
        "--cache-dir",
        type=str,
        help="Memmapped waveform cache from scripts/cache_dataset.py. "
             "Roughly 3-5x faster per epoch than decoding FLAC.",
    )
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-validation", type=int)
    data.add_argument("--max-seconds", type=float)
    data.add_argument(
        "--no-balance",
        dest="balance_classes",
        action="store_false",
        default=None,
        help="Disable class-balanced sampling (ASVspoof LA train is ~90%% spoof).",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--encoder", dest="encoder_name", type=str)
    model.add_argument(
        "--no-spectral",
        dest="use_spectral",
        action="store_false",
        default=None,
        help="Disable the log-Mel CNN branch (SSL-only ablation).",
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=int)
    optimisation.add_argument("--batch-size", type=int)
    optimisation.add_argument("--gradient-accumulation", type=int)
    optimisation.add_argument("--encoder-lr", type=float)
    optimisation.add_argument("--head-lr", type=float)
    optimisation.add_argument("--freeze-epochs", type=int)
    optimisation.add_argument("--unfreeze-top-layers", type=int)
    optimisation.add_argument("--early-stopping-patience", type=int)
    optimisation.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=None,
        help="Trade ~30%% speed for a large VRAM saving. Needed for xls-r on 6 GB.",
    )

    augmentation = parser.add_argument_group("augmentation")
    augmentation.add_argument(
        "--no-augment", dest="augment", action="store_false", default=None
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", type=str)
    runtime.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"])
    runtime.add_argument("--num-workers", type=int)
    runtime.add_argument("--seed", type=int)
    runtime.add_argument("--log-every", type=int)
    runtime.add_argument("--probe-steps", type=int)

    output = parser.add_argument_group("output")
    output.add_argument("--experiment-id", type=str)
    output.add_argument(
        "--experiments-dir",
        type=str,
        help="Where per-run artifacts go (default: experiments/). On Kaggle "
             "point this at /kaggle/working so it survives the session.",
    )
    output.add_argument("--model-out", type=str)
    output.add_argument("--notes", type=str)

    modes = parser.add_argument_group("modes")
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure speed, print the projected runtime, then stop.",
    )
    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="64 train / 32 val files, 1 epoch, no augmentation. Proves the "
             "pipeline runs end to end in well under a minute.",
    )
    modes.add_argument(
        "--config",
        type=str,
        help="Load a saved experiments/<id>/config.json and use it as the base.",
    )
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Continue from experiments/<id>/last.pt if it exists. Restores "
             "weights, optimizer moments, LR schedule and epoch counter. Use "
             "this on Colab, where sessions are capped and get disconnected.",
    )
    modes.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a specific checkpoint path instead of the default.",
    )

    return parser.parse_args()


def apply_smoke_test(config: TrainingConfig) -> TrainingConfig:
    config.limit_train = 64
    config.limit_validation = 32
    config.epochs = 1
    config.batch_size = 2
    config.gradient_accumulation = 1
    config.freeze_epochs = 1
    config.augment = False
    config.num_workers = 0
    config.log_every = 5
    config.probe_steps = 0
    config.experiment_id = "smoke-test"
    config.model_out = "models/smoke_test.pt"
    config.early_stopping_patience = 0
    return config


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------


def build_datasets(config: TrainingConfig):
    augmenter = build_augmenter(config, sample_rate=config.sample_rate)

    if config.cache_dir:
        cache_root = Path(config.cache_dir)

        # The cache subdirectory is named after the manifest, so switching to
        # the attack-held-out pair (train_heldout.csv / val_unseen.csv) picks
        # up datasets/cache/train_heldout automatically instead of silently
        # training on the wrong cached split.
        train_name = Path(config.train_manifest).stem
        validation_name = Path(config.validation_manifest).stem

        train_dir = cache_root / train_name
        validation_dir = cache_root / validation_name

        if not train_dir.exists():
            raise FileNotFoundError(
                f"no waveform cache at {train_dir}\n"
                f"Build it with:\n"
                f"  python backend/scripts/cache_dataset.py "
                f"--manifest {config.train_manifest} --output {train_dir}"
            )

        train_store = WaveformStore(train_dir)
        validation_store = (
            WaveformStore(validation_dir) if validation_dir.exists() else None
        )

        if validation_store is None:
            print(f"  no validation cache at {validation_dir} - training without it")

        print(
            f"  using waveform cache  "
            f"{train_store.size_bytes / (1024 ** 3):.2f} GB  "
            f"({format_count(len(train_store))} clips)"
        )

        train_dataset = VoiceSpoofDataset(
            store=train_store,
            max_seconds=config.max_seconds,
            train=True,
            augment=augmenter,
            seed=config.seed,
        )

        validation_dataset = (
            VoiceSpoofDataset(
                store=validation_store,
                max_seconds=config.max_seconds,
                train=False,
            )
            if validation_store is not None
            else None
        )

        return train_dataset, validation_dataset

    with PhaseTimer("reading train manifest"):
        train_samples = load_manifest(
            config.train_manifest, limit=config.limit_train
        )

    stats = describe(train_samples)
    print(
        f"  train      {format_count(stats['total'])} clips   "
        f"bonafide {stats['bonafide']}   spoof {stats['spoof']}   "
        f"({stats['spoof_ratio'] * 100:.1f}% spoof)"
    )
    if stats["attacks"]:
        print(f"  attacks    {', '.join(sorted(stats['attacks']))}")

    validation_dataset = None

    if config.validation_manifest and Path(config.validation_manifest).exists():
        with PhaseTimer("reading validation manifest"):
            validation_samples = load_manifest(
                config.validation_manifest, limit=config.limit_validation
            )

        validation_stats = describe(validation_samples)
        print(
            f"  validation {format_count(validation_stats['total'])} clips   "
            f"bonafide {validation_stats['bonafide']}   "
            f"spoof {validation_stats['spoof']}"
        )

        validation_dataset = VoiceSpoofDataset(
            samples=validation_samples,
            max_seconds=config.max_seconds,
            train=False,
        )

    train_dataset = VoiceSpoofDataset(
        samples=train_samples,
        max_seconds=config.max_seconds,
        train=True,
        augment=augmenter,
        seed=config.seed,
    )

    return train_dataset, validation_dataset


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config = (
        TrainingConfig.load(args.config) if args.config else TrainingConfig()
    )

    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    if args.smoke_test:
        config = apply_smoke_test(config)

    config.git_commit = git_commit()
    config.run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    device = resolve_device(config.device)
    amp_enabled, amp_dtype = resolve_precision(config.precision, device)

    print()
    print(f"device: {describe_device(device)}")

    if device == "cpu":
        print(
            "\n  WARNING: running on CPU. A wav2vec2 epoch over ASVspoof LA train\n"
            "  takes on the order of a day on CPU versus tens of minutes on this\n"
            "  GPU. Install the CUDA build of torch before starting a real run:\n"
            "    pip install --index-url https://download.pytorch.org/whl/cu130 \\\n"
            "        torch==2.14.0+cu130 torchaudio==2.11.0+cu130\n"
        )

    print()
    train_dataset, validation_dataset = build_datasets(config)

    train_loader = build_loader(train_dataset, config, train=True)
    validation_loader = (
        build_loader(validation_dataset, config, train=False)
        if validation_dataset is not None
        else None
    )

    with PhaseTimer(f"loading encoder {config.encoder_name}"):
        model = AntiSpoofModel(
            encoder_name=config.encoder_name,
            dropout=config.dropout,
            use_spectral=config.use_spectral,
            spectral_dim=config.spectral_dim,
            hidden_dim=config.hidden_dim,
            sample_rate=config.sample_rate,
            freeze_encoder=config.freeze_epochs > 0,
            freeze_feature_extractor=config.freeze_feature_extractor,
            unfreeze_top_layers=config.unfreeze_top_layers,
            gradient_checkpointing=config.gradient_checkpointing,
        )

    parameters = model.parameter_summary()

    trainer = AntiSpoofTrainer(model=model, config=config, device=device)

    steps_per_epoch = max(
        len(train_loader) // max(config.gradient_accumulation, 1), 1
    )

    progress = TrainingProgress(
        total_epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=config.effective_batch_size,
        run_dir=config.run_dir,
        log_every=config.log_every,
    )

    summary = config.summary()
    summary["device"] = describe_device(device)
    summary["precision"] = (
        f"{str(amp_dtype).replace('torch.', '')} autocast" if amp_enabled else "fp32"
    )
    summary["parameters"] = (
        f"{parameters['trainable']:,} trainable / {parameters['total']:,} total"
    )
    summary["train clips"] = format_count(len(train_dataset))
    summary["val clips"] = (
        format_count(len(validation_dataset)) if validation_dataset else "none"
    )

    progress.run_header(summary)
    config.save()

    # -- resume ------------------------------------------------------------

    resume_from = None

    if args.resume_from:
        resume_from = Path(args.resume_from)
        if not resume_from.exists():
            raise SystemExit(f"--resume-from: no such checkpoint: {resume_from}")
    elif args.resume:
        candidate = config.run_dir / "last.pt"
        if candidate.exists():
            resume_from = candidate
        else:
            print(f"  --resume: no checkpoint at {candidate}, starting fresh\n")

    # -- speed probe -------------------------------------------------------
    # Skipped when resuming: the probe runs optimiser steps, which would
    # corrupt the restored trajectory before the first real epoch.

    if resume_from is not None:
        config.probe_steps = 0

    if config.probe_steps > 0:
        seconds_per_step = trainer.probe_speed(train_loader, steps=config.probe_steps)

        if seconds_per_step > 0:
            projection = progress.estimate(
                seconds_per_step,
                note=f"measured over {config.probe_steps} real training steps "
                     f"(stage: frozen encoder)",
            )

            if config.freeze_epochs > 0 and config.freeze_epochs < config.epochs:
                print(
                    "  note: epochs after the unfreeze are slower - budget roughly\n"
                    "        1.7-2.2x this per-epoch figure for the fine-tuning stage.\n"
                )

            if args.dry_run:
                print("  --dry-run: stopping before training.\n")
                return
    elif args.dry_run:
        print("  --dry-run with --probe-steps 0: nothing measured.\n")
        return

    # The probe already ran a few optimiser steps on the untrained head. Reset
    # so the real run starts from the pretrained weights rather than from a
    # dozen arbitrary updates.
    if config.probe_steps > 0:
        trainer.optimizer = trainer._build_optimizer()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # -- train -------------------------------------------------------------

    result = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        progress=progress,
        resume_from=resume_from,
    )

    print()
    print(f"Experiment artifacts: {config.run_dir}")
    print(f"  config.json     the exact recipe")
    print(f"  history.json    per-epoch metrics")
    print(f"  progress.jsonl  per-step log")
    print(f"  summary.json    final result")
    print()

    if result.get("best_eer") is not None:
        print(f"Next: evaluate on the held-out test split")
        print(
            f"  python backend/scripts/evaluate_model.py "
            f"--manifest datasets/manifests/test.csv --model {config.model_out}"
        )
    print()


if __name__ == "__main__":
    main()
