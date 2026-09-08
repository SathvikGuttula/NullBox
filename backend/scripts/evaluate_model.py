"""
Evaluate a trained VoxShield checkpoint.

Reports EER, minDCF, ROC-AUC, a per-attack breakdown, and three operating
points (0.5, the EER threshold, and a 1%-false-alarm threshold). Writes a JSON
report plus ROC/DET plots next to the checkpoint.

    python backend/scripts/evaluate_model.py \
        --manifest datasets/manifests/test.csv \
        --model models/voxshield_antispoof.pt

Add ``--calibrate-on datasets/manifests/validation.csv`` to fit a temperature
on the validation split and report calibrated probabilities. Fit calibration on
validation, report on test - never both on the same split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.ml.calibration import ProbabilityCalibrator  # noqa: E402
from app.ml.dataset import (  # noqa: E402
    VoiceSpoofDataset,
    collate,
    describe,
    load_manifest,
)
from app.ml.metrics import (  # noqa: E402
    calculate_metrics,
    format_report,
    plot_curves,
    save_report,
)
from app.ml.model import AntiSpoofModel  # noqa: E402
from app.ml.progress import IterationProgress, PhaseTimer, format_count  # noqa: E402
from app.ml.training import describe_device, resolve_device, resolve_precision  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VoxShield anti-spoof checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument(
        "--prior-spoof",
        type=float,
        default=0.05,
        help="Prior probability of a spoof attempt, used by minDCF.",
    )
    parser.add_argument("--calibrate-on", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Write per-utterance scores.csv for error analysis.",
    )

    return parser.parse_args()


def load_model(model_path: Path, device: str) -> tuple[AntiSpoofModel, dict]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {model_path}\n"
            "Train one first: python backend/scripts/train_model.py"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{model_path} is a bare state_dict with no architecture record, so "
            "there is no safe way to rebuild the model it belongs to. Retrain "
            "with the current train_model.py, which saves the architecture "
            "alongside the weights."
        )

    architecture = checkpoint.get("architecture", {})

    model = AntiSpoofModel(
        encoder_name=architecture.get("encoder_name", "facebook/wav2vec2-base"),
        use_spectral=architecture.get("use_spectral", True),
        spectral_dim=architecture.get("spectral_dim", 128),
        hidden_dim=architecture.get("hidden_dim", 256),
        dropout=architecture.get("dropout", 0.2),
        sample_rate=architecture.get("sample_rate", 16000),
    )

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the rebuilt architecture:\n"
            f"  missing:    {list(missing)[:5]}\n"
            f"  unexpected: {list(unexpected)[:5]}"
        )

    return model.to(device).eval(), checkpoint


@torch.no_grad()
def score_manifest(
    model: AntiSpoofModel,
    manifest: Path,
    split: str | None,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    max_seconds: float,
    device: str,
    amp_enabled: bool,
    amp_dtype,
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    samples = load_manifest(manifest, split=split, limit=limit)

    stats = describe(samples)
    print(
        f"  {label}: {format_count(stats['total'])} clips   "
        f"bonafide {stats['bonafide']}   spoof {stats['spoof']}"
    )

    dataset = VoiceSpoofDataset(
        samples=samples, max_seconds=max_seconds, train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    scores: list[float] = []
    labels: list[int] = []

    progress = IterationProgress(
        total=len(loader), label=f"scoring {label}", log_every=max(len(loader) // 10, 1)
    )

    for batch in loader:
        input_values = batch["input_values"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(input_values)

        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]

        scores.extend(probabilities.cpu().tolist())
        labels.extend(batch["label"].tolist())

        progress.update(1)

    progress.close()

    return (
        np.asarray(scores),
        np.asarray(labels),
        [s.attack or "" for s in samples],
        [s.path for s in samples],
    )


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    amp_enabled, amp_dtype = resolve_precision(args.precision, device)

    print()
    print(f"device: {describe_device(device)}")

    with PhaseTimer(f"loading {args.model}"):
        model, checkpoint = load_model(args.model, device)

    metadata = checkpoint.get("metadata", {})
    architecture = checkpoint.get("architecture", {})

    print(f"  encoder          {architecture.get('encoder_name')}")
    print(f"  spectral branch  {architecture.get('use_spectral')}")
    if metadata:
        rendered = "   ".join(f"{k}={v}" for k, v in metadata.items())
        print(f"  trained          {rendered}")
    print()

    calibrator = ProbabilityCalibrator()

    if args.calibrate_on:
        validation_scores, validation_labels, _, _ = score_manifest(
            model,
            args.calibrate_on,
            None,
            None,
            args.batch_size,
            args.num_workers,
            args.max_seconds,
            device,
            amp_enabled,
            amp_dtype,
            "calibration set",
        )

        before = ProbabilityCalibrator.expected_calibration_error(
            validation_scores, validation_labels
        )
        temperature = calibrator.fit(validation_scores, validation_labels)
        after = ProbabilityCalibrator.expected_calibration_error(
            calibrator.calibrate_array(validation_scores), validation_labels
        )

        print(
            f"  calibration: T={temperature:.3f}   "
            f"ECE {before:.4f} -> {after:.4f}"
        )

        if calibrator.hit_bound:
            print(
                "  !! the temperature search hit its bound, so this is not a\n"
                "     converged fit. It means the scores carry little\n"
                "     information - fix the model before reading anything into\n"
                "     the calibrated probabilities."
            )

        calibrator.save(args.model.with_suffix(".calibration.json"))
        print()

    scores, labels, attacks, paths = score_manifest(
        model,
        args.manifest,
        args.split,
        args.limit,
        args.batch_size,
        args.num_workers,
        args.max_seconds,
        device,
        amp_enabled,
        amp_dtype,
        args.manifest.stem,
    )

    if calibrator.fitted:
        scores = calibrator.calibrate_array(scores)

    report = calculate_metrics(
        labels, scores, attacks=attacks, prior_spoof=args.prior_spoof
    )

    report["checkpoint"] = str(args.model)
    report["manifest"] = str(args.manifest)
    report["calibrated"] = calibrator.fitted
    report["calibration_temperature"] = calibrator.temperature
    report["architecture"] = architecture
    report["training_metadata"] = metadata

    print()
    print(format_report(report))
    print()

    output_dir = args.output_dir or args.model.parent / f"eval_{args.manifest.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_report(report, output_dir / "report.json")
    print(f"  report  {report_path}")

    if args.save_scores:
        scores_path = output_dir / "scores.csv"
        with scores_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("path,label,attack,spoof_score\n")
            for path, label, attack, score in zip(paths, labels, attacks, scores):
                handle.write(f"{path},{label},{attack},{score:.6f}\n")
        print(f"  scores  {scores_path}")

    if not args.no_plots:
        written = plot_curves(labels, scores, output_dir, prefix=args.manifest.stem)
        for path in written:
            print(f"  plot    {path}")
        if not written:
            print("  plots   skipped (matplotlib not installed)")

    print()
    print(
        "  Report EER, minDCF and the per-attack breakdown together. A single\n"
        "  accuracy figure on a 90%-spoof set is not a result."
    )
    print()


if __name__ == "__main__":
    main()
