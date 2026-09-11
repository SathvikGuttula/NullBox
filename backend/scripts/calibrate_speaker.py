"""
Calibrate speaker-verification thresholds against real trials.

Replaces the ``v0-UNCALIBRATED`` placeholders in ``SpeakerThresholds`` with
measured operating points, and reports the EER of the speaker branch so it can
be quoted the same way the anti-spoof EER is.

Trials are built from the manifests this project already produces — bonafide
utterances grouped by the ``speaker`` column — so no extra protocol files or
downloads are needed:

    target      two bonafide utterances from the same speaker
    nontarget   two bonafide utterances from different speakers

    python backend/scripts/calibrate_speaker.py \
        --manifest datasets/manifests/test.csv \
        --output models/speaker_thresholds.json

The LA eval split has 7,355 bonafide utterances across 67 speakers, which is
ample. Embedding is the expensive step, so ``--max-speakers`` and
``--max-probes`` bound the work; the defaults produce tens of thousands of
trials in a few minutes on a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.ml.dataset import load_manifest  # noqa: E402
from app.ml.progress import IterationProgress, PhaseTimer, format_count  # noqa: E402
from app.ml.speaker import SpeakerEncoder  # noqa: E402
from app.ml.speaker_calibration import (  # noqa: E402
    build_trials,
    calibrate,
    score_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate speaker-verification thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/speaker_thresholds.json"))
    parser.add_argument(
        "--apply",
        type=Path,
        default=None,
        help="Load thresholds calibrated elsewhere and report the error rates "
             "they actually achieve on THIS manifest. Calibrate on one speaker "
             "set and apply to another - quoting rates from the same speakers "
             "you tuned on is the speaker-verification version of reporting "
             "validation EER.",
    )
    parser.add_argument("--enroll-samples", type=int, default=5)
    parser.add_argument("--nontarget-per-probe", type=int, default=4)
    parser.add_argument("--max-probes", type=int, default=20,
                        help="Probe utterances per speaker.")
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--savedir", type=str, default="models/ecapa")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- gather bonafide utterances by speaker -----------------------------

    samples = load_manifest(args.manifest)

    bonafide = [s for s in samples if s.label == 0 and s.speaker]

    if not bonafide:
        raise SystemExit(
            f"{args.manifest} has no bonafide rows with a speaker id.\n"
            "Speaker calibration needs genuine audio labelled by speaker - "
            "rebuild the manifest with build_manifest.py, which writes the "
            "speaker column."
        )

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(bonafide):
        by_speaker[sample.speaker].append(index)

    if args.max_speakers:
        keep = sorted(by_speaker, key=lambda s: -len(by_speaker[s]))[: args.max_speakers]
        by_speaker = {s: by_speaker[s] for s in keep}

    print()
    print(f"  bonafide utterances : {format_count(len(bonafide))}")
    print(f"  speakers            : {len(by_speaker)}")
    print(
        f"  utterances/speaker  : "
        f"min {min(len(v) for v in by_speaker.values())}  "
        f"max {max(len(v) for v in by_speaker.values())}"
    )

    # -- build the trial list ----------------------------------------------

    trial_set = build_trials(
        by_speaker,
        enroll_samples=args.enroll_samples,
        nontarget_per_probe=args.nontarget_per_probe,
        max_probes_per_speaker=args.max_probes,
        seed=args.seed,
    )

    print(
        f"  trials              : {format_count(len(trial_set.trials))}  "
        f"({trial_set.targets} target, {trial_set.nontargets} nontarget)"
    )

    # Only utterances actually referenced need embedding.
    needed = sorted(
        {t.probe_index for t in trial_set.trials}
        | {i for indices in trial_set.enrollment.values() for i in indices}
    )
    print(f"  utterances to embed : {format_count(len(needed))}")
    print()

    # -- embed --------------------------------------------------------------

    with PhaseTimer(f"loading ECAPA-TDNN on {args.device}"):
        encoder = SpeakerEncoder(savedir=args.savedir, device=args.device)

    embeddings = np.zeros((len(bonafide), 192), dtype=np.float64)
    failures = 0

    progress = IterationProgress(
        total=len(needed), label="embedding", log_every=max(len(needed) // 20, 1)
    )

    for index in needed:
        try:
            embeddings[index] = encoder.embed_file(bonafide[index].path)
        except Exception:
            failures += 1
        progress.update(1)

    progress.close()

    if failures:
        print(f"  {failures} utterance(s) failed to embed and score as zero")

    # -- calibrate ----------------------------------------------------------

    labels, scores = score_trials(trial_set, embeddings)
    result = calibrate(labels, scores, encoder=encoder.model_name)
    result.speakers = len(trial_set.enrollment)

    target_scores = scores[labels == 1]
    nontarget_scores = scores[labels == 0]

    print()
    print("=" * 70)
    print("  SPEAKER VERIFICATION CALIBRATION")
    print("=" * 70)
    print(f"  encoder   {result.encoder}")
    print(f"  speakers  {result.speakers}     "
          f"trials {format_count(result.targets + result.nontargets)}")
    print()
    print(f"  target    similarity  mean {target_scores.mean():.3f}   "
          f"std {target_scores.std():.3f}")
    print(f"  nontarget similarity  mean {nontarget_scores.mean():.3f}   "
          f"std {nontarget_scores.std():.3f}")
    print()
    print(f"  EER       {result.eer * 100:.2f} %   at threshold "
          f"{result.eer_threshold:.4f}")
    print("-" * 70)
    print(f"  {'mode':<16}{'match':>9}{'no_match':>10}"
          f"{'false accept':>15}{'false reject':>15}")

    for name in ("high_security", "balanced", "low_friction"):
        thresholds = result.modes[name]
        point = result.operating_points[name]
        print(
            f"  {name:<16}{thresholds.match:>9.4f}{thresholds.no_match:>10.4f}"
            f"{point['false_accept_rate'] * 100:>14.2f}%"
            f"{point['false_reject_rate'] * 100:>14.2f}%"
        )

    print("=" * 70)
    print()

    # -- do thresholds from elsewhere still hold here? ----------------------

    if args.apply and args.apply.exists():
        borrowed = json.loads(args.apply.read_text(encoding="utf-8"))

        print()
        print("=" * 70)
        print(f"  THRESHOLDS FROM {args.apply} APPLIED TO THESE SPEAKERS")
        print("=" * 70)
        print(f"  {'mode':<16}{'match':>9}{'false accept':>16}{'false reject':>16}")

        target_mask = labels == 1
        nontarget_mask = labels == 0

        for name, mode in borrowed.get("modes", {}).items():
            threshold = float(mode["match"])
            false_accept = float((scores[nontarget_mask] >= threshold).mean())
            false_reject = float((scores[target_mask] < threshold).mean())
            print(
                f"  {name:<16}{threshold:>9.4f}"
                f"{false_accept * 100:>15.2f}%{false_reject * 100:>15.2f}%"
            )

        print("=" * 70)
        print()
        print("  If these rates are close to the ones the calibration targeted,")
        print("  the thresholds transfer. If they are far off, the two speaker")
        print("  sets differ enough that a single threshold cannot serve both -")
        print("  which is exactly what you need to know before deploying one.")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"  written to {args.output}")
    print()
    print("  Use a calibrated mode:")
    print("      from app.ml.speaker import SpeakerRegistry, SpeakerThresholds")
    print(f"      data = json.load(open('{args.output}'))['modes']['balanced']")
    print("      registry = SpeakerRegistry(SpeakerThresholds(**data))")
    print()
    print("  These thresholds describe THIS corpus and these enrolment")
    print("  conditions. Re-calibrate for telephone audio before quoting them")
    print("  for a phone deployment.")
    print()


if __name__ == "__main__":
    main()
