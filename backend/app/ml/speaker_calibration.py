"""
Calibrating speaker-verification thresholds against real trials.

Why this exists
---------------
``SpeakerThresholds`` ships placeholders labelled ``v0-UNCALIBRATED``. There is
no universal cosine threshold for "same speaker" — a value tuned on studio
audio fails on a phone line, and one tuned on VoxCeleb does not transfer to
your enrolment conditions. Quoting a decision rate from an uncalibrated
threshold is the mistake the handoff warns about in §48 and §76.

This module turns "0.55 feels about right" into a measured operating point.

Where the trials come from
--------------------------
Not from the ASVspoof ASV protocol files, though those exist. The manifests
this project already builds carry ``speaker`` and ``label`` per utterance, so
target and nontarget pairs can be constructed directly:

    target      two bonafide utterances from the SAME speaker
    nontarget   two bonafide utterances from DIFFERENT speakers

Only bonafide audio is used. A spoofed clip is a question for the anti-spoof
branch; mixing it in here would measure something that is neither speaker
verification nor spoof detection.

The error rates, and which one is which
---------------------------------------
    false accept (FAR)   an impostor is accepted    — the SECURITY cost
    false reject (FRR)   a genuine speaker is not   — the USABILITY cost

These trade against each other, and the right balance is a policy decision
rather than a technical one, which is why three operating modes are produced
instead of a single number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.ml.metrics import equal_error_rate, threshold_at_false_alarm
from app.ml.speaker import SpeakerThresholds, centroid, cosine_similarity


@dataclass
class Trial:
    """One comparison: a probe utterance against a claimed identity."""

    identity: str
    probe_index: int
    is_target: bool


@dataclass
class TrialSet:
    enrollment: dict[str, list[int]]      # identity -> enrolment sample indices
    trials: list[Trial] = field(default_factory=list)

    @property
    def targets(self) -> int:
        return sum(1 for t in self.trials if t.is_target)

    @property
    def nontargets(self) -> int:
        return len(self.trials) - self.targets


def build_trials(
    speakers: dict[str, list[int]],
    enroll_samples: int = 5,
    nontarget_per_probe: int = 4,
    max_probes_per_speaker: int = 20,
    seed: int = 1234,
) -> TrialSet:
    """
    Split each speaker's utterances into enrolment and probes, then pair them.

    ``speakers`` maps a speaker id to the indices of their utterances.

    Enrolment samples are held out of the probe pool. Scoring a speaker against
    a prototype that contains the probe itself is a leak, and it inflates
    target similarity towards 1.0 — the calibration would then choose a
    threshold that no live comparison can ever reach.
    """

    rng = random.Random(seed)

    usable = {
        identity: indices
        for identity, indices in speakers.items()
        if len(indices) > enroll_samples
    }

    if len(usable) < 2:
        raise ValueError(
            f"need at least 2 speakers with more than {enroll_samples} "
            f"utterances each; found {len(usable)}. Lower --enroll-samples or "
            f"use a split with more audio per speaker."
        )

    enrollment: dict[str, list[int]] = {}
    probes: dict[str, list[int]] = {}

    for identity, indices in usable.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        enrollment[identity] = shuffled[:enroll_samples]
        probes[identity] = shuffled[enroll_samples : enroll_samples + max_probes_per_speaker]

    identities = sorted(enrollment)
    trials: list[Trial] = []

    for identity in identities:
        others = [o for o in identities if o != identity]

        for probe in probes[identity]:
            trials.append(Trial(identity, probe, True))

            for impostor in rng.sample(
                others, k=min(nontarget_per_probe, len(others))
            ):
                # The probe belongs to `identity`, but is claimed to be
                # `impostor` - a zero-effort impostor attempt.
                trials.append(Trial(impostor, probe, False))

    rng.shuffle(trials)

    return TrialSet(enrollment=enrollment, trials=trials)


def score_trials(
    trial_set: TrialSet,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(labels, similarities)`` — 1 for target, 0 for nontarget.

    This label convention lines up with ``app.ml.metrics`` so the same EER and
    threshold code serves both branches: there, "false alarm" is
    ``P(score >= t | label 0)``, which here is an impostor being accepted, and
    "miss" is a genuine speaker being rejected.
    """

    prototypes = {
        identity: centroid([embeddings[i] for i in indices])
        for identity, indices in trial_set.enrollment.items()
    }

    labels = np.empty(len(trial_set.trials), dtype=np.int64)
    scores = np.empty(len(trial_set.trials), dtype=np.float64)

    for position, trial in enumerate(trial_set.trials):
        labels[position] = 1 if trial.is_target else 0
        scores[position] = cosine_similarity(
            prototypes[trial.identity], embeddings[trial.probe_index]
        )

    return labels, scores


@dataclass
class CalibrationResult:
    eer: float
    eer_threshold: float
    targets: int
    nontargets: int
    speakers: int
    modes: dict[str, SpeakerThresholds] = field(default_factory=dict)
    operating_points: dict[str, dict] = field(default_factory=dict)
    encoder: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "eer": self.eer,
            "eer_percent": self.eer * 100.0,
            "eer_threshold": self.eer_threshold,
            "targets": self.targets,
            "nontargets": self.nontargets,
            "speakers": self.speakers,
            "encoder": self.encoder,
            "created_at": self.created_at,
            "modes": {
                name: {
                    "match": t.match,
                    "no_match": t.no_match,
                    "version": t.version,
                }
                for name, t in self.modes.items()
            },
            "operating_points": self.operating_points,
        }


def _rates_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    target = labels == 1
    nontarget = labels == 0

    return {
        "threshold": float(threshold),
        "false_accept_rate": float((scores[nontarget] >= threshold).mean()),
        "false_reject_rate": float((scores[target] < threshold).mean()),
    }


def calibrate(
    labels: np.ndarray,
    scores: np.ndarray,
    encoder: str = "",
    version_tag: str | None = None,
) -> CalibrationResult:
    """
    Derive three operating modes from scored trials.

        high_security   0.1% false accept   — an impostor almost never gets in
        balanced        the EER point       — errors split evenly
        low_friction    1% false reject     — genuine callers rarely challenged

    The ``no_match`` boundary is set where genuine speakers essentially never
    land (0.1% false reject). Below it, a rejection is safe to act on; between
    the two boundaries the answer is UNCERTAIN and the policy engine should ask
    for a second factor rather than guess.
    """

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0:
        raise ValueError("no trials to calibrate on")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "calibration needs both target and nontarget trials; got only "
            f"{'targets' if labels[0] == 1 else 'nontargets'}"
        )

    eer, eer_threshold = equal_error_rate(labels, scores)

    # All three modes are parameterised on the SAME axis - the false-accept
    # budget - and differ only in how much of it they will spend. That is what
    # makes them comparable, and it is what an operator is actually choosing
    # between: how many impostors am I willing to let through in exchange for
    # troubling fewer genuine callers?
    #
    #   high_security   spend LESS than the balance point   strictest
    #   balanced        the EER point                       errors split evenly
    #   low_friction    spend MORE than the balance point   most permissive
    #
    # The budgets are defined RELATIVE to the measured EER, not as fixed rates.
    # A fixed rate cannot bracket an unknown EER: 1% false accept is permissive
    # for a system with a 3% EER and strict for one with a 0.3% EER, so a fixed
    # pair silently collapses onto the balance point on one side or the other
    # depending on how good the model turned out to be. Both real calibration
    # runs hit exactly that - low_friction came out identical to balanced.
    #
    # Anchoring to the EER and an absolute floor keeps the three modes
    # meaningfully distinct across the whole range of model quality, while
    # still refusing to call anything "high security" if it admits more than
    # 0.1% of impostors.
    strict_budget = min(0.001, eer / 3.0) if eer > 0 else 0.001
    lenient_budget = max(0.01, eer * 3.0)

    strict = threshold_at_false_alarm(labels, scores, strict_budget)
    lenient = threshold_at_false_alarm(labels, scores, lenient_budget)

    # no_match is a different question - "is this confidently NOT the person?"
    # - so it comes off the target distribution: a score low enough that
    # genuine speakers essentially never reach it.
    floor = _threshold_at_false_reject(labels, scores, 0.005)

    # Enforce the ordering rather than assume it. When the classes separate
    # very well every budget is satisfied at once and the raw thresholds can
    # come out in any order - a "high security" mode more permissive than
    # "balanced" would be actively dangerous, since the mode name is what an
    # operator selects on. Where they genuinely coincide, they collapse, which
    # is the honest outcome.
    strict = max(strict, eer_threshold)
    lenient = min(lenient, eer_threshold)
    floor = min(floor, lenient)

    stamp = version_tag or datetime.now(timezone.utc).strftime("%Y%m%d")

    modes = {
        "high_security": SpeakerThresholds(
            match=float(strict), no_match=float(floor), version=f"v1-{stamp}-high-security"
        ),
        "balanced": SpeakerThresholds(
            match=float(eer_threshold), no_match=float(floor), version=f"v1-{stamp}-balanced"
        ),
        "low_friction": SpeakerThresholds(
            match=float(lenient), no_match=float(floor), version=f"v1-{stamp}-low-friction"
        ),
    }

    return CalibrationResult(
        eer=float(eer),
        eer_threshold=float(eer_threshold),
        targets=int((labels == 1).sum()),
        nontargets=int((labels == 0).sum()),
        speakers=0,
        modes=modes,
        operating_points={
            "high_security": _rates_at(labels, scores, strict),
            "balanced": _rates_at(labels, scores, eer_threshold),
            "low_friction": _rates_at(labels, scores, lenient),
            "no_match_floor": _rates_at(labels, scores, floor),
        },
        encoder=encoder,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _threshold_at_false_reject(
    labels: np.ndarray, scores: np.ndarray, target_rate: float
) -> float:
    """
    Highest threshold whose false-reject rate stays within budget.

    The mirror of ``threshold_at_false_alarm``: there the constraint is on
    impostors accepted, here on genuine speakers turned away. Highest, because
    among thresholds meeting the usability budget we want the most secure one.
    """

    target_scores = np.sort(scores[labels == 1])

    if target_scores.size == 0:
        return float(scores.min())

    # The k-th lowest genuine score is the highest threshold that rejects at
    # most k genuine speakers.
    index = int(np.floor(target_rate * target_scores.size))
    index = min(max(index, 0), target_scores.size - 1)

    return float(target_scores[index])
