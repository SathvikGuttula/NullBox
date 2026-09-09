"""
Speaker verification — "does this voice belong to the person it claims to be?"

This is the half of VoxShield that the anti-spoof model cannot do. A real human
impersonating the CEO produces genuine, unsynthesised speech: the anti-spoof
detector correctly reports "not synthetic" and waves them through. Only an
identity check catches that attacker.

The two branches also fail in opposite directions, which is why both are
needed:

    cloned CEO voice      anti-spoof HIGH   identity may MATCH
    human impersonator    anti-spoof LOW    identity MISMATCH
    genuine CEO           anti-spoof LOW    identity MATCH

Design
------
Embeddings come from a pretrained ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb,
192-d). Nothing here is trained; this is assembly.

Enrollment averages several L2-normalised embeddings into a re-normalised
centroid, which is the standard speaker-prototype construction and is far more
stable than a single sample.

Only embeddings are stored, never audio (see the handoff's privacy model).
Embeddings are still biometric data and must be protected accordingly — the
registry below is deliberately a plain local store so that adding encryption
at rest is a single, obvious place to change.

On thresholds
-------------
There is no universal cosine threshold for "same speaker". A value that works
on studio audio fails on a phone line, and one tuned for VoxCeleb does not
transfer to your enrolment conditions. The defaults here are **placeholders**
and are labelled as such in ``SpeakerThresholds.version``. Calibrate them
against real trials — ``scripts/calibrate_speaker.py`` does this from the
ASVspoof LA ASV protocols, which contain exactly the target/nontarget pairs
this needs — before quoting any decision behaviour.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EMBEDDING_DIM = 192

MATCH = "MATCH"
UNCERTAIN = "UNCERTAIN"
NO_MATCH = "NO_MATCH"


# ---------------------------------------------------------------------------
# vector helpers - pure maths, no model required
# ---------------------------------------------------------------------------


def l2_normalize(vector: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Scale to unit length so cosine similarity is a plain dot product."""

    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm < epsilon:
        return vector

    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Normalises defensively."""

    return float(np.clip(l2_normalize(a) @ l2_normalize(b), -1.0, 1.0))


def centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Speaker prototype: mean of unit vectors, re-normalised.

    Normalising *before* averaging matters. Raw ECAPA embeddings vary in
    magnitude with utterance length and loudness, so a plain mean lets the
    longest enrolment sample dominate the prototype.
    """

    if not len(embeddings):
        raise ValueError("cannot build a centroid from zero embeddings")

    stacked = np.stack([l2_normalize(e) for e in embeddings])
    return l2_normalize(stacked.mean(axis=0))


def enrollment_consistency(embeddings: list[np.ndarray]) -> float:
    """
    Mean pairwise cosine among enrolment samples.

    A quality gate. If someone enrols five clips that are not all the same
    person - a shared handset, a mislabelled file, background speech - this
    drops, and the resulting prototype is a blur that will happily match
    several people. Better to refuse the enrolment than to serve a bad
    prototype for the life of the account.
    """

    if len(embeddings) < 2:
        return 1.0

    normalized = np.stack([l2_normalize(e) for e in embeddings])
    similarities = normalized @ normalized.T

    upper = similarities[np.triu_indices(len(normalized), k=1)]
    return float(upper.mean())


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


@dataclass
class SpeakerThresholds:
    """
    Decision boundaries, with a deliberate UNCERTAIN band between them.

    A two-way match/no-match split forces a guess on exactly the scores where
    the system knows least. A three-way split lets the policy engine ask for a
    second factor instead of inventing confidence it does not have.

    ``version`` travels with every decision so an audit record can say which
    calibration produced it.
    """

    match: float = 0.55
    no_match: float = 0.35
    version: str = "v0-UNCALIBRATED"

    # Operating modes from the handoff. All placeholders until calibrated.
    @classmethod
    def high_security(cls) -> "SpeakerThresholds":
        return cls(match=0.70, no_match=0.45, version="v0-UNCALIBRATED-high-security")

    @classmethod
    def balanced(cls) -> "SpeakerThresholds":
        return cls(match=0.55, no_match=0.35, version="v0-UNCALIBRATED-balanced")

    @classmethod
    def low_friction(cls) -> "SpeakerThresholds":
        return cls(match=0.45, no_match=0.25, version="v0-UNCALIBRATED-low-friction")

    @property
    def calibrated(self) -> bool:
        return "UNCALIBRATED" not in self.version

    def decide(self, similarity: float) -> str:
        if similarity >= self.match:
            return MATCH
        if similarity < self.no_match:
            return NO_MATCH
        return UNCERTAIN


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass
class SpeakerProfile:
    """An enrolled speaker. Holds embeddings, never audio."""

    identity: str
    centroid: np.ndarray
    sample_count: int
    consistency: float
    encoder: str
    created_at: str
    threshold_version: str = "v0-UNCALIBRATED"

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "centroid": [float(x) for x in self.centroid],
            "sample_count": self.sample_count,
            "consistency": self.consistency,
            "encoder": self.encoder,
            "created_at": self.created_at,
            "threshold_version": self.threshold_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        return cls(
            identity=data["identity"],
            centroid=np.asarray(data["centroid"], dtype=np.float64),
            sample_count=int(data["sample_count"]),
            consistency=float(data["consistency"]),
            encoder=data.get("encoder", "unknown"),
            created_at=data.get("created_at", ""),
            threshold_version=data.get("threshold_version", "v0-UNCALIBRATED"),
        )


@dataclass
class VerificationResult:
    identity: str
    similarity: float
    decision: str
    threshold_version: str
    enrollment_consistency: float
    enrollment_samples: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "similarity": round(self.similarity, 4),
            "decision": self.decision,
            "threshold_version": self.threshold_version,
            "enrollment_consistency": round(self.enrollment_consistency, 4),
            "enrollment_samples": self.enrollment_samples,
            "reasons": self.reasons,
            "calibrated": "UNCALIBRATED" not in self.threshold_version,
        }


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------


class SpeakerEncoder:
    """
    Pretrained ECAPA-TDNN wrapper.

    SpeechBrain is imported lazily so this module, the registry and the whole
    API remain importable without it — the anti-spoof path must not acquire a
    hard dependency on the speaker path.
    """

    MODEL = "speechbrain/spkrec-ecapa-voxceleb"

    _shared: "SpeakerEncoder | None" = None

    def __init__(
        self,
        model: str = MODEL,
        savedir: str | Path = "models/ecapa",
        device: str = "cpu",
        sample_rate: int = 16000,
    ) -> None:
        self.model_name = model
        self.device = device
        self.sample_rate = sample_rate

        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "speaker verification needs speechbrain:\n"
                "    pip install speechbrain\n"
                f"(original error: {exc})"
            ) from exc

        self._torch = torch

        # LocalStrategy.COPY, not the default SYMLINK. SpeechBrain symlinks
        # from the HuggingFace cache into savedir, and creating a symlink on
        # Windows needs administrator rights or Developer Mode - otherwise
        # loading dies with "WinError 1314: A required privilege is not held".
        # Copying costs a few tens of MB once and works everywhere.
        self.encoder = EncoderClassifier.from_hparams(
            source=model,
            savedir=str(savedir),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )

    @classmethod
    def get_shared(cls, **kwargs) -> "SpeakerEncoder":
        """One encoder per process - it is ~80 MB and stateless."""

        if cls._shared is None:
            cls._shared = cls(**kwargs)
        return cls._shared

    @classmethod
    def reset_shared(cls) -> None:
        cls._shared = None

    def embed(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Waveform -> unit-length 192-d speaker embedding."""

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            raise ValueError("cannot embed empty audio")

        if sample_rate is not None and sample_rate != self.sample_rate:
            import torchaudio.functional as AF

            audio = AF.resample(
                self._torch.from_numpy(audio), sample_rate, self.sample_rate
            ).numpy()

        tensor = self._torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        with self._torch.inference_mode():
            embedding = self.encoder.encode_batch(tensor)

        return l2_normalize(embedding.squeeze().detach().cpu().numpy())

    def embed_file(self, path: str | Path) -> np.ndarray:
        from app.ml.dataset import read_audio

        return self.embed(read_audio(path, self.sample_rate), self.sample_rate)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


class LiveSpeakerTracker:
    """
    Rolling speaker verification over the course of a call.

    Two things make this different from scoring one clip.

    First, it re-embeds on a duty cycle rather than per analysis window.
    Identity does not change mid-call, so running ECAPA every hop would burn
    GPU for no new information.

    Second, and more importantly, it accumulates the caller's embeddings into a
    running centroid and compares *that* to the enrolled prototype, rather than
    averaging per-window similarities. A three-second window is a noisy estimate
    of a voice; ten seconds of accumulated speech is a much better one. Both
    sides of the comparison are then prototypes built the same way, which is
    what the enrolment maths assumes.

    The confidence therefore improves as the call goes on, which is exactly the
    behaviour a live risk display should have.
    """

    def __init__(
        self,
        registry: "SpeakerRegistry",
        encoder,
        identity: str,
        sample_rate: int = 16000,
        interval_seconds: float = 3.0,
        max_embeddings: int = 20,
    ) -> None:
        self.registry = registry
        self.encoder = encoder
        self.identity = identity
        self.sample_rate = sample_rate
        self.interval_samples = int(sample_rate * interval_seconds)
        self.max_embeddings = max_embeddings

        self.buffer = np.zeros(0, dtype=np.float32)
        self.embeddings: list[np.ndarray] = []
        self.result: VerificationResult | None = None
        self.updates = 0

    @property
    def speech_seconds(self) -> float:
        """How much speech has gone into the current estimate."""

        return len(self.embeddings) * self.interval_samples / self.sample_rate

    def add_speech(self, audio: np.ndarray) -> VerificationResult | None:
        """
        Feed speech-only audio. Returns a fresh result when one was computed.

        Only speech should reach this - embedding silence produces a vector
        that says nothing about the speaker but still drags the centroid.
        """

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            return None

        self.buffer = np.concatenate([self.buffer, audio])

        if self.buffer.size < self.interval_samples:
            return None

        window = self.buffer[: self.interval_samples]
        self.buffer = self.buffer[self.interval_samples :]

        try:
            embedding = self.encoder.embed(window, self.sample_rate)
        except Exception:
            return None

        self.embeddings.append(embedding)

        # Keep the most recent N. A caller who hands the phone to someone else
        # should eventually be scored as that someone else, not as an average
        # of both for the rest of the call.
        if len(self.embeddings) > self.max_embeddings:
            self.embeddings = self.embeddings[-self.max_embeddings :]

        self.updates += 1
        self.result = self.registry.verify(self.identity, centroid(self.embeddings))

        return self.result

    def to_dict(self) -> dict | None:
        if self.result is None:
            return None

        data = self.result.to_dict()
        data["speech_seconds_analysed"] = round(self.speech_seconds, 1)
        data["updates"] = self.updates
        return data


class SpeakerRegistry:
    """
    Enrolled speakers and the verification decision.

    Deliberately independent of the encoder: it takes embeddings, not audio.
    That keeps every decision rule testable without an 80 MB model, and it
    means swapping ECAPA for something else touches one class.
    """

    MINIMUM_SAMPLES = 3
    RECOMMENDED_SAMPLES = 5
    MINIMUM_CONSISTENCY = 0.45

    def __init__(
        self,
        thresholds: SpeakerThresholds | None = None,
        store_path: str | Path | None = None,
    ) -> None:
        self.thresholds = thresholds or SpeakerThresholds.balanced()
        self.store_path = Path(store_path) if store_path else None
        self.profiles: dict[str, SpeakerProfile] = {}

        if self.store_path and self.store_path.exists():
            self.load()

    # -- enrollment --------------------------------------------------------

    def enroll(
        self,
        identity: str,
        embeddings: list[np.ndarray],
        encoder: str = SpeakerEncoder.MODEL,
        allow_inconsistent: bool = False,
    ) -> SpeakerProfile:
        """
        Build and store a speaker prototype from several embeddings.

        Refuses fewer than three samples, and refuses an inconsistent set
        unless explicitly overridden. Both refusals are deliberate: a bad
        prototype is not a one-off error, it is a permanently weakened account
        that quietly matches the wrong people.
        """

        if len(embeddings) < self.MINIMUM_SAMPLES:
            raise ValueError(
                f"need at least {self.MINIMUM_SAMPLES} enrolment samples for "
                f"{identity!r}, got {len(embeddings)}. A single sample makes a "
                f"prototype that reflects one recording condition rather than "
                f"one person."
            )

        consistency = enrollment_consistency(embeddings)

        if consistency < self.MINIMUM_CONSISTENCY and not allow_inconsistent:
            raise ValueError(
                f"enrolment samples for {identity!r} are inconsistent "
                f"(mean pairwise similarity {consistency:.3f} < "
                f"{self.MINIMUM_CONSISTENCY}). They may not all be the same "
                f"speaker, or some may be unusable. Pass allow_inconsistent=True "
                f"to override."
            )

        profile = SpeakerProfile(
            identity=identity,
            centroid=centroid(embeddings),
            sample_count=len(embeddings),
            consistency=consistency,
            encoder=encoder,
            created_at=datetime.now(timezone.utc).isoformat(),
            threshold_version=self.thresholds.version,
        )

        self.profiles[identity] = profile

        if self.store_path:
            self.save()

        return profile

    # -- verification ------------------------------------------------------

    def verify(self, identity: str, embedding: np.ndarray) -> VerificationResult:
        """Score a live embedding against a claimed identity."""

        profile = self.profiles.get(identity)

        if profile is None:
            return VerificationResult(
                identity=identity,
                similarity=0.0,
                decision=NO_MATCH,
                threshold_version=self.thresholds.version,
                enrollment_consistency=0.0,
                enrollment_samples=0,
                reasons=[f"No enrolled voice profile for '{identity}'"],
            )

        similarity = cosine_similarity(profile.centroid, embedding)
        decision = self.thresholds.decide(similarity)

        return VerificationResult(
            identity=identity,
            similarity=similarity,
            decision=decision,
            threshold_version=self.thresholds.version,
            enrollment_consistency=profile.consistency,
            enrollment_samples=profile.sample_count,
            reasons=self._reasons(profile, similarity, decision),
        )

    def _reasons(
        self, profile: SpeakerProfile, similarity: float, decision: str
    ) -> list[str]:
        reasons: list[str] = []

        if decision == NO_MATCH:
            reasons.append(
                f"Voice does not match the enrolled profile for "
                f"'{profile.identity}' (similarity {similarity:.2f})"
            )
        elif decision == UNCERTAIN:
            reasons.append(
                f"Voice is an ambiguous match for '{profile.identity}' "
                f"(similarity {similarity:.2f}) - secondary verification advised"
            )
        else:
            reasons.append(
                f"Voice matches the enrolled profile for '{profile.identity}' "
                f"(similarity {similarity:.2f})"
            )

        if profile.sample_count < self.RECOMMENDED_SAMPLES:
            reasons.append(
                f"Profile built from only {profile.sample_count} samples - "
                f"{self.RECOMMENDED_SAMPLES} recommended"
            )

        if profile.consistency < 0.6:
            reasons.append(
                f"Enrolment samples were only loosely consistent "
                f"({profile.consistency:.2f}), so this comparison is less reliable"
            )

        if not self.thresholds.calibrated:
            reasons.append(
                "Decision thresholds are uncalibrated placeholders - "
                "not suitable for production decisions"
            )

        return reasons

    def identify(self, embedding: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """
        Closest enrolled speakers, best first.

        Open-set identification is much harder than verification and this does
        not attempt it - a voice belonging to nobody enrolled still returns a
        ranked list. Use it for diagnostics, not decisions.
        """

        scored = [
            (identity, cosine_similarity(profile.centroid, embedding))
            for identity, profile in self.profiles.items()
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.store_path

        if path is None:
            raise ValueError("no store path configured")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "thresholds": asdict(self.thresholds),
                    "profiles": [p.to_dict() for p in self.profiles.values()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def load(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else self.store_path

        if path is None or not path.exists():
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        if "thresholds" in data:
            self.thresholds = SpeakerThresholds(**data["thresholds"])

        self.profiles = {
            entry["identity"]: SpeakerProfile.from_dict(entry)
            for entry in data.get("profiles", [])
        }

    def __len__(self) -> int:
        return len(self.profiles)

    def __contains__(self, identity: str) -> bool:
        return identity in self.profiles
