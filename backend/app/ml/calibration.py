"""
Probability calibration.

A neural classifier's softmax output is a score, not a probability. Modern
networks are badly overconfident: a model can be 99% sure and wrong 20% of the
time. That is tolerable when all you do is rank, and unacceptable the moment a
number is shown to an operator or fed into a risk score that mixes it with
other signals - the miscalibrated branch silently dominates the fusion.

Why temperature scaling alone is not enough here
------------------------------------------------
Temperature scaling divides the logit by a scalar T. It is monotone, so it
cannot change the ranking (EER and ROC-AUC are untouched), and it is the right
tool when a model's *confidence* is wrong but its *decision boundary* is not.

This model's boundary is wrong. Measured on the 71,237-utterance ASVspoof 2019
LA evaluation set, bonafide logits sit at -3.44 +/- 0.31 and spoof logits at
+2.67 +/- 1.95, and the equal-error operating point falls at a raw probability
of 0.044 - not 0.5. Judging "is this synthetic?" by ``p >= 0.5`` therefore
misses 12% of attacks. Temperature scaling cannot repair that, because
dividing a logit by a positive T never changes its sign: a boundary below 0.5
stays below 0.5 for every T. ``test_temperature_cannot_move_the_boundary``
pins this down.

The fix is an affine map in log-odds space - Platt scaling:

    calibrated = sigmoid(scale * logit(p) + bias)

``scale`` does what temperature did; ``bias`` moves the boundary. The shift is
real and large: the training run used class-balanced sampling and label
smoothing, while the evaluation set is 89.7% spoof, so the model's implied
prior and the data's prior disagree by several nats.

On priors
---------
A fitted ``bias`` bakes in the prior of whatever data it was fitted on. Real
calls are not 89.7% spoof, so ``with_prior`` re-expresses the calibration at a
chosen operating prior by shifting the bias in log-odds. The default shipped
calibration uses an equal prior (0.5), which is the prior-independent
"how much does the audio itself say" reading and puts the decision boundary
back at a meaningful 0.5. A deployment with a real base rate should call
``with_prior`` with it.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

EPSILON = 1e-7


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _log_odds(prior: float) -> float:
    prior = min(max(float(prior), EPSILON), 1.0 - EPSILON)
    return math.log(prior / (1.0 - prior))


def _sigmoid_array(z):
    """Overflow-safe sigmoid: exp(-z) overflows for large negative z."""

    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)

    return out


class ProbabilityCalibrator:
    """
    Affine calibration in log-odds space.

        calibrator = ProbabilityCalibrator()
        calibrator.fit_affine(eval_probabilities, eval_labels)
        deployment = calibrator.with_prior(0.5, fitted_prior=eval_labels.mean())
        p = deployment.calibrate(0.93)

    ``scale`` 1.0 with ``bias`` 0.0 is the identity, so an unfitted calibrator
    is a no-op and the pipeline still runs before calibration data exists.

    ``temperature`` remains available as ``1 / scale`` for the pure
    temperature-scaling case, so existing callers and saved files keep working.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        scale: float | None = None,
        bias: float = 0.0,
    ) -> None:
        if scale is not None:
            self.scale = max(float(scale), 1e-6)
        else:
            self.scale = 1.0 / max(float(temperature), 1e-6)

        self.bias = float(bias)
        self.fitted = False
        self.hit_bound = False
        self.metadata: dict = {}

    # -- temperature compatibility ----------------------------------------

    @property
    def temperature(self) -> float:
        return 1.0 / self.scale

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.scale = 1.0 / max(float(value), 1e-6)

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.bias == 0.0

    # -- application -------------------------------------------------------

    def calibrate(self, probability: float) -> float:
        """Apply the calibration to a single spoof probability."""

        if self.is_identity:
            return min(max(float(probability), 0.0), 1.0)

        z = self.scale * float(_logit(probability)) + self.bias

        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))

        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def calibrate_array(self, probabilities) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if self.is_identity:
            return np.clip(probabilities, 0.0, 1.0)

        return _sigmoid_array(self.scale * _logit(probabilities) + self.bias)

    def log_odds(self, probability: float) -> float:
        """
        The calibrated score in log-odds.

        Above roughly +36 the probability saturates to 1.0 in float64 and
        thresholds expressed as probabilities stop discriminating. Operating
        points in that region must be compared here instead.
        """

        return self.scale * float(_logit(probability)) + self.bias

    # -- priors ------------------------------------------------------------

    def with_prior(
        self, target_prior: float, fitted_prior: float
    ) -> "ProbabilityCalibrator":
        """
        Re-express this calibration at a different base rate.

        A bias fitted on data that was ``fitted_prior`` spoof encodes that
        prior. Subtracting its log-odds and adding the target's moves the
        calibration to the new operating prior without touching the ranking.
        """

        if not 0.0 < target_prior < 1.0:
            raise ValueError(f"target_prior must be in (0, 1), got {target_prior}")
        if not 0.0 < fitted_prior < 1.0:
            raise ValueError(f"fitted_prior must be in (0, 1), got {fitted_prior}")

        shifted = ProbabilityCalibrator(
            scale=self.scale,
            bias=self.bias - _log_odds(fitted_prior) + _log_odds(target_prior),
        )
        shifted.fitted = self.fitted
        shifted.hit_bound = self.hit_bound
        shifted.metadata = {
            **self.metadata,
            "prior": float(target_prior),
            "fitted_prior": float(fitted_prior),
        }
        return shifted

    # -- fitting -----------------------------------------------------------

    def fit_affine(self, probabilities, labels, max_iterations: int = 200) -> tuple:
        """
        Fit ``scale`` and ``bias`` by minimising negative log-likelihood.

        Two-parameter logistic regression on the logit. The objective is
        convex, so Newton's method converges in a handful of steps from a
        coarse grid start; the grid exists only to keep the first Newton step
        away from a region where the Hessian is numerically singular.

        Fit this on held-out data and report calibration quality on data the
        fit never saw - see ``scripts/calibrate_detector.py``, which splits the
        evaluation set in half stratified by attack type.
        """

        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels).astype(np.float64)

        if probabilities.size == 0:
            raise ValueError("no samples to calibrate on")

        if probabilities.size != labels.size:
            raise ValueError(
                f"got {probabilities.size} probabilities and {labels.size} labels"
            )

        if np.unique(labels).size < 2:
            raise ValueError(
                "calibration needs both classes present; got only label "
                f"{int(labels.flat[0])}"
            )

        logits = _logit(probabilities)

        def nll(a: float, b: float) -> float:
            z = a * logits + b
            return float(
                np.mean(np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z))))
            )

        best = (1.0, 0.0, nll(1.0, 0.0))
        for a in np.geomspace(0.05, 8.0, 60):
            for b in np.linspace(-14.0, 14.0, 57):
                value = nll(float(a), float(b))
                if value < best[2]:
                    best = (float(a), float(b), value)

        a, b = best[0], best[1]

        for _ in range(max_iterations):
            probability = _sigmoid_array(a * logits + b)
            residual = probability - labels

            grad_a = float(np.mean(residual * logits))
            grad_b = float(np.mean(residual))

            weight = probability * (1.0 - probability)
            h_aa = float(np.mean(weight * logits * logits))
            h_ab = float(np.mean(weight * logits))
            h_bb = float(np.mean(weight))

            determinant = h_aa * h_bb - h_ab * h_ab
            if abs(determinant) < 1e-14:
                break

            step_a = (h_bb * grad_a - h_ab * grad_b) / determinant
            step_b = (-h_ab * grad_a + h_aa * grad_b) / determinant

            a -= step_a
            b -= step_b

            if max(abs(step_a), abs(step_b)) < 1e-12:
                break

        self.scale = max(float(a), 1e-6)
        self.bias = float(b)
        self.fitted = True
        self.hit_bound = False

        return self.scale, self.bias

    def fit(
        self,
        probabilities,
        labels,
        search_range: tuple[float, float] = (0.05, 10.0),
        steps: int = 400,
    ) -> float:
        """
        Fit temperature only, leaving the decision boundary at 0.5.

        Kept because it is the correct tool when a model's boundary is already
        right and only its confidence is not. For this project's detector it is
        not - use ``fit_affine``.
        """

        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels).astype(int)

        if probabilities.size == 0:
            raise ValueError("no samples to calibrate on")

        logits = _logit(probabilities)
        candidates = np.geomspace(search_range[0], search_range[1], steps)

        best_temperature = 1.0
        best_loss = float("inf")

        for temperature in candidates:
            scaled = logits / temperature
            loss = float(
                np.mean(
                    np.maximum(scaled, 0)
                    - scaled * labels
                    + np.log1p(np.exp(-np.abs(scaled)))
                )
            )
            if loss < best_loss:
                best_loss = loss
                best_temperature = float(temperature)

        self.scale = 1.0 / best_temperature
        self.bias = 0.0
        self.fitted = True

        # Landing on a search bound means the optimum is outside the range, so
        # this is not a converged fit - it is the grid giving up. It happens
        # when the scores carry almost no information (an undertrained model
        # whose outputs all sit near 0.5 wants T -> infinity). Reporting that
        # as a calibration temperature would hide the real problem.
        lower, upper = search_range
        self.hit_bound = (
            best_temperature <= lower * 1.01 or best_temperature >= upper * 0.99
        )

        if self.hit_bound:
            warnings.warn(
                f"temperature fit hit the search bound (T={best_temperature:.3f} "
                f"in [{lower}, {upper}]). The scores probably carry little "
                f"information - check the model before trusting calibrated "
                f"probabilities.",
                RuntimeWarning,
                stacklevel=2,
            )

        return best_temperature

    # -- diagnostics -------------------------------------------------------

    @staticmethod
    def expected_calibration_error(probabilities, labels, bins: int = 15) -> float:
        """
        ECE: mean gap between confidence and accuracy across equal-width bins.

        0.0 is perfect. Report it before and after fitting - if it does not
        drop, the scores are mis-ranked rather than mis-scaled and no monotone
        calibration will help.
        """

        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels).astype(int)

        edges = np.linspace(0.0, 1.0, bins + 1)
        error = 0.0

        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (probabilities > lower) & (probabilities <= upper)
            if not mask.any():
                continue
            confidence = probabilities[mask].mean()
            accuracy = labels[mask].mean()
            error += mask.mean() * abs(confidence - accuracy)

        return float(error)

    @staticmethod
    def brier_score(probabilities, labels) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels).astype(np.float64)
        return float(np.mean((probabilities - labels) ** 2))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "scale": self.scale,
                    "bias": self.bias,
                    "temperature": self.temperature,
                    "fitted": self.fitted,
                    "hit_search_bound": self.hit_bound,
                    "metadata": self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        # Files written before affine calibration existed carry only a
        # temperature. Reading one must not silently produce scale 1 / bias 0 -
        # fall back to the temperature it does record.
        if "scale" in data:
            calibrator = cls(scale=data["scale"], bias=data.get("bias", 0.0))
        else:
            calibrator = cls(temperature=data.get("temperature", 1.0))

        calibrator.fitted = bool(data.get("fitted", False))
        calibrator.hit_bound = bool(data.get("hit_search_bound", False))
        calibrator.metadata = data.get("metadata", {}) or {}
        return calibrator

    def to_dict(self) -> dict:
        payload = {
            "scale": round(self.scale, 6),
            "bias": round(self.bias, 6),
            "fitted": self.fitted,
            "identity": self.is_identity,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload
