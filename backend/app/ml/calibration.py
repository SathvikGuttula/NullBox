"""
Probability calibration.

A neural classifier's softmax output is a score, not a probability. Modern
networks are badly overconfident: a model can be 99% sure and wrong 20% of the
time. That is tolerable when all you do is rank, and unacceptable the moment a
number is shown to an operator or fed into a risk score that mixes it with
other signals - the miscalibrated branch silently dominates the fusion.

Temperature scaling is the standard fix: divide the logits by a single scalar
T fitted on held-out data. It cannot change the ranking, so EER and ROC-AUC
are untouched, but it makes the number mean what it says.

The previous implementation stored a temperature and then returned its input
unchanged, so nothing was ever calibrated. That is fixed here.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np


class ProbabilityCalibrator:
    """
    Single-parameter temperature scaling over a two-class problem.

        calibrator = ProbabilityCalibrator()
        calibrator.fit(validation_logits, validation_labels)
        p = calibrator.calibrate(0.93)

    ``temperature`` of 1.0 is the identity, so an unfitted calibrator is a
    no-op and the pipeline still runs before calibration data exists.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(float(temperature), 1e-6)
        self.fitted = False
        self.hit_bound = False

    # -- application -------------------------------------------------------

    def calibrate(self, probability: float) -> float:
        """
        Apply the temperature to a single spoof probability.

        The probability is converted back to a logit, divided by T, and pushed
        through the sigmoid again - which is exactly what dividing the
        two-class logit difference by T does.
        """

        probability = min(max(float(probability), 0.0), 1.0)

        if self.temperature == 1.0:
            return probability

        epsilon = 1e-7
        clipped = min(max(probability, epsilon), 1.0 - epsilon)

        logit = math.log(clipped / (1.0 - clipped))

        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    def calibrate_array(self, probabilities) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if self.temperature == 1.0:
            return probabilities

        epsilon = 1e-7
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        logits = np.log(clipped / (1.0 - clipped))

        return 1.0 / (1.0 + np.exp(-logits / self.temperature))

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        probabilities,
        labels,
        search_range: tuple[float, float] = (0.05, 10.0),
        steps: int = 400,
    ) -> float:
        """
        Fit T by minimising negative log-likelihood on held-out data.

        A grid search over log-spaced T is used instead of gradient descent:
        the objective is one-dimensional and convex in practice, 400 points
        resolve T to well under a percent, and it takes milliseconds - so
        there is no reason to carry an optimiser and its failure modes.

        Fit this on the VALIDATION split, never on training data, and never on
        the same split you then report calibrated numbers from.
        """

        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels).astype(int)

        if probabilities.size == 0:
            raise ValueError("no samples to calibrate on")

        epsilon = 1e-7
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        logits = np.log(clipped / (1.0 - clipped))

        candidates = np.geomspace(search_range[0], search_range[1], steps)

        best_temperature = 1.0
        best_loss = float("inf")

        for temperature in candidates:
            scaled = logits / temperature
            # Numerically stable binary cross-entropy from logits.
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

        self.temperature = best_temperature
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
        drop, temperature scaling was not the right tool and the scores are
        probably mis-ranked rather than mis-scaled.
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
            error += (mask.mean()) * abs(confidence - accuracy)

        return float(error)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "temperature": self.temperature,
                    "fitted": self.fitted,
                    "hit_search_bound": self.hit_bound,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calibrator = cls(temperature=data.get("temperature", 1.0))
        calibrator.fitted = bool(data.get("fitted", False))
        calibrator.hit_bound = bool(data.get("hit_search_bound", False))
        return calibrator
