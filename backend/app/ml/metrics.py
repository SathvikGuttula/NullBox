from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(
    labels: list[int],
    probabilities: list[float],
) -> dict:

    labels_np = np.asarray(
        labels
    )

    probabilities_np = np.asarray(
        probabilities
    )

    predictions = (
        probabilities_np >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                labels_np,
                predictions,
            )
        ),

        "f1": float(
            f1_score(
                labels_np,
                predictions,
                zero_division=0,
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels_np,
                predictions,
            ).tolist(),
    }

    if len(
        np.unique(labels_np)
    ) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                labels_np,
                probabilities_np,
            )
        )

    else:

        metrics["roc_auc"] = None

    return metrics