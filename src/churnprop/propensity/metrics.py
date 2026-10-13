"""Metrics for an imbalanced binary problem where the *ranking* is what gets used.

Accuracy is reported but not trusted: with 26.5% churn, "nobody churns" scores
73.5%. The metrics marketing can act on are the ranking ones (ROC-AUC, PR-AUC,
lift in the top decile, recall at a precision floor) and the ones that say
whether the probabilities mean what they say (Brier, expected calibration error).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def lift_at_top(y_true: np.ndarray, p: np.ndarray, frac: float = 0.1) -> float:
    """Churn rate among the top ``frac`` scored customers, divided by the base rate."""
    n = max(1, int(round(len(p) * frac)))
    top = np.argsort(-p)[:n]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else float("nan")


def recall_at_precision(y_true: np.ndarray, p: np.ndarray, floor: float = 0.5) -> float:
    """Highest recall achievable while precision stays >= ``floor``."""
    prec, rec, _ = precision_recall_curve(y_true, p)
    ok = prec >= floor
    return float(rec[ok].max()) if ok.any() else 0.0


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append(
                {
                    "bin": b,
                    "p_lo": b / n_bins,
                    "p_hi": (b + 1) / n_bins,
                    "n": int(m.sum()),
                    "mean_predicted": float(p[m].mean()),
                    "observed_rate": float(y_true[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def score_all(
    y_true, p, *, top_frac: float = 0.1, precision_floor: float = 0.5, n_bins: int = 10
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "lift_top_decile": lift_at_top(y_true, p, top_frac),
        "recall_at_precision": recall_at_precision(y_true, p, precision_floor),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
        "ece": expected_calibration_error(y_true, p, n_bins),
    }


METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "lift_top_decile": "Lift @ top decile",
    "recall_at_precision": "Recall @ precision ≥ 0.5",
    "f1": "F1 @ 0.5",
    "accuracy": "Accuracy @ 0.5",
    "brier": "Brier (lower is better)",
    "log_loss": "Log loss (lower is better)",
    "ece": "ECE (lower is better)",
}
HIGHER_IS_BETTER = {k: k not in ("brier", "log_loss", "ece") for k in METRIC_LABELS}
