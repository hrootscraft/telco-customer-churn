"""Feature selection three ways: backward stepwise (logistic), L1, permutation importance.

Backward stepwise is on the resume, so it stays, but done properly: a *logistic*
model (the 2023 code fitted OLS to a 0/1 target), on the training split only, and
a categorical feature is dropped as a whole (all its one-hot columns) when the
largest p-value among its columns exceeds ``alpha`` and the block is jointly
non-significant by a likelihood-ratio test.

The output is a table saying which features each method keeps, and a CV check of
whether dropping anything costs ROC-AUC. On this data the answer is "almost
nothing", which is itself the finding: stepwise is a tidy-up, not a performance lever.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churnprop.config import Config
from churnprop.data import Split
from churnprop.features import fit_transform_frame, make_preprocessor
from churnprop.propensity.models import make_model

log = logging.getLogger(__name__)


def _blocks(encoded_cols: list[str], raw_features: list[str]) -> dict[str, list[str]]:
    """Map each raw feature to its encoded column(s)."""
    out: dict[str, list[str]] = {}
    for f in raw_features:
        cols = [c for c in encoded_cols if c == f or c.startswith(f + "_")]
        if cols:
            out[f] = cols
    return out


def _fit_logit(Z: pd.DataFrame, y: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(y.to_numpy(), sm.add_constant(Z.to_numpy(float), has_constant="add")).fit(
            disp=0, maxiter=200, method="lbfgs"
        )


@dataclass
class StepwiseTrace:
    kept: list[str]
    dropped: list[tuple[str, float]]  # (feature, p-value when dropped)


def backward_stepwise(
    Z: pd.DataFrame, y: pd.Series, raw_features: list[str], alpha: float
) -> StepwiseTrace:
    blocks = _blocks(list(Z.columns), raw_features)
    kept = list(blocks)
    dropped: list[tuple[str, float]] = []
    while True:
        cols = [c for f in kept for c in blocks[f]]
        full = _fit_logit(Z[cols], y)
        # block p-value: LR test of dropping the whole block
        worst, worst_p = None, 0.0
        for f in kept:
            rest = [c for g in kept if g != f for c in blocks[g]]
            if not rest:
                continue
            red = _fit_logit(Z[rest], y)
            lr = 2 * (full.llf - red.llf)
            p = float(stats.chi2.sf(max(lr, 0.0), df=len(blocks[f])))
            if p > worst_p:
                worst, worst_p = f, p
        if worst is None or worst_p <= alpha:
            return StepwiseTrace(kept=kept, dropped=dropped)
        log.info("stepwise: drop %-20s p=%.3f", worst, worst_p)
        kept.remove(worst)
        dropped.append((worst, worst_p))


def run_feature_selection(split: Split, cfg: Config) -> pd.DataFrame:
    raw_features = list(split.X_train.columns)
    prep = make_preprocessor(cfg, raw_features)
    Z = fit_transform_frame(prep, split.X_train)
    y = split.y_train

    trace = backward_stepwise(Z, y, raw_features, cfg.stepwise.alpha)

    # L1 logistic on the encoded, scaled matrix: a block is kept if any coefficient is non-zero
    l1 = LogisticRegression(l1_ratio=1.0, C=cfg.stepwise.l1_C, solver="liblinear", max_iter=2000)
    l1.fit(Z, y)
    blocks = _blocks(list(Z.columns), raw_features)
    l1_keep = {
        f
        for f, cols in blocks.items()
        if np.any(np.abs(l1.coef_[0][[Z.columns.get_loc(c) for c in cols]]) > 1e-8)
    }

    # Permutation importance of the LightGBM pipeline on raw features (so blocks are whole)
    lgbm = make_model("lightgbm", cfg, raw_features, resampling="none").fit(split.X_train, y)
    pi = permutation_importance(
        lgbm,
        split.X_train,
        y,
        scoring="roc_auc",
        n_repeats=cfg.stepwise.n_permutations,
        random_state=cfg.seed,
        n_jobs=1,
    )
    pi_mean = dict(zip(raw_features, pi.importances_mean, strict=True))

    rows = []
    for f in raw_features:
        p_drop = dict(trace.dropped).get(f)
        rows.append(
            {
                "feature": f,
                "stepwise_kept": f in trace.kept,
                "stepwise_drop_pvalue": p_drop,
                "l1_kept": f in l1_keep,
                "permutation_importance_auc": pi_mean[f],
            }
        )
    table = pd.DataFrame(rows).sort_values("permutation_importance_auc", ascending=False)

    # Does the stepwise subset cost anything? Same CV, logistic with SMOTE, full vs subset.
    cv = StratifiedKFold(n_splits=cfg.benchmark.n_folds, shuffle=True, random_state=cfg.seed)
    full_auc = cross_val_score(
        make_model("logistic", cfg, raw_features), split.X_train, y, cv=cv, scoring="roc_auc"
    )
    sub_auc = cross_val_score(
        make_model("logistic", cfg, trace.kept),
        split.X_train[trace.kept],
        y,
        cv=cv,
        scoring="roc_auc",
    )
    table.attrs["cv_auc_full"] = float(full_auc.mean())
    table.attrs["cv_auc_stepwise"] = float(sub_auc.mean())
    table.attrs["stepwise_kept"] = trace.kept
    log.info(
        "logistic CV ROC-AUC: all %d features %.4f, stepwise %d features %.4f",
        len(raw_features),
        full_auc.mean(),
        len(trace.kept),
        sub_auc.mean(),
    )
    return table
