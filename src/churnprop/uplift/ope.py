"""Off-policy evaluation of a targeting policy from randomised exploration rows.

Each logged row has features, the arm actually given (``a``), the probability it was
given (``p``, uniform 1/K in the exploration holdout), and the reward observed
(``r``, dollars: minus the offer cost, minus the value lost if the customer churned).
A *target* policy is a deterministic arm per row (``pi``). We want E[r under pi].

**IPS** (inverse propensity scoring): average of ``1{a == pi} / p · r``. Unbiased
when ``p > 0`` for every arm the target might pick; variance grows with 1/p.

**DR** (doubly robust): fit an outcome model m(x, k) ≈ E[r | x, k]; estimate
``m(x, pi) + 1{a == pi} / p · (r − m(x, a))``. Unbiased if *either* the propensities
or the outcome model are right, and lower-variance than IPS when the model is decent.
Here the propensities are known exactly (we randomised), so DR's gain is variance.

Both are reported per customer-week; multiply by the active book to get dollars per
week. The simulator's planted truth is reported next to them in the results so a
reader can see the estimators land on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def ips_value(a: np.ndarray, p: np.ndarray, r: np.ndarray, pi: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of the IPS estimate of E[r | policy pi]."""
    w = (a == pi) / p
    est = w * r
    return float(est.mean()), float(est.std(ddof=1) / np.sqrt(len(est)))


def dr_value(
    X: pd.DataFrame,
    a: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
    pi: np.ndarray,
    n_arms: int,
    *,
    seed: int = 0,
    n_folds: int = 3,
) -> tuple[float, float]:
    """Doubly robust estimate with a cross-fitted LightGBM outcome model m(x, k)."""
    n = len(r)
    Xa = X.to_numpy(float)
    m_pi = np.zeros(n)
    m_a = np.zeros(n)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, n)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() < 20:
            continue
        Z_tr = np.column_stack([Xa[tr], np.eye(n_arms)[a[tr]]])
        model = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            verbose=-1,
            random_state=seed,
        )
        model.fit(Z_tr, r[tr])
        m_pi[te] = model.predict(np.column_stack([Xa[te], np.eye(n_arms)[pi[te]]]))
        m_a[te] = model.predict(np.column_stack([Xa[te], np.eye(n_arms)[a[te]]]))
    w = (a == pi) / p
    est = m_pi + w * (r - m_a)
    return float(est.mean()), float(est.std(ddof=1) / np.sqrt(n))
