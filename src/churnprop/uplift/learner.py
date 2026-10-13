"""A multi-arm uplift learner for "which offer, if any, lowers this customer's churn?"

Two classic meta-learners, both on LightGBM, both trained on the randomised
exploration rows (uniform assignment, so no reweighting is needed; weights are
accepted for the general case):

* **T-learner**: one churn model per arm, ``p_k(x) = P(churn | x, arm k)``. Uplift of
  arm *k* is ``p_0(x) − p_k(x)``. Simple and unbiased per arm; each model sees only
  its arm's rows, so it is data-hungry.
* **S-learner**: one model on ``[x, onehot(arm)]``; uplift is the model's prediction
  with the arm toggled. Shares strength across arms, can shrink the effect toward
  zero when the arm indicator is weak.

tierlift benchmarks six learners (S/T/X/R/DR, causal forest) against planted truth
on Qini and AUUC; this repository picks one and runs it weekly. The choice is a
config switch so the comparison can be rerun here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def _lgbm(seed: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        verbose=-1,
        random_state=seed,
    )


class MultiArmUplift:
    def __init__(self, n_arms: int, kind: str = "t", seed: int = 0, n_estimators: int = 200):
        if kind not in ("t", "s"):
            raise ValueError("kind must be 't' or 's'")
        self.n_arms = n_arms
        self.kind = kind
        self.seed = seed
        self.n_estimators = n_estimators
        self.models_: dict = {}
        self.columns_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        arm: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        self.columns_ = list(X.columns)
        Xa = X.to_numpy(float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.kind == "t":
            for k in range(self.n_arms):
                m = arm == k
                if m.sum() < 30 or len(np.unique(y[m])) < 2:
                    self.models_[k] = None
                    continue
                self.models_[k] = _lgbm(self.seed + k, self.n_estimators).fit(
                    Xa[m], y[m], sample_weight=w[m]
                )
        else:
            Z = np.column_stack([Xa, np.eye(self.n_arms)[arm]])
            self.models_["s"] = _lgbm(self.seed, self.n_estimators).fit(Z, y, sample_weight=w)
        return self

    def predict_churn(self, X: pd.DataFrame) -> np.ndarray:
        """n × K predicted churn probability under each arm."""
        Xa = X[self.columns_].to_numpy(float)
        out = np.zeros((len(Xa), self.n_arms))
        if self.kind == "t":
            base = None
            for k in range(self.n_arms):
                m = self.models_.get(k)
                if m is None:
                    out[:, k] = np.nan
                else:
                    out[:, k] = m.predict_proba(Xa)[:, 1]
                    base = out[:, k] if base is None else base
            # an arm without a model inherits control (uplift 0)
            for k in range(self.n_arms):
                if np.isnan(out[:, k]).all():
                    out[:, k] = (
                        out[:, 0]
                        if not np.isnan(out[:, 0]).all()
                        else (base if base is not None else 0.0)
                    )
        else:
            s = self.models_["s"]
            for k in range(self.n_arms):
                Z = np.column_stack([Xa, np.tile(np.eye(self.n_arms)[k], (len(Xa), 1))])
                out[:, k] = s.predict_proba(Z)[:, 1]
        return out

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """n × K reduction in churn probability vs control (column 0 is zero)."""
        p = self.predict_churn(X)
        return p[:, [0]] - p
