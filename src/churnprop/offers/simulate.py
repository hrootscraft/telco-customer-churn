"""Plant each customer's response to each offer, and draw coherent potential outcomes.

For customer *i* and arm *k* the planted quantity is a relative change in the 4-week
churn hazard, ``tau[i, k]`` (0 for control). The baseline hazard ``q[i]`` comes from
the customer's *out-of-fold* phase-1 propensity score, rescaled so the book's mean
4-week churn is ``baseline_4w_churn_mean`` (D13). The hazard under arm *k* is

    h[i, k] = clip(q[i] * (1 - tau[i, k]), 0, 0.95)

Potential outcomes are drawn with **one uniform per customer per week**, shared
across arms: customer *i* churns under arm *k* in week *w* iff ``u[i, w] < h[i, k]``.
That makes the arms' outcomes coherent (an offer that lowers the hazard can only
turn a churn into a stay, never the reverse) and makes every policy comparable on
identical draws.

The simulator holds the truth; the policies never see ``tau`` or ``h``. Only the
``oracle`` policy is handed ``tau``, as the ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from churnprop.offers.menu import OffersConfig


def planted_effects(X: pd.DataFrame, ocfg: OffersConfig, rng: np.random.Generator) -> pd.DataFrame:
    """``tau`` as a DataFrame (customers × arms); column 0 is control (all zeros).

    ``X`` must carry the snapshot columns and any event features the rules reference.
    Rules are evaluated with ``DataFrame.eval``; every matching rule adds its effect.
    """
    n = len(X)
    cols = {"control": np.zeros(n)}
    for o in ocfg.offers:
        t = np.full(n, o.base)
        for r in o.rules:
            m = X.eval(r.when).to_numpy(bool)
            t = t + r.add * m
        t = t + rng.normal(0, ocfg.heterogeneity_sd, n)
        cols[o.name] = np.clip(t, *ocfg.heterogeneity_clip)
    return pd.DataFrame(cols, index=X.index)


def baseline_hazard(p_oof: np.ndarray, ocfg: OffersConfig) -> np.ndarray:
    """Scale out-of-fold churn scores to a 4-week hazard with the configured book mean."""
    p = np.clip(np.asarray(p_oof, float), 1e-4, 1 - 1e-4)
    q = p * (ocfg.baseline_4w_churn_mean / p.mean())
    return np.clip(q, 1e-4, 0.95)


@dataclass
class PotentialOutcomes:
    customer_id: np.ndarray
    arms: tuple[str, ...]
    tau: np.ndarray  # n × K relative hazard reductions, control column = 0
    q: np.ndarray  # n baseline 4-week hazards
    value: np.ndarray  # n dollars at stake if the customer churns
    u: np.ndarray  # n × n_weeks shared uniforms

    @property
    def hazard(self) -> np.ndarray:
        """n × K hazards under each arm."""
        return np.clip(self.q[:, None] * (1 - self.tau), 0.0, 0.95)

    def churn_if(self, week: int, arm_idx: np.ndarray) -> np.ndarray:
        """Realised churn in ``week`` for each customer under the arm they were given."""
        h = self.hazard[np.arange(len(arm_idx)), arm_idx]
        return self.u[:, week] < h

    def expected_reward(self, arm_idx: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """True expected weekly reward (−cost − hazard × value) under the given arms."""
        h = self.hazard[np.arange(len(arm_idx)), arm_idx]
        return -costs[arm_idx] - h * self.value

    def true_uplift_value(self) -> np.ndarray:
        """n × K: expected dollars saved vs control by each arm, before cost."""
        h = self.hazard
        return (h[:, [0]] - h) * self.value[:, None]


def plant_responses(
    X: pd.DataFrame,
    customer_id: np.ndarray,
    p_oof: np.ndarray,
    value: np.ndarray,
    ocfg: OffersConfig,
    seed: int,
) -> PotentialOutcomes:
    rng = np.random.default_rng(seed)
    tau = planted_effects(X, ocfg, rng).to_numpy(float)
    q = baseline_hazard(p_oof, ocfg)
    u = rng.random((len(X), ocfg.n_weeks + ocfg.outcome_weeks))
    return PotentialOutcomes(
        customer_id=np.asarray(customer_id),
        arms=ocfg.arms,
        tau=tau,
        q=q,
        value=np.asarray(value, float) * ocfg.value_margin,
        u=u,
    )
