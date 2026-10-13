"""Pick at most one offer per customer under a weekly budget.

Problem: maximise Σ_i Σ_k x_ik · ev_ik subject to Σ_i Σ_k x_ik · c_k ≤ B and
Σ_k x_ik ≤ 1, x_ik ∈ {0, 1}, where ``ev_ik`` is the *net* expected value of giving
offer *k* to customer *i* (expected dollars saved minus cost).

:func:`allocate_budget` solves the Lagrangian relaxation: for a multiplier λ ≥ 0 on
cost, each customer independently takes ``argmax_k (ev_ik − λ·c_k)`` if that is
positive, else nothing. Spend is monotone non-increasing in λ, so a bisection finds
the smallest λ whose spend fits the budget; at λ = 0 the budget is slack and the
greedy choice is already optimal. This is the "one knob to stay on budget" an
operations team actually turns (D16). :func:`lp_allocation` solves the LP
relaxation with ``scipy.optimize.linprog`` to show the gap is tiny.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class Allocation:
    arm: np.ndarray  # chosen arm index per customer, 0 = control / nothing
    spend: float
    expected_value: float  # Σ ev of the chosen arms (as estimated, not the truth)
    lam: float
    n_offers: int


def _choose(ev: np.ndarray, costs: np.ndarray, lam: float) -> np.ndarray:
    score = ev - lam * costs[None, :]
    score[:, 0] = 0.0  # control: nothing spent, nothing gained
    return np.argmax(score, axis=1)


def allocate_budget(
    ev: np.ndarray, costs: np.ndarray, budget: float, *, tol: float = 1e-6
) -> Allocation:
    """``ev`` is n × K net expected values with column 0 = control (ignored, treated as 0)."""
    ev = np.asarray(ev, float)
    costs = np.asarray(costs, float)
    arm = _choose(ev, costs, 0.0)
    spend = float(costs[arm].sum())
    lam = 0.0
    if spend > budget + tol:
        lo, hi = 0.0, 1.0
        while costs[_choose(ev, costs, hi)].sum() > budget:
            hi *= 2
            if hi > 1e9:
                break
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if costs[_choose(ev, costs, mid)].sum() > budget:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        lam = hi
        arm = _choose(ev, costs, lam)
        spend = float(costs[arm].sum())
        # bisection can leave a little slack because spend jumps in discrete steps;
        # fill it greedily by the best remaining ratio
        slack = budget - spend
        if slack > costs[1:].min():
            idx = np.where(arm == 0)[0]
            best_k = np.argmax(ev[idx, 1:], axis=1) + 1
            gain = ev[idx, best_k]
            order = idx[np.argsort(-gain / costs[best_k])]
            for i in order:
                k = int(np.argmax(ev[i, 1:]) + 1)
                if ev[i, k] <= 0:
                    break
                if costs[k] <= slack:
                    arm[i] = k
                    slack -= costs[k]
            spend = float(costs[arm].sum())
    value = float(ev[np.arange(len(arm)), arm][arm > 0].sum())
    return Allocation(
        arm=arm, spend=spend, expected_value=value, lam=lam, n_offers=int((arm > 0).sum())
    )


def lp_allocation(ev: np.ndarray, costs: np.ndarray, budget: float) -> float:
    """Optimal value of the LP relaxation (fractional offers); an upper bound for the integer problem."""
    ev = np.asarray(ev, float)
    n, K = ev.shape
    k = K - 1
    c = -ev[:, 1:].ravel()  # maximise
    # budget row
    A = [np.tile(costs[1:], n)]
    b = [budget]
    # one offer per customer
    A_rows = np.zeros((n, n * k))
    for i in range(n):
        A_rows[i, i * k : (i + 1) * k] = 1
    res = linprog(
        c,
        A_ub=np.vstack([A, A_rows]),
        b_ub=np.concatenate([b, np.ones(n)]),
        bounds=(0, 1),
        method="highs",
    )
    return float(-res.fun)
