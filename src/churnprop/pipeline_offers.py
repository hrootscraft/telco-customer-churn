"""Phase 3 end to end: run every policy through the weekly loop, value them three ways.

For each policy in ``configs/offers.yaml``:

* **Realised**: customers still active after the horizon, dollars of value lost to
  churn, dollars spent, all on identical draws, so the differences are the policies'.
* **Planted truth**: the expected weekly reward of the policy's decisions under the
  simulator's hazards (no sampling noise).
* **Off-policy**: IPS and doubly robust estimates of the same quantity from the
  policy's own randomised exploration rows only, which is what a real team would
  have. Shown next to the truth so a reader can see the estimators land on it.

Plus the uplift policy's learning curve, its offer mix over the weeks, how many
offers each policy sent to planted sleeping dogs, and the budget frontier (the
Lagrange multiplier as the knob that keeps a week on budget).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop import plots
from churnprop.config import REPO_ROOT, Config
from churnprop.loop import LoopData, LoopResult, prepare_loop_data, run_loop
from churnprop.offers.menu import OffersConfig, load_offers
from churnprop.uplift.allocate import allocate_budget, lp_allocation
from churnprop.uplift.ope import dr_value, ips_value

log = logging.getLogger(__name__)


def off_policy_estimates(res: LoopResult, data: LoopData) -> dict:
    """IPS / DR value of the policy's intended arms, from its matured exploration rows, in $/week."""
    ex = res.log[res.log["explore"] & res.log["matured"]]
    if len(ex) < 50:
        return {"ope_rows": int(len(ex))}
    a = ex["arm"].to_numpy()
    p = ex["prob"].to_numpy()
    r = ex["reward"].to_numpy()
    pi = ex["intended_arm"].to_numpy()
    zero = np.zeros_like(pi)
    K = len(data.ocfg.arms)
    # features for DR: per row, as of its week
    parts = [data.features(int(w)).iloc[g["row"].to_numpy()] for w, g in ex.groupby("week")]
    X = pd.concat(parts, ignore_index=True)
    order = np.concatenate([g.index.to_numpy() for _, g in ex.groupby("week")])
    # realign a, p, r, pi to X's row order
    pos = {idx: i for i, idx in enumerate(ex.index)}
    sel = np.array([pos[i] for i in order])
    a, p, r, pi, zero = a[sel], p[sel], r[sel], pi[sel], zero[sel]

    ips_pi, ips_se = ips_value(a, p, r, pi)
    ips_0, _ = ips_value(a, p, r, zero)
    dr_pi, dr_se = dr_value(X, a, p, r, pi, K, seed=data.seed)
    dr_0, _ = dr_value(X, a, p, r, zero, K, seed=data.seed)
    # scale per customer-week to dollars per week over the matured weeks' eligible book
    weeks = ex["week"].unique()
    n_elig = res.weekly.set_index("week").loc[weeks, "n_eligible"].mean()
    truth = res.weekly.set_index("week").loc[weeks, "truth_reward_intended"].mean()
    dr_se_week = float(dr_se * n_elig * np.sqrt(2))
    # two-sided 5% test at 80% power detects an effect of about 2.8 standard errors;
    # standard errors shrink with the square root of the number of logged weeks
    mde = 2.8 * dr_se_week
    weeks_needed = len(weeks) * (mde / truth) ** 2 if truth > 0 else float("inf")
    return {
        "ope_rows": int(len(ex)),
        "ope_weeks": int(len(weeks)),
        "ips_per_week": float((ips_pi - ips_0) * n_elig),
        "ips_se_per_week": float(ips_se * n_elig * np.sqrt(2)),
        "dr_per_week": float((dr_pi - dr_0) * n_elig),
        "dr_se_per_week": dr_se_week,
        "truth_per_week_same_weeks": float(truth),
        "mde_per_week_80pct_power": float(mde),
        "weeks_to_detect_truth": float(weeks_needed),
    }


def replicate_realised(
    data: LoopData, policies: tuple[str, ...], n_replicates: int
) -> pd.DataFrame:
    """Re-draw the customers' coin flips ``n_replicates`` times and rerun every policy.

    The planted hazards stay fixed; only the uniforms that decide who actually churns
    change. The spread across replicates is the sampling noise a single run of the
    programme would carry.
    """
    u0 = data.po.u.copy()
    rows = []
    try:
        for rep in range(n_replicates):
            rng = np.random.default_rng(data.seed * 31 + 1000 + rep)
            data.po.u = rng.random(u0.shape)
            res = {pol: run_loop(pol, data) for pol in policies}
            base_active = int(res["no_offers"].final_active.sum())
            base_lost = float(res["no_offers"].weekly["value_lost"].sum())
            for pol, r in res.items():
                spend = float(r.weekly["spend_policy"].sum())
                lost = float(r.weekly["value_lost"].sum())
                rows.append(
                    {
                        "replicate": rep,
                        "policy": pol,
                        "incremental_retained": int(r.final_active.sum()) - base_active,
                        "net_value": base_lost - lost - spend,
                        "spend_policy": spend,
                    }
                )
    finally:
        data.po.u = u0
    return pd.DataFrame(rows)


def sleeping_dog_share(res: LoopResult, data: LoopData, threshold: float = -0.1) -> dict:
    pol = res.log[~res.log["explore"] & (res.log["arm"] > 0)]
    if len(pol) == 0:
        return {"offers": 0, "sleeping_dog_offers": 0, "sleeping_dog_share": float("nan")}
    tau = data.po.tau[pol["row"].to_numpy(), pol["arm"].to_numpy()]
    return {
        "offers": int(len(pol)),
        "sleeping_dog_offers": int((tau < threshold).sum()),
        "sleeping_dog_share": float((tau < threshold).mean()),
        "offers_with_positive_true_uplift_share": float((tau > 0.05).mean()),
    }


def budget_frontier(data: LoopData, budgets: list[float]) -> pd.DataFrame:
    """Expected weekly reward (truth) of oracle allocation at several budgets, on the week-1 book."""
    costs = np.array(data.ocfg.costs)
    eligible = data.active0
    ev = data.po.true_uplift_value() - costs[None, :]
    ev[:, 0] = 0
    ev[~eligible] = -1e12
    h4 = data.po.hazard
    rows = []
    for b in budgets:
        a = allocate_budget(ev, costs, b)
        idx = np.where(eligible)[0]
        truth = float(
            (-(costs[a.arm[idx]]) - (h4[idx, a.arm[idx]] - h4[idx, 0]) * data.value[idx]).sum()
        )
        rows.append(
            {
                "budget": b,
                "spend": a.spend,
                "n_offers": a.n_offers,
                "lambda": a.lam,
                "truth_reward": truth,
                "truth_reward_per_dollar": truth / max(a.spend, 1),
            }
        )
    return pd.DataFrame(rows)


def run_phase3(
    cfg: Config,
    ocfg: OffersConfig | None = None,
    out_dir: Path | str | None = None,
    fig_dir: Path | str | None = None,
    *,
    n_customers: int | None = None,
    n_replicates: int = 5,
) -> dict:
    ocfg = ocfg or load_offers()
    out = Path(out_dir or REPO_ROOT / "results")
    figs = Path(fig_dir or REPO_ROOT / "reports" / "figures")
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_loop_data(cfg, ocfg, n_customers=n_customers)
    results: dict[str, LoopResult] = {}
    for pol in ocfg.policies:
        log.info("running policy %s", pol)
        results[pol] = run_loop(pol, data)

    base = results["no_offers"]
    base_lost = base.weekly["value_lost"].sum()
    base_active = int(base.final_active.sum())
    rows = []
    for pol, r in results.items():
        w = r.weekly
        spend = float(w["spend_policy"].sum())
        lost = float(w["value_lost"].sum())
        inc_value = base_lost - lost
        ope = off_policy_estimates(r, data) if pol != "no_offers" else {}
        dogs = sleeping_dog_share(r, data)
        rows.append(
            {
                "policy": pol,
                "final_active": int(r.final_active.sum()),
                "incremental_retained": int(r.final_active.sum()) - base_active,
                "churned": int(w["churned"].sum()),
                "spend_policy": spend,
                "spend_exploration": float(w["spend"].sum() - spend),
                "value_lost": lost,
                "incremental_value_retained": inc_value,
                "net_value": inc_value - spend,
                "net_per_dollar": (inc_value - spend) / spend if spend > 0 else 0.0,
                "truth_reward_per_week": r.truth_weekly_reward,
                "truth_reward_last_half": float(w["truth_reward"].tail(ocfg.n_weeks // 2).mean()),
                **{f"n_{k}": int(w[f"n_{k}"].sum()) for k in ocfg.arms[1:]},
                **dogs,
                **ope,
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "policy_comparison.csv", index=False)
    weekly = pd.concat([r.weekly for r in results.values()], ignore_index=True)
    weekly.to_csv(out / "policy_weekly.csv", index=False)
    results["uplift"].log.to_parquet(out / "uplift_policy_log.parquet", index=False)

    if n_replicates > 0:
        log.info("replaying realised outcomes %d times", n_replicates)
        reps = replicate_realised(data, tuple(ocfg.policies), n_replicates)
        reps.to_csv(out / "policy_replicates.csv", index=False)
        agg = reps.groupby("policy").agg(
            realised_net_mean=("net_value", "mean"),
            realised_net_sd=("net_value", "std"),
            realised_retained_mean=("incremental_retained", "mean"),
            realised_retained_sd=("incremental_retained", "std"),
        )
        comp = comp.merge(agg, left_on="policy", right_index=True, how="left")
        comp.to_csv(out / "policy_comparison.csv", index=False)

    budgets = [5000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 120000]
    frontier = budget_frontier(data, budgets)
    frontier.to_csv(out / "budget_frontier.csv", index=False)

    # LP check of the allocator on the week-1 oracle problem at the configured budget
    costs = np.array(ocfg.costs)
    ev = data.po.true_uplift_value() - costs[None, :]
    ev[:, 0] = 0
    ev[~data.active0] = -1e12
    sub = np.where(data.active0)[0][:2500]
    alloc = allocate_budget(ev[sub], costs, ocfg.weekly_budget * 0.6)
    lp = lp_allocation(ev[sub], costs, ocfg.weekly_budget * 0.6)
    lp_gap = 1 - alloc.expected_value / lp if lp > 0 else 0.0

    plots.plot_policy_comparison(comp, ocfg.n_weeks, figs)
    plots.plot_learning_curve(weekly, figs)
    plots.plot_offer_mix(results["uplift"].weekly, ocfg.arms, figs)
    plots.plot_budget_frontier(frontier, ocfg.weekly_budget, figs)
    plots.plot_ope_power(comp, figs)

    c = comp.set_index("policy")
    headline = {
        "n_weeks": ocfg.n_weeks,
        "active_at_start": int(data.active0.sum()),
        "weekly_budget": ocfg.weekly_budget,
        "exploration_share": ocfg.exploration_share,
        "baseline_4w_churn_mean": ocfg.baseline_4w_churn_mean,
        "train_on": ocfg.train_on,
        "learner": ocfg.learner,
        "policies": comp.round(4).to_dict(orient="records"),
        "uplift_vs_risk_top_truth": float(
            c.loc["uplift", "truth_reward_per_week"] / c.loc["risk_top", "truth_reward_per_week"]
            - 1
        )
        if c.loc["risk_top", "truth_reward_per_week"]
        else None,
        "uplift_vs_risk_value_last_half": float(
            c.loc["uplift", "truth_reward_last_half"]
            / c.loc["risk_value", "truth_reward_last_half"]
            - 1
        )
        if c.loc["risk_value", "truth_reward_last_half"]
        else None,
        "oracle_truth_per_week": float(c.loc["oracle", "truth_reward_per_week"]),
        "lp_gap": float(lp_gap),
        "n_replicates": n_replicates,
        "sleeping_dog_share": {
            p: float(c.loc[p, "sleeping_dog_share"]) for p in c.index if p != "no_offers"
        },
        "ope_uplift": {
            k: float(c.loc["uplift", k])
            for k in (
                "dr_per_week",
                "dr_se_per_week",
                "mde_per_week_80pct_power",
                "weeks_to_detect_truth",
            )
            if k in c.columns
        },
        "uplift_learning": results["uplift"].uplift_history,
        "fast": cfg.fast,
    }
    with open(out / "headline_phase3.json", "w") as f:
        json.dump(
            headline, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)
        )
    for r in rows:
        log.info(
            "%-16s net $%9.0f  truth $/wk %8.0f  DR $/wk %s",
            r["policy"],
            r["net_value"],
            r["truth_reward_per_week"],
            f"{r['dr_per_week']:.0f}" if "dr_per_week" in r else "-",
        )
    return headline
