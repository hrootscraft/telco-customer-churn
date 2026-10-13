"""The weekly retention loop: score, allocate under budget, observe, log, retrain.

Every policy runs on the *same* book, the same planted responses, the same weekly
draws and the same randomised exploration holdout, so differences between policies
are differences in decisions, not in luck.

One week, for one policy:

1. **Eligible** = active customers not inside the cooldown of a previous offer.
2. **Exploration holdout**: a fixed random ``exploration_share`` of the eligible
   customers get a uniformly random arm (control or one of the offers) with known
   probability 1/K. Identical across policies. This is the data an uplift model can
   be trained on and any policy can be valued on (D15).
3. **Policy**: for the remaining eligible customers the policy picks at most one
   offer each, within the weekly budget net of what exploration spent.
4. **Outcomes**: an offer changes the customer's hazard for the next
   ``outcome_weeks`` weeks. Each active customer churns this week iff their shared
   uniform is below the weekly hazard of the arm in force. The offer's logged
   outcome ("churned within the window") matures ``outcome_weeks`` later.
5. **Log** every eligible customer-week: arm, assignment probability, whether it
   was exploration, the policy's own intended arm (for off-policy evaluation), and
   the matured outcome and reward when they arrive.

The uplift policy retrains each week on the matured exploration rows so far and
falls back to its warm-start policy until it has ``min_training_rows``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churnprop.config import Config
from churnprop.data import load_clean
from churnprop.features import add_engineered, fit_transform_frame, make_preprocessor
from churnprop.features_sql.build import build_event_features
from churnprop.offers.menu import OffersConfig
from churnprop.offers.simulate import PotentialOutcomes, plant_responses
from churnprop.pipeline_events import EVENT_FEATURE_COLS, EVENTS_ROOT, ensure_events
from churnprop.propensity.models import make_model
from churnprop.uplift.allocate import allocate_budget
from churnprop.uplift.learner import MultiArmUplift

log = logging.getLogger(__name__)

POLICIES = ("no_offers", "random_outreach", "risk_top", "risk_value", "uplift", "oracle")


# ----------------------------------------------------------------------------- data


@dataclass
class LoopData:
    """Everything the loop needs that does not depend on the policy."""

    customer_id: np.ndarray
    raw: pd.DataFrame  # snapshot columns + engineered, for the rules and reporting
    p_oof: np.ndarray  # out-of-fold phase-1 churn score
    value: np.ndarray  # dollars at stake (CLTV × margin)
    active0: np.ndarray  # active at the snapshot (did not churn)
    Z_snapshot: pd.DataFrame  # encoded snapshot features (one row per customer)
    event_features: dict[
        int, pd.DataFrame
    ]  # week -> event features as of that week (customer order)
    week_dates: dict[int, pd.Timestamp]
    po: PotentialOutcomes
    ocfg: OffersConfig
    seed: int

    def features(self, week: int) -> pd.DataFrame:
        """Model features for ``week``: encoded snapshot + event features as of the week + p_oof."""
        ev = self.event_features[week][EVENT_FEATURE_COLS].reset_index(drop=True)
        out = pd.concat([self.Z_snapshot.reset_index(drop=True), ev], axis=1)
        out["p_oof"] = self.p_oof
        return out


def prepare_loop_data(
    cfg: Config, ocfg: OffersConfig, *, seed: int | None = None, n_customers: int | None = None
) -> LoopData:
    seed = cfg.seed if seed is None else seed
    df = add_engineered(load_clean(cfg))
    if n_customers is not None:
        df = df.sample(n_customers, random_state=seed).reset_index(drop=True)
    ids = df[cfg.data.id_col].to_numpy()
    y = df[cfg.data.target].to_numpy()
    X = df[list(cfg.data.feature_cols)]

    # Out-of-fold propensity: every customer scored by a model that never saw them.
    model = make_model("lightgbm", cfg, list(X.columns), resampling="none")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    prep = make_preprocessor(cfg, list(X.columns))
    Z = fit_transform_frame(prep, X)

    ev = ensure_events(cfg)
    weeks = range(0, ocfg.n_weeks + 1)
    week_dates = {w: ev.snapshot_date + pd.Timedelta(days=7 * w) for w in weeks}
    event_features = {
        w: build_event_features(week_dates[w], events_root=EVENTS_ROOT, customer_ids=ids)
        for w in weeks
    }

    rules_frame = pd.concat(
        [df.reset_index(drop=True), event_features[0][EVENT_FEATURE_COLS].reset_index(drop=True)],
        axis=1,
    )
    value = df[ocfg.value_column].to_numpy(float)
    po = plant_responses(rules_frame, ids, p_oof, value, ocfg, seed=seed)
    return LoopData(
        customer_id=ids,
        raw=df,
        p_oof=p_oof,
        value=value * ocfg.value_margin,
        active0=(y == 0),
        Z_snapshot=Z,
        event_features=event_features,
        week_dates=week_dates,
        po=po,
        ocfg=ocfg,
        seed=seed,
    )


# ----------------------------------------------------------------------------- policies


def _spend_rank(
    score: np.ndarray, eligible: np.ndarray, cost: float, budget: float, arm_k: int, n: int
) -> np.ndarray:
    """Give arm ``arm_k`` to eligible customers in descending ``score`` until the budget runs out."""
    arm = np.zeros(n, dtype=int)
    n_afford = int(budget // cost)
    idx = np.where(eligible)[0]
    order = idx[np.argsort(-score[idx])][:n_afford]
    arm[order] = arm_k
    return arm


class UpliftState:
    def __init__(self, ocfg: OffersConfig, seed: int):
        self.ocfg = ocfg
        self.seed = seed
        self.model: MultiArmUplift | None = None
        self.n_train = 0
        self.history: list[dict] = []


def policy_arms(
    name: str,
    week: int,
    data: LoopData,
    eligible: np.ndarray,
    budget: float,
    log_df: pd.DataFrame | None,
    state: UpliftState | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Return the policy's intended arm for every customer (0 where not eligible) and diagnostics."""
    n = len(data.customer_id)
    costs = np.array(data.ocfg.costs)
    K = len(costs)
    info: dict = {"policy_used": name}
    if name == "no_offers":
        return np.zeros(n, dtype=int), info
    if name == "random_outreach":
        return _spend_rank(rng.random(n), eligible, costs[1], budget, 1, n), info
    if name == "risk_top":
        return _spend_rank(data.p_oof, eligible, costs[1], budget, 1, n), info
    if name == "risk_value":
        s = data.ocfg.assumed_save_rate
        ev = data.p_oof[:, None] * s * data.value[:, None] - costs[None, :]
        ev[:, 0] = 0
        ev[~eligible] = -np.inf
        alloc = allocate_budget(np.where(np.isfinite(ev), ev, -1e12), costs, budget)
        info["lambda"] = alloc.lam
        return alloc.arm, info
    if name == "oracle":
        ev = data.po.true_uplift_value() - costs[None, :]
        ev[:, 0] = 0
        ev[~eligible] = -1e12
        alloc = allocate_budget(ev, costs, budget)
        info["lambda"] = alloc.lam
        return alloc.arm, info
    if name == "uplift":
        assert state is not None
        matured = None
        if log_df is not None and len(log_df):
            m = log_df["matured"]
            if data.ocfg.train_on == "exploration":
                m = m & log_df["explore"]
            matured = log_df[m]
        if matured is None or len(matured) < data.ocfg.min_training_rows:
            arm, sub = policy_arms(
                data.ocfg.warm_start_policy, week, data, eligible, budget, log_df, None, rng
            )
            info["policy_used"] = f"warm_start:{data.ocfg.warm_start_policy}"
            info["n_train"] = 0 if matured is None else int(len(matured))
            return arm, info
        # train on matured exploration rows, each with the features as of its own week
        parts = []
        for w, g in matured.groupby("week"):
            Xw = data.features(int(w)).iloc[g["row"].to_numpy()]
            parts.append(
                Xw.assign(_arm=g["arm"].to_numpy(), _y=g["churned_window"].to_numpy().astype(int))
            )
        train = pd.concat(parts, ignore_index=True)
        model = MultiArmUplift(
            K, kind=data.ocfg.learner, seed=data.seed + week, n_estimators=data.ocfg.n_estimators
        )
        model.fit(
            train.drop(columns=["_arm", "_y"]), train["_arm"].to_numpy(), train["_y"].to_numpy()
        )
        state.model, state.n_train = model, len(train)
        uplift = model.predict_uplift(data.features(week))  # reduction in 4-week churn prob
        ev = uplift * data.value[:, None] - costs[None, :]
        ev[:, 0] = 0
        ev[~eligible] = -1e12
        alloc = allocate_budget(ev, costs, budget)
        info.update(lambda_=alloc.lam, n_train=len(train))
        # realised-vs-predicted uplift diagnostic for monitoring (phase 4)
        state.history.append(
            {
                "week": week,
                "n_train": len(train),
                "mean_pred_uplift": float(np.nanmean(uplift[eligible, 1:])),
            }
        )
        return alloc.arm, info
    raise KeyError(name)


# ----------------------------------------------------------------------------- the loop


@dataclass
class LoopResult:
    policy: str
    weekly: pd.DataFrame
    log: pd.DataFrame
    final_active: np.ndarray
    truth_weekly_reward: float  # mean expected reward per week vs control, from planted hazards
    uplift_history: list[dict] = field(default_factory=list)


def run_loop(policy: str, data: LoopData) -> LoopResult:
    ocfg = data.ocfg
    n = len(data.customer_id)
    K = len(ocfg.arms)
    costs = np.array(ocfg.costs)
    po = data.po
    weekly_h = 1 - (1 - po.hazard) ** (1 / ocfg.outcome_weeks)  # n × K weekly hazards

    active = data.active0.copy()
    last_offer_week = np.full(n, -(10**6))
    arm_in_force = np.zeros(n, dtype=int)
    rows: list[dict] = []
    weekly: list[dict] = []
    state = UpliftState(ocfg, data.seed)
    truth_rewards = []
    log_df: pd.DataFrame | None = None

    for w in range(1, ocfg.n_weeks + 1):
        rng_explore = np.random.default_rng(data.seed * 7919 + w)  # identical across policies
        rng_policy = np.random.default_rng(data.seed * 104729 + w)
        eligible = active & ((w - last_offer_week) >= ocfg.cooldown_weeks)
        explore = eligible & (rng_explore.random(n) < ocfg.exploration_share)
        explore_arm = rng_explore.integers(0, K, n)
        explore_spend = float(costs[explore_arm[explore]].sum())

        # mature outcomes for rows whose window closed before this week
        if rows:
            log_df = pd.DataFrame(rows)

        budget = max(0.0, ocfg.weekly_budget - explore_spend)
        pol_eligible = eligible & ~explore
        pi_arm, info = policy_arms(policy, w, data, eligible, budget, log_df, state, rng_policy)
        intended = (
            pi_arm.copy()
        )  # the policy's choice for every eligible customer, explore ones included
        assign = np.where(explore, explore_arm, np.where(pol_eligible, pi_arm, 0))
        prob = np.where(explore, 1.0 / K, 1.0)

        # offers given this week take force now
        got_offer = eligible & (assign > 0)
        arm_in_force = np.where(eligible, assign, arm_in_force)
        last_offer_week = np.where(got_offer, w, last_offer_week)
        # an earlier offer stays in force only within its window
        expired = (w - last_offer_week) >= ocfg.outcome_weeks
        arm_in_force = np.where(expired & ~eligible, 0, arm_in_force)

        churn_now = active & (po.u[:, w - 1] < weekly_h[np.arange(n), arm_in_force])
        # expected reward of this week's *policy* decisions vs control, from the planted truth
        # (exploration rows are excluded: identical for every policy)
        h4 = po.hazard
        elig_idx = np.where(eligible)[0]
        pol_idx = np.where(pol_eligible)[0]
        truth_week = float(
            (
                -(costs[assign[pol_idx]])
                - (h4[pol_idx, assign[pol_idx]] - h4[pol_idx, 0]) * data.value[pol_idx]
            ).sum()
        )
        truth_rewards.append(truth_week)
        # the same quantity for the policy's *intended* arms over every eligible customer,
        # exploration rows included: this is what off-policy evaluation estimates
        truth_intended = float(
            (
                -(costs[intended[elig_idx]])
                - (h4[elig_idx, intended[elig_idx]] - h4[elig_idx, 0]) * data.value[elig_idx]
            ).sum()
        )

        for i in elig_idx:
            rows.append(
                {
                    "week": w,
                    "row": int(i),
                    "customer_id": data.customer_id[i],
                    "arm": int(assign[i]),
                    "prob": float(prob[i]),
                    "explore": bool(explore[i]),
                    "intended_arm": int(intended[i]),
                    "p_oof": float(data.p_oof[i]),
                    "value": float(data.value[i]),
                    "churned_window": False,
                    "matured": False,
                }
            )
        active &= ~churn_now
        # update window outcomes: any row whose window includes this week and whose customer churned now
        churned_ids = set(np.where(churn_now)[0])
        for r in rows:
            if w - r["week"] < ocfg.outcome_weeks and r["row"] in churned_ids:
                r["churned_window"] = True
            if w - r["week"] == ocfg.outcome_weeks - 1:
                r["matured"] = True
        weekly.append(
            {
                "policy": policy,
                "week": w,
                "date": data.week_dates[w],
                "n_active_start": int((active | churn_now).sum()),
                "n_eligible": int(eligible.sum()),
                "n_explore": int(explore.sum()),
                **{f"n_{ocfg.arms[k]}": int((assign[eligible] == k).sum()) for k in range(1, K)},
                "spend": float(costs[assign[eligible]].sum()),
                "spend_policy": float(costs[assign[pol_eligible]].sum()),
                "churned": int(churn_now.sum()),
                "value_lost": float(data.value[churn_now].sum()),
                "n_active_end": int(active.sum()),
                "policy_used": info.get("policy_used", policy),
                "n_train": info.get("n_train", np.nan),
                "lambda": info.get("lambda", info.get("lambda_", np.nan)),
                "truth_reward": truth_week,
                "truth_reward_intended": truth_intended,
            }
        )
        log.info(
            "%-16s week %2d: eligible %4d, offers %3d, spend %6.0f, churned %3d, active %4d (%s)",
            policy,
            w,
            eligible.sum(),
            int((assign[eligible] > 0).sum()),
            costs[assign[eligible]].sum(),
            churn_now.sum(),
            active.sum(),
            info.get("policy_used", policy),
        )

    log_df = pd.DataFrame(rows)
    log_df["reward"] = (
        -costs[log_df["arm"].to_numpy()]
        - log_df["churned_window"].to_numpy() * log_df["value"].to_numpy()
    )
    return LoopResult(
        policy=policy,
        weekly=pd.DataFrame(weekly),
        log=log_df,
        final_active=active,
        truth_weekly_reward=float(np.mean(truth_rewards)),
        uplift_history=state.history,
    )
