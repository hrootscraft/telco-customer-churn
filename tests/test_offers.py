"""Phase 3: offer menu, planted responses, budget allocation, OPE, uplift learner, loop."""

import numpy as np
import pandas as pd
import pytest

from churnprop.loop import POLICIES, prepare_loop_data, run_loop
from churnprop.offers.menu import load_offers
from churnprop.offers.simulate import planted_effects
from churnprop.uplift.allocate import allocate_budget, lp_allocation
from churnprop.uplift.learner import MultiArmUplift
from churnprop.uplift.ope import dr_value, ips_value


@pytest.fixture(scope="module")
def ocfg():
    return load_offers()


# ----------------------------------------------------------------------------- menu + planting


def test_menu_arms_and_costs(ocfg):
    assert ocfg.arms[0] == "control"
    assert ocfg.costs[0] == 0.0
    assert len(ocfg.arms) == len(ocfg.offers) + 1
    assert all(o.cost > 0 for o in ocfg.offers)


def test_rules_plant_the_intended_direction(ocfg):
    base = {
        "Monthly Charges": 50.0,
        "Payment Method": "Mailed check",
        "Tenure Months": 30,
        "Contract": "Month-to-month",
        "Internet Service": "DSL",
        "Tech Support": "Yes",
        "Online Security": "Yes",
        "tickets_90d": 0,
    }
    X = pd.DataFrame(
        [
            base,
            {**base, "Monthly Charges": 95.0},  # price sensitive: bill credit works better
            {**base, "Tenure Months": 2},  # new customer: bill credit backfires
            {**base, "Contract": "Two year"},  # locked in: sleeping dog for the credit
            {**base, "Internet Service": "Fiber optic", "Tech Support": "No"},  # bundle target
        ]
    )
    quiet = load_offers(heterogeneity={"sd": 0.0})
    tau = planted_effects(X, quiet, np.random.default_rng(0))
    bc, tb = tau["bill_credit"].to_numpy(), tau["tech_bundle"].to_numpy()
    assert (tau["control"] == 0).all()
    assert bc[1] > bc[0] > bc[2]
    assert bc[2] < 0 and bc[3] < 0
    assert tb[4] > tb[0]


# ----------------------------------------------------------------------------- allocation


def test_allocation_respects_budget_and_one_offer_each():
    rng = np.random.default_rng(0)
    n, costs = 2000, np.array([0.0, 45.0, 60.0, 70.0])
    ev = np.column_stack([np.zeros(n), rng.normal(20, 60, (n, 3))])
    budget = 15_000
    a = allocate_budget(ev, costs, budget)
    assert a.spend <= budget + 1e-6
    assert a.arm.shape == (n,)
    # no customer is given an arm with negative expected value
    assert (ev[np.arange(n), a.arm] >= 0).all()
    # with the budget binding, the multiplier is positive and the budget is nearly spent
    assert a.lam > 0 and a.spend > 0.97 * budget


def test_allocation_slack_budget_takes_every_positive_offer():
    rng = np.random.default_rng(1)
    n, costs = 500, np.array([0.0, 45.0, 60.0])
    ev = np.column_stack([np.zeros(n), rng.normal(10, 40, (n, 2))])
    a = allocate_budget(ev, costs, 1e9)
    best = ev.argmax(axis=1)
    assert a.lam == 0
    assert (a.arm == np.where(ev.max(axis=1) > 0, best, 0)).all()


def test_allocation_matches_lp_bound():
    rng = np.random.default_rng(2)
    n, costs = 1500, np.array([0.0, 45.0, 60.0, 70.0])
    ev = np.column_stack([np.zeros(n), rng.normal(20, 60, (n, 3))])
    budget = 12_000
    a = allocate_budget(ev, costs, budget)
    lp = lp_allocation(ev, costs, budget)
    assert a.expected_value <= lp + 1e-6
    assert a.expected_value >= 0.99 * lp


# ----------------------------------------------------------------------------- OPE


def _synthetic_log(n=20_000, K=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    a = rng.integers(0, K, n)
    p = np.full(n, 1 / K)
    mean = np.column_stack([X["x1"], X["x1"] + 2 * X["x2"], 1 - X["x2"]])  # n × K true E[r|x,k]
    r = mean[np.arange(n), a] + rng.normal(0, 1, n)
    pi = np.where(X["x2"] > 0, 1, 2)  # a target policy that depends on x
    truth = mean[np.arange(n), pi].mean()
    return X, a, p, r, pi, truth


def test_ips_and_dr_recover_the_truth():
    X, a, p, r, pi, truth = _synthetic_log()
    ips, ips_se = ips_value(a, p, r, pi)
    dr, dr_se = dr_value(X, a, p, r, pi, 3, seed=0)
    assert abs(ips - truth) < 3 * ips_se
    assert abs(dr - truth) < 3 * dr_se
    assert dr_se < ips_se  # the outcome model buys variance


def test_ips_uses_only_matching_rows():
    a = np.array([0, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    r = np.array([1.0, 2.0, 4.0, 3.0])
    v, _ = ips_value(a, p, r, np.array([1, 1, 1, 1]))
    assert v == pytest.approx((2 + 4) / 0.5 / 4)


# ----------------------------------------------------------------------------- learner


@pytest.mark.parametrize("kind", ["t", "s"])
def test_learner_recovers_planted_heterogeneity(kind):
    rng = np.random.default_rng(3)
    n = 6000
    X = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n)})
    arm = rng.integers(0, 2, n)
    p0 = 0.3
    tau = np.where(X["x"] > 0, 0.15, -0.10)  # offer helps when x > 0, hurts otherwise
    p1 = p0 - tau
    prob = np.where(arm == 1, p1, p0)
    y = (rng.random(n) < prob).astype(int)
    m = MultiArmUplift(2, kind=kind, seed=0, n_estimators=150).fit(X, arm, y)
    up = m.predict_uplift(X)[:, 1]
    assert up[X["x"] > 0.5].mean() > 0.05
    assert up[X["x"] < -0.5].mean() < -0.02
    assert (m.predict_uplift(X)[:, 0] == 0).all()


# ----------------------------------------------------------------------------- loop


@pytest.fixture(scope="module")
def loop_data(cfg):
    ocfg = load_offers(
        horizon={"n_weeks": 6}, budget={"weekly": 4000}, uplift={"min_training_rows": 60}
    )
    return prepare_loop_data(cfg, ocfg, n_customers=700)


def test_loop_budget_cooldown_and_log(loop_data):
    ocfg = loop_data.ocfg
    K = len(ocfg.arms)
    for pol in ("risk_top", "uplift"):
        res = run_loop(pol, loop_data)
        w, lg = res.weekly, res.log
        assert len(w) == ocfg.n_weeks
        assert (w["spend"] <= ocfg.weekly_budget + 1e-6).all()
        # exploration rows are uniform over arms and logged with that probability
        ex = lg[lg["explore"]]
        assert np.allclose(ex["prob"].to_numpy(), 1 / K)
        assert set(ex["arm"].unique()) <= set(range(K))
        # a customer who got an offer is not offered again inside the cooldown
        offered = lg[lg["arm"] > 0].sort_values(["customer_id", "week"])
        gaps = offered.groupby("customer_id")["week"].diff().dropna()
        assert (gaps >= ocfg.cooldown_weeks).all()
        # the active book only shrinks
        assert (w["n_active_end"].diff().dropna() <= 0).all()
        assert (
            w["n_active_start"].iloc[1:].to_numpy() == w["n_active_end"].iloc[:-1].to_numpy()
        ).all()


def test_no_offers_spends_nothing_beyond_exploration(loop_data):
    res = run_loop("no_offers", loop_data)
    assert (res.weekly["spend_policy"] == 0).all()
    assert (res.log[~res.log["explore"]]["arm"] == 0).all()


def test_oracle_beats_random_on_truth(loop_data):
    ora = run_loop("oracle", loop_data).truth_weekly_reward
    rnd = run_loop("random_outreach", loop_data).truth_weekly_reward
    assert ora > rnd
    assert ora > 0


def test_every_configured_policy_is_implemented(ocfg):
    assert set(ocfg.policies) <= set(POLICIES)
