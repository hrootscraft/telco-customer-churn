"""Page 2: this week's offer plan under a budget."""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app import common as C  # noqa: E402
from churnprop.uplift.allocate import allocate_budget  # noqa: E402

ARM_COLOR = {"bill_credit": C.BLUE, "tech_bundle": C.GREEN, "contract_upgrade": C.PURPLE}


def _eligible(df: pd.DataFrame, arm: str) -> np.ndarray:
    """Who each offer makes sense for at all (the uplift model learns the rest)."""
    if arm == "tech_bundle":
        return (df["Internet Service"] != "No").to_numpy() & (
            (df["Tech Support"] != "Yes") | (df["Online Security"] != "Yes")
        ).to_numpy()
    if arm == "contract_upgrade":
        return (df["Contract"] == "Month-to-month").to_numpy()
    return np.ones(len(df), bool)


def render():
    st.title("This week's offer plan")
    o = C.offers()
    book = C.holdout_scored()
    c = C.cfg()
    st.caption(
        "The hold-out book (1,409 customers the model never saw), scored by the production model and "
        "priced with IBM's CLTV. One arm per customer, total cost within the budget, solved with a "
        "Lagrange multiplier (learn note 05). This page uses the risk × value heuristic; the learned "
        "multi-offer uplift policy that replaces it once outcomes mature is phase 3's `churnprop loop`."
    )
    with st.sidebar:
        st.header("Assumptions")
        scale = len(book) / 5174
        budget = st.slider(
            "Weekly budget ($)", 1000, 40000, int(round(o.weekly_budget * scale / 500) * 500), 500
        )
        st.caption(f"Default scales the loop's ${o.weekly_budget:,.0f}/week to this book's size.")
        save = {}
        defaults = {"bill_credit": 0.25, "tech_bundle": 0.35, "contract_upgrade": 0.40}
        for off in o.offers:
            save[off.name] = st.slider(
                f"Assumed save rate: {off.label}",
                0.0,
                0.6,
                defaults.get(off.name, float(o.assumed_save_rate)),
                0.05,
                help=f"Share of churn hazard the offer removes for an eligible customer. Cost ${off.cost:.0f}.",
            )
        min_score = st.slider("Only consider customers with score ≥", 0.0, 0.8, 0.0, 0.05)

    costs = np.array(o.costs)
    value = book["CLTV"].to_numpy(float) * o.value_margin
    p = book["score"].to_numpy()
    ev = np.zeros((len(book), len(o.arms)))
    for k, arm in enumerate(o.arms):
        if k == 0:
            continue
        elig = _eligible(book, arm) & (p >= min_score)
        ev[:, k] = np.where(elig, p * save[arm] * value - costs[k], -1e12)
    alloc = allocate_budget(ev, costs, budget)
    arm = alloc.arm
    offered = arm > 0
    exp_saves = float((p[offered] * np.array([save[o.arms[k]] for k in arm[offered]])).sum())
    exp_value = float(
        (p[offered] * np.array([save[o.arms[k]] for k in arm[offered]]) * value[offered]).sum()
    )

    a, b, c2, d, e = st.columns(5)
    a.metric("Customers offered", f"{int(offered.sum()):,}", help=f"of {len(book):,} in the book")
    b.metric("Spend", C.money(alloc.spend), help=f"budget {C.money(budget)}")
    c2.metric("Expected churners saved", f"{exp_saves:,.1f}")
    d.metric(
        "Expected value retained",
        C.money(exp_value),
        help="Σ score × save rate × CLTV over offered customers",
    )
    e.metric(
        "λ: $ retained per $ spent, at the margin",
        f"{alloc.lam:.2f}",
        help="0 means the budget is not binding: every offer with positive expected value is sent.",
    )
    if alloc.lam == 0 and alloc.spend < budget * 0.95:
        st.info(
            "The budget is not binding: every offer with positive expected value under these assumptions is already in the plan. Lower the save rates or the budget to see λ bite."
        )

    plan = book.loc[
        offered,
        [
            c.data.id_col,
            "Contract",
            "Internet Service",
            "Tenure Months",
            "Monthly Charges",
            "CLTV",
            "score",
        ],
    ].copy()
    plan["offer"] = [o.arms[k] for k in arm[offered]]
    plan["cost"] = costs[arm[offered]]
    plan["expected_value"] = ev[offered, arm[offered]]
    plan = plan.sort_values("expected_value", ascending=False)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("By segment")
        seg = plan.assign(segment=plan["Contract"] + " · " + plan["Internet Service"])
        allseg = book.assign(segment=book["Contract"] + " · " + book["Internet Service"])
        tbl = (
            seg.groupby("segment")
            .agg(
                offered=("offer", "size"),
                spend=("cost", "sum"),
                expected_value=("expected_value", "sum"),
                mean_score=("score", "mean"),
            )
            .join(allseg.groupby("segment").size().rename("in_book"))
            .assign(share_offered=lambda t: t["offered"] / t["in_book"])
            .sort_values("expected_value", ascending=False)
        )
        st.dataframe(
            tbl.style.format(
                {
                    "spend": "${:,.0f}",
                    "expected_value": "${:,.0f}",
                    "mean_score": "{:.0%}",
                    "share_offered": "{:.0%}",
                }
            ),
            width="stretch",
        )
    with right:
        st.subheader("Offer mix")
        mix = plan.groupby("offer")["cost"].agg(["size", "sum"])
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.bar(
            mix.index, mix["size"], color=[ARM_COLOR.get(k, C.BLUE) for k in mix.index], width=0.6
        )
        for i, (n, s) in enumerate(zip(mix["size"], mix["sum"], strict=True)):
            ax.text(i, n, f"{n}\n${s:,.0f}", ha="center", va="bottom", fontsize=8.5, color=C.INK)
        ax.set_ylabel("Customers")
        ax.set_ylim(0, max(1, mix["size"].max()) * 1.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=C.GRID, lw=0.6)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.subheader("The list")
    st.dataframe(
        plan.style.format(
            {
                "Monthly Charges": "${:.2f}",
                "CLTV": "${:,.0f}",
                "score": "{:.0%}",
                "cost": "${:.0f}",
                "expected_value": "${:,.0f}",
            }
        ),
        width="stretch",
        height=380,
    )
    buf = io.StringIO()
    plan.to_csv(buf, index=False)
    st.download_button("Download the plan (CSV)", buf.getvalue(), "offer_plan.csv", "text/csv")

    with st.expander("What the learned uplift policy does differently (phase 3)"):
        st.markdown(
            "The sliders above assume one save rate per offer for everyone. The weekly loop replaces that "
            "with a T-learner trained on the randomised exploration holdout and the logged policy rows, "
            "which learns **who** each offer helps and who it pushes out (sleeping dogs). On the simulated "
            "year it earned $14.4k/week against $6.8k for this heuristic over the second half, with 19% "
            "fewer offers. The comparison is in the README and `results/policy_comparison.csv`."
        )
        p12 = C.ROOT / "reports" / "figures" / "12_policy_comparison.png"
        if p12.exists():
            st.image(str(p12))
