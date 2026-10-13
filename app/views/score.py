"""Page 1: score a customer, and see why."""

from __future__ import annotations

import io

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app import common as C  # noqa: E402

FORM_FIELDS = [
    ("Contract", ["Month-to-month", "One year", "Two year"]),
    ("Internet Service", ["Fiber optic", "DSL", "No"]),
    (
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    ),
    ("Tech Support", ["No", "Yes", "No internet service"]),
    ("Online Security", ["No", "Yes", "No internet service"]),
    ("Paperless Billing", ["Yes", "No"]),
    ("Dependents", ["No", "Yes"]),
    ("Senior Citizen", ["No", "Yes"]),
]


def _shap_bar(values: pd.Series, title: str):
    v = values.sort_values(key=abs, ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(v) + 1.0))
    ax.barh(range(len(v)), v.to_numpy(), color=[C.RED if x > 0 else C.BLUE for x in v], height=0.62)
    ax.set_yticks(range(len(v)), v.index)
    ax.axvline(0, color=C.INK2, lw=0.8)
    ax.set_xlabel("Push on the log-odds of churn (red up, blue down)")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=C.GRID, lw=0.6)
    fig.tight_layout()
    return fig


def _explain_row(row: pd.DataFrame):
    folded = C.shap_for(row)
    vals = folded.iloc[0]
    cols = st.columns([3, 2])
    with cols[0]:
        st.pyplot(_shap_bar(vals, "What moves this customer's score"), clear_figure=True)
    with cols[1]:
        st.caption("Top reasons, in the model's own units")
        shown = vals.sort_values(key=abs, ascending=False).head(6)
        raw = row.iloc[0]
        for f, s in shown.items():
            val = raw[f] if f in raw else ""
            if isinstance(val, float):
                val = f"{val:,.1f}"
            st.markdown(f"{'🔺' if s > 0 else '🔻'} **{f}** = {val}  ·  {s:+.2f}")
        st.caption(
            "SHAP values from the production XGBoost, one-hot columns folded back to the raw "
            "feature (learn note 02). They describe the model, not a causal effect."
        )


def _score_card(score: float, card: dict):
    base = card.get("train_churn_rate", 0.265)
    q = C.quantile_rank(score)
    a, b, c = st.columns(3)
    a.metric(
        "4-week churn score",
        C.pct(score),
        help="Calibrated probability from the production model (ECE 0.02 on the hold-out).",
    )
    b.metric(
        "vs the book",
        f"{score / base:.1f}× base rate",
        help=f"Book-wide churn rate at training: {C.pct(base)}",
    )
    c.metric(
        "Percentile in the hold-out",
        f"{100 * q:.0f}th",
        help="Share of hold-out customers with a lower score.",
    )
    if score >= 0.5:
        st.error(
            "High risk: in the top decile, where 78% of customers churn. A candidate for this week's offer plan."
        )
    elif score >= 0.3:
        st.warning(
            "Elevated risk: above the base rate. Whether an offer helps is the uplift question, not the propensity one (learn note 03)."
        )
    else:
        st.success(
            "Low risk: below the base rate. Offers here are mostly wasted, and some customers are sleeping dogs."
        )


def render():
    st.title("Score a customer")
    m = C.model()
    card = m.card
    st.caption(
        f"Production model **v{card['version']}** · {card['family_pretty']} on {card['n_features']} features "
        f"(snapshot + event features as of the scoring date) · hold-out PR-AUC {card['holdout']['pr_auc']:.3f}, "
        f"ROC-AUC {card['holdout']['roc_auc']:.3f}, ECE {card['holdout']['ece']:.3f} · trained {card['trained_at'][:10]}"
    )
    mode = st.radio(
        "Who do you want to score?",
        ["A hold-out customer", "A customer I describe", "A CSV I upload"],
        horizontal=True,
    )
    c = C.cfg()

    if mode == "A hold-out customer":
        book = C.holdout_scored()
        st.caption("1,409 customers the model never trained on, ranked by score. Pick one.")
        q = st.text_input("Filter by id, contract or internet service", "")
        view = book
        if q:
            mask = (
                book[c.data.id_col].str.contains(q, case=False)
                | book["Contract"].str.contains(q, case=False)
                | book["Internet Service"].str.contains(q, case=False)
            )
            view = book[mask]
        view = view.sort_values("score", ascending=False)
        labels = [
            f"{r[c.data.id_col]} · {r['Contract']} · {r['Internet Service']} · {r['Tenure Months']:.0f} months · score {100 * r['score']:.0f}%"
            for _, r in view.head(300).iterrows()
        ]
        if not labels:
            st.info("No customer matches that filter.")
            return
        pick = st.selectbox("Customer", labels, index=0)
        cid = pick.split(" · ")[0]
        row = book[book[c.data.id_col] == cid].head(1)
        _score_card(float(row["score"].iloc[0]), card)
        with st.expander("Customer details"):
            show = row[list(c.data.feature_cols) + ["CLTV"]].T
            show.columns = ["value"]
            st.dataframe(show, use_container_width=True)
        _explain_row(row)
        if int(row["Churn Value"].iloc[0]) == 1:
            st.caption(
                "This customer did churn in the IBM snapshot (the label the model never saw for hold-out rows)."
            )
        else:
            st.caption("This customer had not churned in the IBM snapshot.")

    elif mode == "A customer I describe":
        st.caption(
            "Describe the contract and services; everything you leave alone takes the book's typical value, and event features take their documented defaults (no history)."
        )
        book = C.snapshot()
        typical = book.mode(numeric_only=False).iloc[0]
        with st.form("describe"):
            cols = st.columns(4)
            vals = {}
            for i, (f, options) in enumerate(FORM_FIELDS):
                vals[f] = cols[i % 4].selectbox(f, options, index=0)
            n1, n2, n3 = st.columns(3)
            vals["Tenure Months"] = n1.slider("Tenure (months)", 0, 72, 12)
            vals["Monthly Charges"] = n2.slider("Monthly charges ($)", 18.0, 120.0, 80.0, 0.5)
            vals["Total Charges"] = n3.number_input(
                "Total charges ($)",
                0.0,
                9000.0,
                float(vals["Tenure Months"] * vals["Monthly Charges"]),
                10.0,
            )
            submitted = st.form_submit_button("Score")
        if submitted:
            row = typical.to_frame().T.copy()
            for k, v in vals.items():
                row[k] = v
            for col in (
                "Streaming TV",
                "Streaming Movies",
                "Online Backup",
                "Device Protection",
                "Multiple Lines",
                "Phone Service",
                "Partner",
                "Gender",
            ):
                if col in row:
                    row[col] = typical[col]
            if vals["Internet Service"] == "No":
                for col in (
                    "Tech Support",
                    "Online Security",
                    "Online Backup",
                    "Device Protection",
                    "Streaming TV",
                    "Streaming Movies",
                ):
                    if col in row:
                        row[col] = "No internet service"
            row[c.data.id_col] = "described"
            row["Total Charges"] = float(vals["Total Charges"])
            for num in ("Tenure Months", "Monthly Charges", "Total Charges"):
                row[num] = row[num].astype(float)
            s, _ = C.score_rows(row, explain=False)
            _score_card(float(s["score"].iloc[0]), card)
            _explain_row(row)

    else:
        st.caption(
            "Upload a CSV with the snapshot columns (same names as the IBM file: Contract, Tenure Months, Monthly Charges, Total Charges, Internet Service, …). "
            "Event-feature columns are optional; missing ones take their documented defaults."
        )
        up = st.file_uploader("CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            missing = [f for f in c.data.numeric + c.data.categorical if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                return
            s, reasons = C.score_rows(df, explain=len(df) <= 2000)
            out = s[[c.data.id_col, "score", "decile", "model_version"]]
            if reasons is not None:
                out = pd.concat([out, reasons.iloc[:, :3]], axis=1)
            st.success(
                f"Scored {len(out):,} rows with model v{m.version}. Mean score {C.pct(out['score'].mean())}."
            )
            st.dataframe(
                out.sort_values("score", ascending=False), use_container_width=True, height=420
            )
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download scores (CSV)", buf.getvalue(), "scores.csv", "text/csv")
