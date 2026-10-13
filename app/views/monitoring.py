"""Page 3: is the model still right this week?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import common as C


def render():
    st.title("Monitoring")
    latest = C.read_json("results/monitoring/latest.json")
    summary = C.read_csv("results/monitoring/summary.csv")
    reg = C.read_csv("results/registry.csv")
    card = C.model().card

    if not latest:
        st.warning("No monitoring reports yet. Run `churnprop production` or the weekly job.")
        return

    st.caption(
        f"Reports in `results/monitoring/`, written by `churnprop weekly` (the Airflow DAG and the Monday "
        f"GitHub Actions job call the same function). Latest: batch week {latest['week']} "
        f"({latest.get('batch_date', '')}), model v{latest['model_version']}."
    )
    a, b, c, d = st.columns(4)
    a.metric("Latest verdict", C.status_badge(latest["status"]))
    b.metric("Retrain recommended", "yes" if latest["retrain"] else "no")
    worst = latest["top_drifted"][0] if latest.get("top_drifted") else {"feature": "-", "psi": 0}
    c.metric("Worst feature PSI", f"{worst['psi']:.2f}", help=worst["feature"])
    d.metric(
        "Score PSI",
        f"{latest['scores']['score_psi']:.3f}",
        help=f"mean score {latest['scores']['ref_mean_score']:.3f} at training → {latest['scores']['mean_score']:.3f}",
    )
    if latest["reasons"]:
        st.error("Alerts this week:\n\n" + "\n".join(f"- {r}" for r in latest["reasons"]))
    else:
        st.success("No alerts this week.")

    tabs = st.tabs(
        ["The year at a glance", "Feature drift", "Week report", "Model registry", "Model card"]
    )
    with tabs[0]:
        p18 = C.ROOT / "reports" / "figures" / "18_monitoring_timeline.png"
        if p18.exists():
            st.image(str(p18), width="stretch")
        if len(summary):
            show = summary[
                [
                    "week",
                    "batch_date",
                    "status",
                    "retrain",
                    "top_feature",
                    "max_feature_psi",
                    "score_psi",
                    "latest_observed_churn",
                    "latest_predicted_churn",
                    "latest_ece",
                    "latest_pr_auc",
                    "drift_injected",
                ]
            ].copy()
            show["status"] = show["status"].map(C.status_badge)
            st.dataframe(
                show.style.format(
                    {
                        "max_feature_psi": "{:.2f}",
                        "score_psi": "{:.3f}",
                        "latest_observed_churn": "{:.3f}",
                        "latest_predicted_churn": "{:.3f}",
                        "latest_ece": "{:.3f}",
                        "latest_pr_auc": "{:.3f}",
                    },
                    na_rep="",
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The three injected drifts (price rise week 5, label shift week 7, mix shift week 8) and how the monitors saw them: inputs immediately, outcomes four weeks later (D21, D22)."
            )
    with tabs[1]:
        p17 = C.ROOT / "reports" / "figures" / "17_drift_heatmap.png"
        if p17.exists():
            st.image(str(p17), width="stretch")
        psi = C.read_csv("results/monitoring/feature_psi.csv", index_col=0)
        if len(psi):
            wk = st.select_slider("Week", options=list(psi.index), value=psi.index[-1])
            row = psi.loc[wk].sort_values(ascending=False).head(15).rename("PSI").to_frame()
            thr = C.cfg().extra.get("monitoring", {})
            row["status"] = [
                "alert"
                if v >= thr.get("psi_alert", 0.25)
                else ("warn" if v >= thr.get("psi_warn", 0.10) else "ok")
                for v in row["PSI"]
            ]
            row["status"] = row["status"].map(C.status_badge)
            st.dataframe(row.style.format({"PSI": "{:.3f}"}), width="stretch")
    with tabs[2]:
        weeks = sorted(
            int(p.stem.split("_")[1]) for p in (C.ROOT / "results" / "monitoring").glob("week_*.md")
        )
        if weeks:
            wk = st.selectbox("Week", weeks, index=len(weeks) - 1)
            st.markdown((C.ROOT / "results" / "monitoring" / f"week_{wk:02d}.md").read_text())
            with st.expander("Raw JSON"):
                st.json(C.read_json(f"results/monitoring/week_{wk:02d}.json"), expanded=False)
    with tabs[3]:
        if len(reg):
            st.dataframe(
                reg.style.format({"pr_auc": "{:.3f}", "roc_auc": "{:.3f}", "ece": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
        st.markdown(
            "Versions live in MLflow (local SQLite, `mlflow ui --backend-store-uri sqlite:///mlflow.db`); the one with the "
            "`production` alias is exported to `models/production/` and committed, which is what this app loads. "
            "Promotion runs a gate: hold-out PR-AUC within 0.005 of production and ECE ≤ 0.05 (D19, D20). "
            "v2 above was trained without event features and refused."
        )
    with tabs[4]:
        st.json(card, expanded=False)
        drivers = pd.DataFrame(card.get("top_drivers", []))
        if len(drivers):
            st.caption("Top drivers (share of mean |SHAP| at training)")
            st.bar_chart(drivers.set_index("feature")["share"], color=C.BLUE)
