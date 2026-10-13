"""Phase 2 end to end: event log → point-in-time features → benchmark with and without them.

``churnprop events`` simulates and stores the log; ``churnprop features`` builds the
feature table as of a date; ``churnprop benchmark-events`` reruns a subset of the
model zoo on snapshot features alone and on snapshot + event features, using the
*same* split and folds as phase 1, and reports the lift.

The lift is by construction: the simulator plants a churn signal in the last three
months (``events.signal`` in the config). What the comparison demonstrates is that
the SQL pipeline recovers it as of the right date, and how much a modest planted
signal moves PR-AUC; it is not a claim about real telco event data (D9).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from churnprop import plots
from churnprop.config import REPO_ROOT, Config
from churnprop.data import Split, load_clean, make_split
from churnprop.events.simulate import EventLog, EventsConfig, simulate_events
from churnprop.explain import run_explain
from churnprop.features_sql.build import EVENT_FEATURES, build_event_features
from churnprop.propensity.benchmark import run_benchmark

log = logging.getLogger(__name__)

EVENT_FEATURE_COLS = [*EVENT_FEATURES, "has_usage_history"]
EVENTS_ROOT = REPO_ROOT / "data" / "events"
COMPARE_MODELS = ("logistic", "lightgbm", "xgboost", "mlp")


def ensure_events(cfg: Config, root: Path | str = EVENTS_ROOT, *, force: bool = False) -> EventLog:
    root = Path(root)
    if (root / "SNAPSHOT_DATE").exists() and not force:
        return EventLog.load(root)
    df = load_clean(cfg)
    ev = simulate_events(df, cfg, EventsConfig.from_config(cfg))
    ev.save(root)
    return ev


def attach_event_features(split: Split, F: pd.DataFrame) -> Split:
    """Return a copy of ``split`` with the event features joined on customer id."""
    f = F.set_index("customer_id")[EVENT_FEATURE_COLS]

    def _join(X: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        extra = f.reindex(ids.to_numpy()).reset_index(drop=True)
        return pd.concat([X.reset_index(drop=True), extra], axis=1)

    return replace(
        split,
        X_train=_join(split.X_train, split.train_ids),
        X_test=_join(split.X_test, split.test_ids),
    )


def run_phase2(
    cfg: Config, out_dir: Path | str | None = None, fig_dir: Path | str | None = None
) -> dict:
    out = Path(out_dir or REPO_ROOT / "results")
    figs = Path(fig_dir or REPO_ROOT / "reports" / "figures")
    out.mkdir(parents=True, exist_ok=True)

    ev = ensure_events(cfg)
    df = load_clean(cfg)
    split = make_split(df, cfg)
    F = build_event_features(
        ev.snapshot_date, events_root=EVENTS_ROOT, customer_ids=df[cfg.data.id_col]
    )
    F.to_parquet(out / "event_features_snapshot.parquet", index=False)
    split_ev = attach_event_features(split, F)

    regimes = (cfg.benchmark.resampling,)
    base = run_benchmark(split, cfg, models=COMPARE_MODELS, regimes=regimes)
    with_ev = run_benchmark(split_ev, cfg, models=COMPARE_MODELS, regimes=regimes)

    rows = []
    for name in COMPARE_MODELS:
        b = base.test.set_index("model").loc[name]
        e = with_ev.test.set_index("model").loc[name]
        bc = base.cv_summary.set_index("model").loc[name]
        ec = with_ev.cv_summary.set_index("model").loc[name]
        rows.append(
            {
                "model": name,
                "pretty": b["pretty"],
                "roc_auc_snapshot": b["roc_auc"],
                "roc_auc_events": e["roc_auc"],
                "pr_auc_snapshot": b["pr_auc"],
                "pr_auc_events": e["pr_auc"],
                "pr_auc_lift": e["pr_auc"] - b["pr_auc"],
                "cv_pr_auc_snapshot": bc["pr_auc_mean"],
                "cv_pr_auc_events": ec["pr_auc_mean"],
                "cv_pr_auc_sd": ec["pr_auc_std"],
                "lift_top_decile_snapshot": b["lift_top_decile"],
                "lift_top_decile_events": e["lift_top_decile"],
                "brier_snapshot": b["brier"],
                "brier_events": e["brier"],
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "benchmark_event_features.csv", index=False)
    plots.plot_event_feature_lift(comp, figs)

    expl = run_explain(split_ev, with_ev.fitted["lightgbm"], cfg)
    expl.global_.to_csv(out / "shap_global_events.csv", index=False)
    plots.plot_shap_global(
        expl.global_,
        figs,
        name="11_shap_global_with_events.png",
        title="What drives the score with event features",
    )
    ev_share = (
        expl.global_.set_index("feature")
        .loc[[c for c in EVENT_FEATURE_COLS if c in expl.global_["feature"].values], "share"]
        .sum()
    )

    headline = {
        "snapshot_date": ev.snapshot_date.strftime("%Y-%m-%d"),
        "n_event_rows": int(ev.n_rows),
        "n_billing": int(len(ev.billing)),
        "n_usage": int(len(ev.usage)),
        "n_tickets": int(len(ev.tickets)),
        "n_plan_changes": int(len(ev.plan_changes)),
        "n_event_features": len(EVENT_FEATURE_COLS),
        "planted_signal": EventsConfig.from_config(cfg).signal,
        "comparison": comp.round(4).to_dict(orient="records"),
        "event_feature_shap_share": float(ev_share),
        "top_event_features": expl.global_[expl.global_["feature"].isin(EVENT_FEATURE_COLS)]
        .head(5)[["feature", "share"]]
        .round(4)
        .to_dict(orient="records"),
        "fast": cfg.fast,
    }
    with open(out / "headline_phase2.json", "w") as f:
        json.dump(headline, f, indent=2)
    for r in rows:
        log.info(
            "%-10s PR-AUC %.3f -> %.3f (ROC-AUC %.3f -> %.3f)",
            r["model"],
            r["pr_auc_snapshot"],
            r["pr_auc_events"],
            r["roc_auc_snapshot"],
            r["roc_auc_events"],
        )
    return headline
