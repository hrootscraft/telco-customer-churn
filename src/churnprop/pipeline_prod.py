"""Phase 4: the weekly production job, and the end-to-end demonstration of it.

:func:`run_week` is the unit of work the Airflow DAG and the GitHub Actions job both
call for one batch week: validate → score → monitor → write the report. It needs only
the committed export in ``models/production/`` and the batch file.

:func:`run_phase4` (``churnprop production``) shows the whole thing on the simulated
year: trains and promotes v1, trains a deliberately weaker v2 to show the gate
refusing it, builds the twelve drifted batches, runs the weekly job for each, and
draws the drift figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop import plots
from churnprop.batches import BATCHES_ROOT, batches_config, load_batch, make_batches, validate_batch
from churnprop.config import REPO_ROOT, Config
from churnprop.monitor import monitor_week, write_report
from churnprop.pipeline_events import ensure_events
from churnprop.registry import (
    ProductionConfig,
    load_production,
    promote,
    registry_table,
    train_candidate,
)
from churnprop.scoring import SCORES_ROOT, feature_frame, score_frame

log = logging.getLogger(__name__)
MONITORING_ROOT = REPO_ROOT / "results" / "monitoring"


def next_week(cfg: Config, monitoring_root: Path | str = MONITORING_ROOT) -> int:
    """The week after the last one monitored, wrapping around the simulated year."""
    state = Path(monitoring_root) / "state.json"
    n = int(batches_config(cfg)["n_weeks"])
    if state.exists():
        with open(state) as f:
            last = int(json.load(f).get("last_week", 0))
        return last % n + 1
    return 1


def _labelled_batches(
    cfg: Config, week: int, model, batches_root, scores_root
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(batch_week, y, scores) for every earlier batch whose labels have matured by ``week``."""
    out = []
    for w in range(1, week + 1):
        p = Path(batches_root) / f"week_{w:02d}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p, columns=["label_available_week"])
        if int(b["label_available_week"].iloc[0]) > week:
            continue
        batch = load_batch(w, batches_root)
        sp = Path(scores_root) / f"week_{w:02d}.parquet"
        if sp.exists():
            s = pd.read_parquet(sp)["score"].to_numpy()
        else:
            s = score_frame(batch, model, id_col=cfg.data.id_col)["score"].to_numpy()
        out.append((w, batch["churned"].to_numpy(), s))
    return out


def run_week(
    cfg: Config,
    week: int,
    *,
    batches_root: Path | str = BATCHES_ROOT,
    scores_root: Path | str = SCORES_ROOT,
    monitoring_root: Path | str = MONITORING_ROOT,
    uplift_log_path: Path | str | None = None,
) -> dict:
    """Validate, score and monitor one weekly batch; write results/monitoring/week_NN.{json,md}."""
    model = load_production(cfg)
    batch = load_batch(week, batches_root)
    validate_batch(batch, cfg, model.features)
    scored = score_frame(batch, model, id_col=cfg.data.id_col)
    Path(scores_root).mkdir(parents=True, exist_ok=True)
    scored.to_parquet(Path(scores_root) / f"week_{week:02d}.parquet", index=False)

    X = feature_frame(batch, model)
    labelled = _labelled_batches(cfg, week, model, batches_root, scores_root)
    uplift_log, arms = None, None
    path = (
        Path(uplift_log_path)
        if uplift_log_path
        else REPO_ROOT / "results" / "uplift_policy_log.parquet"
    )
    if path.exists():
        from churnprop.offers.menu import load_offers

        uplift_log, arms = pd.read_parquet(path), list(load_offers().arms)
    report = monitor_week(
        week,
        X,
        scored["score"].to_numpy(),
        model.card,
        model.reference,
        cfg,
        labelled=labelled,
        uplift_log=uplift_log,
        arms=arms,
    )
    report["batch_date"] = str(pd.Timestamp(batch["batch_date"].iloc[0]).date())
    report["drift_injected"] = sorted(
        {t for tags in batch["drift_tags"] for t in tags.split("|") if t}
    )
    mon = Path(monitoring_root)
    js, md = write_report(report, mon)
    (mon / "latest.json").write_text(js.read_text())
    (mon / "latest.md").write_text(md.read_text())
    with open(mon / "state.json", "w") as f:
        json.dump({"last_week": week, "model_version": model.version}, f)
    _append_summary(report, mon)
    log.info(
        "week %2d: %-5s retrain=%s %s",
        week,
        report["status"],
        report["retrain"],
        "; ".join(report["reasons"][:3]),
    )
    return report


def _append_summary(report: dict, mon: Path) -> None:
    row = {
        "week": report["week"],
        "batch_date": report.get("batch_date"),
        "model_version": report["model_version"],
        "status": report["status"],
        "retrain": report["retrain"],
        "n_feature_alerts": sum(f["status"] == "alert" for f in report["features"]),
        "max_feature_psi": max(f["psi"] for f in report["features"]),
        "top_feature": report["features"][0]["feature"],
        "score_psi": report["scores"]["score_psi"],
        "mean_score": report["scores"]["mean_score"],
        "matured_batches": len(report["outcomes"]),
        "latest_observed_churn": report["outcomes"][-1]["observed_churn"]
        if report["outcomes"]
        else np.nan,
        "latest_predicted_churn": report["outcomes"][-1]["predicted_churn"]
        if report["outcomes"]
        else np.nan,
        "latest_ece": report["outcomes"][-1]["ece"] if report["outcomes"] else np.nan,
        "latest_pr_auc": report["outcomes"][-1]["pr_auc"] if report["outcomes"] else np.nan,
        "drift_injected": "|".join(report.get("drift_injected", [])),
        "reasons": " ; ".join(report["reasons"]),
    }
    p = mon / "summary.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    df = df[df["week"] != row["week"]] if len(df) else df
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("week").to_csv(
        p, index=False
    )
    psi_p = mon / "feature_psi.csv"
    psi = pd.read_csv(psi_p, index_col=0) if psi_p.exists() else pd.DataFrame()
    psi.loc[report["week"], [f["feature"] for f in report["features"]]] = [
        f["psi"] for f in report["features"]
    ]
    psi.sort_index().to_csv(psi_p)


def run_phase4(
    cfg: Config, out_dir: Path | str | None = None, fig_dir: Path | str | None = None
) -> dict:
    out = Path(out_dir or REPO_ROOT / "results")
    figs = Path(fig_dir or REPO_ROOT / "reports" / "figures")
    mon = out / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    for stale in (mon / "summary.csv", mon / "feature_psi.csv", mon / "state.json"):
        stale.unlink(missing_ok=True)
    ensure_events(cfg)
    pcfg = ProductionConfig.from_config(cfg)

    # 1. a candidate with event features, promoted through the gate (no production yet)
    v1 = train_candidate(cfg, pcfg, note="phase 4 baseline: snapshot + event features")
    d1 = promote(cfg, v1.version, pcfg=pcfg)
    # 2. a weaker candidate (snapshot features only) that the gate should refuse
    v2 = train_candidate(
        cfg, replace(pcfg, use_event_features=False), note="gate demo: snapshot features only"
    )
    d2 = promote(cfg, v2.version, pcfg=pcfg)
    prod = load_production(cfg)

    # 3. the simulated year of weekly batches, scored and monitored by the production model
    make_batches(cfg, force=True)
    n_weeks = int(batches_config(cfg)["n_weeks"])
    reports = [
        run_week(cfg, w, monitoring_root=mon, scores_root=out / "scores")
        for w in range(1, n_weeks + 1)
    ]

    summary = pd.read_csv(mon / "summary.csv")
    psi = pd.read_csv(mon / "feature_psi.csv", index_col=0)
    reg = registry_table(cfg, pcfg)
    reg.to_csv(out / "registry.csv", index=False)
    plots.plot_drift_heatmap(psi, summary, cfg, figs)
    plots.plot_monitoring_timeline(summary, prod.card, cfg, figs)

    drift_rules = batches_config(cfg)["drift"]
    first_alert = (
        int(summary.loc[summary["status"] == "alert", "week"].min())
        if (summary["status"] == "alert").any()
        else None
    )
    outcome_alert_weeks = [
        r["week"] for r in reports if any(o["status"] == "alert" for o in r["outcomes"])
    ]
    headline = {
        "production_version": prod.version,
        "production_holdout": prod.card["holdout"],
        "production_features": prod.card["n_features"],
        "gate_v1": d1,
        "gate_v2": d2,
        "candidate_v2_holdout": v2.metrics,
        "n_weeks": n_weeks,
        "drift_schedule": [{"from_week": r["from_week"], "kind": r["kind"]} for r in drift_rules],
        "first_alert_week": first_alert,
        "retrain_weeks": [int(w) for w in summary.loc[summary["retrain"], "week"]],
        "weeks_status": summary[["week", "status", "top_feature", "max_feature_psi", "score_psi"]]
        .round(3)
        .to_dict(orient="records"),
        "first_outcome_alert_week": min(outcome_alert_weeks) if outcome_alert_weeks else None,
        "registry": reg.to_dict(orient="records"),
        "fast": cfg.fast,
    }
    with open(out / "headline_phase4.json", "w") as f:
        json.dump(
            headline, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)
        )
    return headline
