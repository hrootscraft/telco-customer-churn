"""Command line: ``churnprop benchmark [--fast] [--config PATH] [--seed N]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from churnprop.config import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="churnprop",
        description="Retention propensity and offer targeting on the Telco churn data.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_ in [
        (
            "benchmark",
            "phase 1: 8-model benchmark, feature selection, SHAP; writes results/ and figures",
        ),
        ("events", "phase 2: simulate the event log and store it under data/events/"),
        ("features", "phase 2: build point-in-time event features as of a date"),
        (
            "benchmark-events",
            "phase 2: rerun four models with and without event features; writes results/ and figures",
        ),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
        sp.add_argument("--seed", type=int, default=None, help="override the config seed")
        sp.add_argument("--out", default=None, help="results directory")
        sp.add_argument("--figures", default=None, help="figures directory")
        sp.add_argument("--fast", action="store_true", help="fewer folds / trees / epochs, for CI")
        sp.add_argument("--as-of", default=None, help="features: as-of date (default: snapshot)")
        sp.add_argument(
            "--engine", default="duckdb", choices=["duckdb", "spark"], help="features: SQL engine"
        )
        sp.add_argument("--force", action="store_true", help="events: regenerate even if present")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    overrides = {"seed": a.seed} if a.seed is not None else {}
    cfg = load_config(a.config, fast=a.fast, **overrides)
    if a.command == "benchmark":
        from churnprop.pipeline import run_phase1

        h = run_phase1(cfg, a.out, a.figures)
        print(
            f"best model on hold-out: {h['best_model_pretty']}  ROC-AUC {h['holdout']['roc_auc']:.3f}  PR-AUC {h['holdout']['pr_auc']:.3f}"
        )
        return 0
    if a.command == "events":
        from churnprop.pipeline_events import ensure_events

        ev = ensure_events(cfg, force=a.force)
        print(f"event log: {ev.n_rows:,} rows, snapshot {ev.snapshot_date.date()}")
        return 0
    if a.command == "features":
        import pandas as pd

        from churnprop.config import REPO_ROOT
        from churnprop.data import load_clean
        from churnprop.features_sql.build import build_event_features
        from churnprop.pipeline_events import ensure_events

        ev = ensure_events(cfg)
        as_of = pd.Timestamp(a.as_of) if a.as_of else ev.snapshot_date
        ids = load_clean(cfg)[cfg.data.id_col]
        F = build_event_features(as_of, customer_ids=ids, engine=a.engine)
        out = Path(a.out or REPO_ROOT / "results") / f"event_features_{as_of.date()}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        F.to_parquet(out, index=False)
        print(f"{len(F):,} customers x {F.shape[1] - 2} features as of {as_of.date()} -> {out}")
        return 0
    if a.command == "benchmark-events":
        from churnprop.pipeline_events import run_phase2

        h = run_phase2(cfg, a.out, a.figures)
        for r in h["comparison"]:
            print(
                f"{r['pretty']:<26} PR-AUC {r['pr_auc_snapshot']:.3f} -> {r['pr_auc_events']:.3f}"
            )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
