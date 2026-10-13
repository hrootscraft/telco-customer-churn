"""Command line: ``churnprop benchmark [--fast] [--config PATH] [--seed N]``."""

from __future__ import annotations

import argparse
import logging
import sys

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
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
        sp.add_argument("--seed", type=int, default=None, help="override the config seed")
        sp.add_argument("--out", default=None, help="results directory")
        sp.add_argument("--figures", default=None, help="figures directory")
        sp.add_argument("--fast", action="store_true", help="fewer folds / trees / epochs, for CI")
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
