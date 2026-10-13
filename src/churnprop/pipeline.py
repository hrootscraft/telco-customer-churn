"""Phase 1 end to end: data → benchmark → feature selection → SHAP → figures → results/.

``churnprop benchmark`` calls :func:`run_phase1`. Everything it writes is read
back by the README, the learning notes and (later) the app, so file names are
part of the contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from churnprop import plots
from churnprop.config import REPO_ROOT, Config
from churnprop.data import churn_by, load_clean, make_split
from churnprop.explain import run_explain
from churnprop.propensity.benchmark import naive_baselines, run_benchmark, save_models
from churnprop.propensity.stepwise import run_feature_selection

log = logging.getLogger(__name__)


def run_phase1(
    cfg: Config, out_dir: Path | str | None = None, fig_dir: Path | str | None = None
) -> dict:
    out = Path(out_dir or REPO_ROOT / "results")
    figs = Path(fig_dir or REPO_ROOT / "reports" / "figures")
    out.mkdir(parents=True, exist_ok=True)

    df = load_clean(cfg)
    split = make_split(df, cfg)
    log.info(
        "data: %d customers, %.1f%% churn, train %d / test %d",
        len(df),
        100 * df[cfg.data.target].mean(),
        len(split.X_train),
        len(split.X_test),
    )

    facts = {
        "n_customers": int(len(df)),
        "churn_rate": float(df[cfg.data.target].mean()),
        "churn_by_contract": churn_by(df, "Contract", cfg).reset_index().to_dict(orient="records"),
        "churn_by_internet": churn_by(df, "Internet Service", cfg)
        .reset_index()
        .to_dict(orient="records"),
        "churn_by_payment": churn_by(df, "Payment Method", cfg)
        .reset_index()
        .to_dict(orient="records"),
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
    }
    plots.plot_churn_by_contract(churn_by(df, "Contract", cfg), figs)

    # Benchmark
    res = run_benchmark(split, cfg)
    res.save(out)
    save_models(res)
    base = naive_baselines(split.y_test)
    base.to_csv(out / "benchmark_baselines.csv", index=False)
    chosen = cfg.benchmark.resampling
    plots.plot_benchmark(res.test, res.cv_summary, chosen, figs)
    plots.plot_regimes(res.cv_summary, figs)
    top3 = list(res.test["model"].head(3)) + (
        ["mlp"] if "mlp" not in list(res.test["model"].head(3)) else []
    )
    plots.plot_curves(res.test_scores, top3, figs)
    plots.plot_calibration(res.calibration, top3, figs)

    # Feature selection
    fs = run_feature_selection(split, cfg)
    fs.to_csv(out / "feature_selection.csv", index=False)
    plots.plot_feature_selection(fs, figs)

    # SHAP on the configured tree model
    expl = run_explain(split, res.fitted[cfg.explain.model], cfg)
    expl.save(out)
    plots.plot_shap_global(expl.global_, figs)
    for i, seg in enumerate(cfg.explain.segments[:2]):
        plots.plot_shap_segments(
            expl.by_segment, seg, figs, f"0{8 + i}_shap_by_{seg.lower().replace(' ', '_')}.png"
        )

    best = res.test.iloc[0]
    headline = {
        **facts,
        "resampling": chosen,
        "best_model": best["model"],
        "best_model_pretty": best["pretty"],
        "holdout": {
            k: float(best[k])
            for k in (
                "roc_auc",
                "pr_auc",
                "lift_top_decile",
                "recall_at_precision",
                "accuracy",
                "brier",
                "ece",
            )
        },
        "nobody_churns_accuracy": float(base.set_index("model").loc["nobody_churns", "accuracy"]),
        "models_ranked": list(res.test["model"]),
        "mlp_holdout": {
            k: float(res.test.set_index("model").loc["mlp", k])
            for k in ("roc_auc", "pr_auc", "accuracy")
        },
        "top_drivers": expl.global_.head(5)[["feature", "share"]].to_dict(orient="records"),
        "stepwise": {
            "kept": fs.attrs["stepwise_kept"],
            "n_kept": len(fs.attrs["stepwise_kept"]),
            "n_total": int(len(fs)),
            "cv_auc_full": fs.attrs["cv_auc_full"],
            "cv_auc_stepwise": fs.attrs["cv_auc_stepwise"],
        },
        "fast": cfg.fast,
        "seed": cfg.seed,
    }
    with open(out / "headline_phase1.json", "w") as f:
        json.dump(headline, f, indent=2, default=_jsonable)
    log.info(
        "best on hold-out: %s ROC-AUC %.3f PR-AUC %.3f",
        best["pretty"],
        best["roc_auc"],
        best["pr_auc"],
    )
    return headline


def _jsonable(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
