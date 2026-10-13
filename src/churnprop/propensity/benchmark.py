"""Repeated stratified cross-validation of the model zoo, then one look at the hold-out.

Protocol
--------
1. Stratified 80/20 split (``data.make_split``); the 20% is not touched until step 4.
2. On the 80%: ``n_repeats × n_folds`` stratified CV. In each fold the pipeline
   fits preprocessing *and SMOTE* on the training part only, then scores the
   validation part as-is. Every metric is averaged over folds (mean ± sd).
3. Three resampling regimes are compared on the same folds: none, SMOTE,
   class weights. (SMOTE is on the resume; the comparison shows what it buys.)
4. Each model is refitted on the whole 80% and scored once on the 20% hold-out;
   that table is the one the README quotes. Calibration curves come from it too.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from churnprop.config import REPO_ROOT, Config
from churnprop.data import Split
from churnprop.propensity.metrics import calibration_table, score_all
from churnprop.propensity.models import FAMILY, PRETTY, make_model

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    cv: pd.DataFrame  # one row per model × regime × repeat × fold
    cv_summary: pd.DataFrame  # mean ± sd per model × regime
    test: pd.DataFrame  # one row per model (chosen regime) on the hold-out
    calibration: pd.DataFrame  # reliability bins per model on the hold-out
    test_scores: pd.DataFrame  # customer-level hold-out scores per model
    fitted: dict[str, object] = field(default_factory=dict)  # model name -> fitted pipeline

    def save(self, out_dir: Path | str = REPO_ROOT / "results") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.cv.to_csv(out / "benchmark_cv_folds.csv", index=False)
        self.cv_summary.to_csv(out / "benchmark_cv.csv", index=False)
        self.test.to_csv(out / "benchmark_test.csv", index=False)
        self.calibration.to_csv(out / "calibration.csv", index=False)
        self.test_scores.to_csv(out / "test_scores.csv", index=False)


def _cv_one(model, X, y, cv, *, cfg: Config, name: str, regime: str) -> list[dict]:
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        t0 = time.perf_counter()
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        row = score_all(
            y.iloc[va],
            p,
            top_frac=cfg.benchmark.top_decile,
            precision_floor=cfg.benchmark.precision_floor,
            n_bins=cfg.benchmark.calibration_bins,
        )
        row.update(
            model=name,
            regime=regime,
            repeat=k // cfg.benchmark.n_folds,
            fold=k % cfg.benchmark.n_folds,
            fit_seconds=time.perf_counter() - t0,
        )
        rows.append(row)
    return rows


def run_benchmark(
    split: Split,
    cfg: Config,
    *,
    models: tuple[str, ...] | None = None,
    regimes: tuple[str, ...] = ("none", "smote", "class_weight"),
    chosen_regime: str | None = None,
    feature_cols: list[str] | None = None,
) -> BenchmarkResult:
    models = models or cfg.benchmark.models
    chosen_regime = chosen_regime or cfg.benchmark.resampling
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.benchmark.n_folds, n_repeats=cfg.benchmark.n_repeats, random_state=cfg.seed
    )
    X, y = split.X_train, split.y_train
    rows: list[dict] = []
    for name in models:
        for regime in regimes:
            # knn has no class_weight; its "class_weight" regime is identical to none, skip it
            if name == "knn" and regime == "class_weight":
                continue
            log.info("cv  %-14s %-12s", name, regime)
            model = make_model(name, cfg, feature_cols, resampling=regime)
            rows += _cv_one(model, X, y, cv, cfg=cfg, name=name, regime=regime)
    cv_df = pd.DataFrame(rows)
    metric_cols = [c for c in cv_df.columns if c not in ("model", "regime", "repeat", "fold")]
    summ = cv_df.groupby(["model", "regime"])[metric_cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["pretty"] = summ["model"].map(PRETTY)
    summ["family"] = summ["model"].map(FAMILY)

    # Hold-out: refit on all training rows in the chosen regime, score the 20% once.
    test_rows, cal_rows, fitted = [], [], {}
    scores = pd.DataFrame({"customer_id": split.test_ids, "y": split.y_test})
    for name in models:
        regime = "none" if (name == "knn" and chosen_regime == "class_weight") else chosen_regime
        log.info("test %-14s %-12s", name, regime)
        m = make_model(name, cfg, feature_cols, resampling=regime)
        t0 = time.perf_counter()
        m.fit(X, y)
        p = m.predict_proba(split.X_test)[:, 1]
        row = score_all(
            split.y_test,
            p,
            top_frac=cfg.benchmark.top_decile,
            precision_floor=cfg.benchmark.precision_floor,
            n_bins=cfg.benchmark.calibration_bins,
        )
        row.update(model=name, pretty=PRETTY[name], family=FAMILY[name], regime=regime)
        row["fit_seconds"] = time.perf_counter() - t0
        test_rows.append(row)
        ct = calibration_table(split.y_test.to_numpy(), p, cfg.benchmark.calibration_bins)
        ct.insert(0, "model", name)
        cal_rows.append(ct)
        scores[name] = p
        fitted[name] = m
    test_df = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return BenchmarkResult(
        cv=cv_df,
        cv_summary=summ,
        test=test_df,
        calibration=pd.concat(cal_rows, ignore_index=True),
        test_scores=scores,
        fitted=fitted,
    )


def save_models(
    result: BenchmarkResult, out_dir: Path | str = REPO_ROOT / "models" / "benchmark"
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in result.fitted.items():
        joblib.dump(m, out / f"{name}.joblib")


def naive_baselines(y_test: pd.Series) -> pd.DataFrame:
    """What 'predict nobody churns' and 'predict the base rate' score, for the README."""
    y = y_test.to_numpy()
    rows = []
    const = np.full(len(y), y.mean())
    r = score_all(y, const)
    r.update(model="base_rate", pretty="Constant = base rate", family="baseline")
    rows.append(r)
    zero = np.full(len(y), 1e-6)
    r = score_all(y, zero)
    r.update(model="nobody_churns", pretty="Predict nobody churns", family="baseline")
    rows.append(r)
    return pd.DataFrame(rows)
