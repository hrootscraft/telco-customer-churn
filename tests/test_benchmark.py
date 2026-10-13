import json

import pytest

from churnprop.explain import run_explain
from churnprop.propensity.benchmark import naive_baselines, run_benchmark
from churnprop.propensity.stepwise import backward_stepwise, run_feature_selection


def test_benchmark_two_models_small(small_split, cfg):
    res = run_benchmark(
        small_split, cfg, models=("logistic", "lightgbm"), regimes=("none", "smote")
    )
    # 2 models × 2 regimes × 3 folds × 1 repeat
    assert len(res.cv) == 2 * 2 * cfg.benchmark.n_folds * cfg.benchmark.n_repeats
    assert set(res.cv_summary["regime"]) == {"none", "smote"}
    assert list(res.test.columns).count("roc_auc") == 1
    assert res.test["roc_auc"].min() > 0.7
    assert len(res.test_scores) == len(small_split.X_test)
    assert set(res.calibration["model"]) == {"logistic", "lightgbm"}
    assert set(res.fitted) == {"logistic", "lightgbm"}


def test_naive_baselines(split):
    b = naive_baselines(split.y_test).set_index("model")
    assert b.loc["nobody_churns", "accuracy"] == pytest.approx(1 - split.y_test.mean())
    assert b.loc["nobody_churns", "roc_auc"] == 0.5


def test_backward_stepwise_drops_noise(small_split, cfg):
    import numpy as np

    from churnprop.features import fit_transform_frame, make_preprocessor

    X = small_split.X_train.copy()
    rng = np.random.default_rng(0)
    X["pure_noise"] = rng.normal(size=len(X))
    feats = list(X.columns)
    prep = make_preprocessor(cfg, feats)
    prep.transformers[0] = (
        "num",
        prep.transformers[0][1],
        prep.transformers[0][2] + ["pure_noise"],
    )
    Z = fit_transform_frame(prep, X)
    trace = backward_stepwise(Z, small_split.y_train, feats, alpha=0.05)
    assert "pure_noise" not in trace.kept
    assert "Contract" in trace.kept
    # tenure survives as the raw months or as the band (the two are collinear)
    assert {"Tenure Months", "tenure_band"} & set(trace.kept)


def test_feature_selection_table(small_split, cfg):
    t = run_feature_selection(small_split, cfg)
    assert set(t.columns) >= {"feature", "stepwise_kept", "l1_kept", "permutation_importance_auc"}
    assert t["stepwise_kept"].sum() >= 5
    assert abs(t.attrs["cv_auc_full"] - t.attrs["cv_auc_stepwise"]) < 0.03


def test_explain_folds_one_hot_back_to_raw(small_split, cfg):
    from churnprop.propensity.models import make_model

    m = make_model("lightgbm", cfg, resampling="none").fit(small_split.X_train, small_split.y_train)
    r = run_explain(small_split, m, cfg)
    assert list(r.values.columns) == list(small_split.X_test.columns)
    assert r.global_.iloc[0]["feature"] in {
        "Contract",
        "Tenure Months",
        "Internet Service",
        "Dependents",
    }
    assert abs(r.global_["share"].sum() - 1) < 1e-9
    assert set(r.by_segment["segment_col"]) <= set(cfg.explain.segments)
    # customer rows: one block per picked customer × every feature
    n_feat = small_split.X_test.shape[1]
    assert len(r.customers) % n_feat == 0


@pytest.mark.slow
def test_phase1_end_to_end(tmp_path, cfg):
    from churnprop.pipeline import run_phase1

    h = run_phase1(cfg, out_dir=tmp_path / "results", fig_dir=tmp_path / "figs")
    assert (tmp_path / "results" / "benchmark_test.csv").exists()
    assert (tmp_path / "results" / "headline_phase1.json").exists()
    assert len(list((tmp_path / "figs").glob("*.png"))) >= 8
    saved = json.loads((tmp_path / "results" / "headline_phase1.json").read_text())
    assert saved["holdout"]["roc_auc"] > 0.82
    assert h["nobody_churns_accuracy"] == pytest.approx(0.7346, abs=1e-3)
