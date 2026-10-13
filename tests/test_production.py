"""Phase 4: registry + gate, drifted batches, scoring, monitors, the weekly job."""

import json

import numpy as np
import pandas as pd
import pytest

from churnprop.batches import batches_config, make_batch, validate_batch
from churnprop.config import load_config
from churnprop.monitor import (
    MonitorConfig,
    feature_drift,
    ks_from_quantiles,
    outcome_drift,
    psi,
    reference_stats,
    score_drift,
)
from churnprop.pipeline_events import ensure_events
from churnprop.registry import (
    ProductionConfig,
    gate,
    load_production,
    promote,
    rollback,
    train_candidate,
)
from churnprop.scoring import score_frame

# ----------------------------------------------------------------------------- statistics


def test_psi_zero_on_identical_and_grows_with_shift():
    ref = np.array([0.1] * 10)
    assert psi(ref, ref) == pytest.approx(0.0)
    shifted = np.array([0.02] * 5 + [0.18] * 5)
    assert psi(ref, shifted) > 0.25
    assert psi(ref, np.array([0.08] * 5 + [0.12] * 5)) < 0.1


def test_ks_handles_ties():
    rng = np.random.default_rng(0)
    ref = rng.poisson(0.3, 5000)  # mostly zeros
    q = np.quantile(ref, np.linspace(0, 1, 21))
    cdf = [np.mean(ref <= v) for v in q]
    same = rng.poisson(0.3, 2000)
    assert ks_from_quantiles(q, same, cdf) < 0.05
    more = rng.poisson(1.5, 2000)
    assert ks_from_quantiles(q, more, cdf) > 0.3


def test_reference_and_feature_drift(cfg, small_split):
    X = small_split.X_train
    ref = reference_stats(X, np.full(len(X), 0.3), small_split.y_train.to_numpy(), cfg)
    mcfg = MonitorConfig.from_config(cfg)
    same = feature_drift(X, ref, mcfg)
    assert (same["status"] == "ok").all()
    assert same["psi"].max() < 0.02
    moved = X.copy()
    moved["Monthly Charges"] = moved["Monthly Charges"] * 1.6
    moved["Contract"] = "Two year"
    fd = feature_drift(moved, ref, mcfg).set_index("feature")
    assert fd.loc["Monthly Charges", "status"] == "alert"
    assert fd.loc["Contract", "status"] == "alert"
    assert fd.loc["Gender", "status"] == "ok"


def test_score_and_outcome_drift(cfg, small_split):
    X = small_split.X_train
    y = small_split.y_train.to_numpy()
    rng = np.random.default_rng(1)
    s_ref = rng.beta(2, 5, len(X))
    ref = reference_stats(X, s_ref, y, cfg)
    mcfg = MonitorConfig.from_config(cfg)
    assert score_drift(rng.beta(2, 5, 3000), ref, mcfg)["status"] == "ok"
    assert score_drift(rng.beta(5, 2, 3000), ref, mcfg)["status"] == "alert"
    card = {"holdout": {"pr_auc": 0.65}}
    p = np.clip(y * 0.5 + rng.random(len(y)) * 0.4, 0, 1)
    ok = outcome_drift(y, p, card, mcfg)
    assert ok["pr_auc"] > 0.8
    # labels flipped up in a segment: observed churn far above predicted
    y2 = y.copy()
    y2[rng.random(len(y)) < 0.4] = 1
    bad = outcome_drift(y2, p, card, mcfg)
    assert bad["status"] == "alert" and bad["churn_rel_gap"] > 0.3


# ----------------------------------------------------------------------------- gate


def test_gate_rules():
    pcfg = ProductionConfig.from_config(load_config())
    cand = {"holdout": {"pr_auc": 0.70, "ece": 0.02}}
    assert gate(cand, None, pcfg)["promote"]
    prod = {"version": 1, "holdout": {"pr_auc": 0.71, "ece": 0.02}}
    assert gate(cand, prod, pcfg)["promote"] is False
    assert "trails production" in gate(cand, prod, pcfg)["reasons"][0]
    assert gate({"holdout": {"pr_auc": 0.709, "ece": 0.02}}, prod, pcfg)["promote"]
    assert gate({"holdout": {"pr_auc": 0.75, "ece": 0.09}}, prod, pcfg)["promote"] is False


# ----------------------------------------------------------------------------- registry e2e


@pytest.fixture(scope="module")
def tmp_registry(tmp_path_factory):
    d = tmp_path_factory.mktemp("registry")
    cfg = load_config(
        fast=True,
        seed=3,
        production={
            "tracking_uri": f"sqlite:///{d / 'mlflow.db'}",
            "export_dir": str(d / "production"),
            "model": "lightgbm",
        },
    )
    return cfg, d


@pytest.mark.slow
def test_train_promote_gate_rollback(tmp_registry, monkeypatch):
    cfg, d = tmp_registry
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    pcfg = ProductionConfig.from_config(cfg)
    c1 = train_candidate(cfg, pcfg, note="test v1")
    d1 = promote(cfg, c1.version, pcfg=pcfg)
    assert d1["promote"] and d1["production_after"] == c1.version
    prod = load_production(export_dir=pcfg.export_dir)
    assert prod.version == c1.version
    assert set(prod.features) == set(c1.card["features"])
    assert (pcfg.export_dir / "model_card.json").exists()
    card = json.loads((pcfg.export_dir / "model_card.json").read_text())
    assert card["holdout"]["pr_auc"] == pytest.approx(c1.metrics["pr_auc"])
    assert card["gate"]["promote"] is True

    # a deliberately weaker candidate: snapshot features only
    from dataclasses import replace

    c2 = train_candidate(cfg, replace(pcfg, use_event_features=False), note="test v2")
    d2 = promote(cfg, c2.version, pcfg=pcfg)
    assert d2["promote"] is False
    assert load_production(export_dir=pcfg.export_dir).version == c1.version
    # forced promotion then rollback
    d3 = promote(cfg, c2.version, force=True, pcfg=pcfg)
    assert d3["forced"] and load_production(export_dir=pcfg.export_dir).version == c2.version
    r = rollback(cfg, pcfg)
    assert r["production_after"] == c1.version
    assert load_production(export_dir=pcfg.export_dir).version == c1.version


# ----------------------------------------------------------------------------- batches + weekly


@pytest.fixture(scope="module")
def small_df(cfg, df):
    return df.sample(600, random_state=0).reset_index(drop=True)


def test_batches_inject_drift_on_schedule(cfg, small_df):
    ev = ensure_events(cfg)
    b = batches_config(cfg)
    rules = {r["kind"]: r for r in b["drift"]}
    early = make_batch(cfg, 1, small_df, ev.snapshot_date)
    late = make_batch(cfg, int(b["n_weeks"]), small_df, ev.snapshot_date)
    assert len(early) == len(late) == b["size"]
    assert (early["drift_tags"] == "").all()
    assert late["drift_tags"].str.contains("price_rise").any()
    # price rise: fibre bills up by the factor, others untouched
    base = small_df.set_index(cfg.data.id_col)["Monthly Charges"]
    fibre = late[late["Internet Service"] == "Fiber optic"]
    ratio = fibre["Monthly Charges"].to_numpy() / base.loc[fibre[cfg.data.id_col]].to_numpy()
    assert np.allclose(ratio, rules["price_rise"]["factor"], atol=0.01)
    dsl = late[late["Internet Service"] == "DSL"]
    assert np.allclose(dsl["Monthly Charges"].to_numpy(), base.loc[dsl[cfg.data.id_col]].to_numpy())
    # mix shift: more fibre than at week 1
    assert (late["Internet Service"] == "Fiber optic").mean() > (
        early["Internet Service"] == "Fiber optic"
    ).mean() + 0.05
    # label shift: churn up in the named segment only
    seg = late.eval(rules["label_shift"]["where"])
    seg_early = early.eval(rules["label_shift"]["where"])
    assert late.loc[seg, "churned"].mean() > early.loc[seg_early, "churned"].mean() + 0.05
    assert late["label_available_week"].iloc[0] == b["n_weeks"] + 4
    assert "batch_date" in late and (late["batch_week"] == b["n_weeks"]).all()


def test_validate_batch_catches_problems(cfg, small_df):
    ev = ensure_events(cfg)
    batch = make_batch(cfg, 2, small_df, ev.snapshot_date)
    feats = list(cfg.data.feature_cols)
    assert validate_batch(batch, cfg, feats)["n_rows"] == len(batch)
    broken = batch.copy()
    broken.loc[0, "Monthly Charges"] = -1
    with pytest.raises(ValueError, match="Monthly Charges"):
        validate_batch(broken, cfg, feats)
    with pytest.raises(ValueError, match="missing"):
        validate_batch(batch, cfg, feats + ["not_a_feature"])


@pytest.mark.slow
def test_weekly_job_end_to_end(tmp_registry, cfg, small_df, tmp_path, monkeypatch):
    """Train+promote into a temp export, score drifted batches, and see the monitors react."""
    from churnprop.pipeline_prod import next_week, run_week

    rcfg, d = tmp_registry
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    pcfg = ProductionConfig.from_config(rcfg)
    if not (pcfg.export_dir / "model.joblib").exists():
        c = train_candidate(rcfg, pcfg)
        promote(rcfg, c.version, pcfg=pcfg)
    ev = ensure_events(cfg)
    broot, sroot, mroot = tmp_path / "batches", tmp_path / "scores", tmp_path / "monitoring"
    broot.mkdir()
    n = int(batches_config(rcfg)["n_weeks"])
    for w in (1, n):
        make_batch(rcfg, w, small_df, ev.snapshot_date).to_parquet(broot / f"week_{w:02d}.parquet")
    assert next_week(rcfg, mroot) == 1
    r1 = run_week(rcfg, 1, batches_root=broot, scores_root=sroot, monitoring_root=mroot)
    assert r1["status"] in ("ok", "warn")
    assert next_week(rcfg, mroot) == 2
    rn = run_week(rcfg, n, batches_root=broot, scores_root=sroot, monitoring_root=mroot)
    assert rn["status"] == "alert" and rn["retrain"]
    top = {f["feature"] for f in rn["top_drifted"]}
    assert "Monthly Charges" in top or "Internet Service" in top
    assert (mroot / f"week_{n:02d}.md").exists() and (mroot / "latest.json").exists()
    summary = pd.read_csv(mroot / "summary.csv")
    assert list(summary["week"]) == [1, n]
    scored = pd.read_parquet(sroot / f"week_{n:02d}.parquet")
    assert scored["score"].between(0, 1).all() and scored["model_version"].nunique() == 1


def test_score_frame_schema(tmp_registry, cfg, small_df):
    rcfg, d = tmp_registry
    pcfg = ProductionConfig.from_config(rcfg)
    if not (pcfg.export_dir / "model.joblib").exists():
        pytest.skip("needs the slow registry test to have exported a model")
    model = load_production(export_dir=pcfg.export_dir)
    ev = ensure_events(cfg)
    batch = make_batch(rcfg, 3, small_df, ev.snapshot_date).head(200)
    s = score_frame(batch, model, explain=True, id_col=cfg.data.id_col)
    assert list(s.columns[:3]) == [cfg.data.id_col, "batch_week", "score"]
    assert s["decile"].between(1, 10).all()
    assert s["reason_1"].str.len().gt(0).mean() > 0.9
    assert (s["model_version"] == model.version).all()
