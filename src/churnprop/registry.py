"""Model registry: MLflow for lineage, a committed export for serving (D19, D20).

Two stores, on purpose:

* **MLflow** (tracking + Model Registry on a local SQLite file) records every training
  run: parameters, hold-out metrics, the data hash, the git commit, the model file and
  the monitoring reference statistics. Versions carry aliases ``candidate``,
  ``production`` and ``previous``.
* **``models/production/``** is the promoted version exported as plain files
  (``model.joblib``, ``model_card.json``, ``reference_stats.json``) and committed, so
  batch scoring, the Airflow DAG, the GitHub Actions job and the Streamlit app load
  the model without an MLflow server or database. Promotion is the only path that
  writes there, and it runs a gate first.

``churnprop train`` registers a candidate; ``churnprop promote`` gates and exports it;
``churnprop promote --rollback`` swaps ``production`` and ``previous``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnprop.config import REPO_ROOT, Config
from churnprop.data import load_clean, make_split
from churnprop.explain import shap_values_folded
from churnprop.monitor import reference_stats
from churnprop.pipeline_events import (
    EVENT_FEATURE_COLS,
    EVENTS_ROOT,
    attach_event_features,
    ensure_events,
)
from churnprop.propensity.metrics import score_all
from churnprop.propensity.models import PRETTY, make_model

log = logging.getLogger(__name__)

ALIAS_PROD, ALIAS_PREV, ALIAS_CAND = "production", "previous", "candidate"


# ----------------------------------------------------------------------------- config view


@dataclass(frozen=True)
class ProductionConfig:
    model: str
    model_name: str
    use_event_features: bool
    tracking_uri: str
    export_dir: Path
    outcome_weeks: int
    min_pr_auc_delta: float
    max_ece: float

    @classmethod
    def from_config(cls, cfg: Config) -> ProductionConfig:
        p = cfg.extra.get("production", {})
        gate = p.get("gate", {})
        export = Path(p.get("export_dir", "models/production"))
        uri = p.get("tracking_uri", "sqlite:///mlflow.db")
        if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///") :]).is_absolute():
            uri = f"sqlite:///{REPO_ROOT / uri[len('sqlite:///') :]}"
        return cls(
            model=p.get("model", "xgboost"),
            model_name=p.get("model_name", "churnprop-propensity"),
            use_event_features=bool(p.get("use_event_features", True)),
            tracking_uri=uri,
            export_dir=export if export.is_absolute() else REPO_ROOT / export,
            outcome_weeks=int(p.get("outcome_weeks", 4)),
            min_pr_auc_delta=float(gate.get("min_pr_auc_delta", -0.005)),
            max_ece=float(gate.get("max_ece", 0.05)),
        )


def production_feature_cols(cfg: Config, pcfg: ProductionConfig | None = None) -> list[str]:
    pcfg = pcfg or ProductionConfig.from_config(cfg)
    cols = list(cfg.data.feature_cols)
    if pcfg.use_event_features:
        cols += EVENT_FEATURE_COLS
    return cols


# ----------------------------------------------------------------------------- lineage helpers


def data_hash(cfg: Config) -> str:
    h = hashlib.sha256()
    with open(cfg.raw_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ----------------------------------------------------------------------------- training


@dataclass
class Candidate:
    version: int
    run_id: str
    metrics: dict
    card: dict
    pipeline: object
    reference: dict


def training_split(cfg: Config, pcfg: ProductionConfig):
    """The phase-1 split, with event features as of the snapshot joined on (phase 2)."""
    from churnprop.features_sql.build import build_event_features

    df = load_clean(cfg)
    split = make_split(df, cfg)
    if not pcfg.use_event_features:
        return split
    ev = ensure_events(cfg)
    F = build_event_features(
        ev.snapshot_date, events_root=EVENTS_ROOT, customer_ids=df[cfg.data.id_col]
    )
    return attach_event_features(split, F)


def train_candidate(
    cfg: Config, pcfg: ProductionConfig | None = None, *, note: str = ""
) -> Candidate:
    """Fit the production pipeline, score the hold-out, log and register it as ``candidate``."""
    import mlflow
    from mlflow import MlflowClient

    pcfg = pcfg or ProductionConfig.from_config(cfg)
    split = training_split(cfg, pcfg)
    cols = list(split.X_train.columns)
    pipe = make_model(pcfg.model, cfg, feature_cols=cols, resampling="none")
    pipe.fit(split.X_train, split.y_train)
    p_test = pipe.predict_proba(split.X_test)[:, 1]
    metrics = score_all(
        split.y_test,
        p_test,
        top_frac=cfg.benchmark.top_decile,
        precision_floor=cfg.benchmark.precision_floor,
        n_bins=cfg.benchmark.calibration_bins,
    )
    p_train = pipe.predict_proba(split.X_train)[:, 1]
    ref = reference_stats(split.X_train, p_train, split.y_train.to_numpy(), cfg)

    top_drivers: list[dict] = []
    if pcfg.model in ("xgboost", "lightgbm", "random_forest", "gradient_boosting"):
        try:
            sample = split.X_train.sample(min(1500, len(split.X_train)), random_state=cfg.seed)
            folded, _ = shap_values_folded(pipe, sample)
            imp = folded.abs().mean().sort_values(ascending=False)
            top_drivers = [
                {"feature": k, "share": float(v / imp.sum())} for k, v in imp.head(8).items()
            ]
        except Exception as exc:  # pragma: no cover - SHAP is a nice-to-have on the card
            log.warning("SHAP on the candidate failed: %s", exc)

    card = {
        "model_name": pcfg.model_name,
        "family": pcfg.model,
        "family_pretty": PRETTY.get(pcfg.model, pcfg.model),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "data_hash": data_hash(cfg),
        "seed": cfg.seed,
        "fast": cfg.fast,
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
        "train_churn_rate": float(split.y_train.mean()),
        "features": cols,
        "n_features": len(cols),
        "event_features": pcfg.use_event_features,
        "holdout": {k: float(v) for k, v in metrics.items()},
        "top_drivers": top_drivers,
        "note": note,
        "intended_use": (
            "Rank active customers by 4-week churn risk for the weekly offer loop; the score is "
            "multiplied by CLTV in the decision layer, so calibration matters as much as ranking."
        ),
        "limitations": (
            "Trained on the IBM snapshot plus a synthetic event log (D9). Event features are "
            "computed as of the scoring date; the model has not seen prices or mixes outside "
            "the training book, which the drift monitors watch for."
        ),
    }

    mlflow.set_tracking_uri(pcfg.tracking_uri)
    mlflow.set_experiment("churnprop-propensity")
    with mlflow.start_run(run_name=f"{pcfg.model}-{card['git_sha']}") as run:
        mlflow.log_params(
            {
                "family": pcfg.model,
                "n_features": len(cols),
                "event_features": pcfg.use_event_features,
                "seed": cfg.seed,
                "fast": cfg.fast,
                "data_hash": card["data_hash"],
                "git_sha": card["git_sha"],
            }
        )
        mlflow.log_metrics({f"holdout_{k}": float(v) for k, v in metrics.items()})
        mlflow.log_dict(card, "model_card.json")
        mlflow.log_dict(ref, "reference_stats.json")
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            registered_model_name=pcfg.model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        version = int(info.registered_model_version)
    client = MlflowClient(tracking_uri=pcfg.tracking_uri)
    client.set_registered_model_alias(pcfg.model_name, ALIAS_CAND, str(version))
    for k, v in (("git_sha", card["git_sha"]), ("data_hash", card["data_hash"])):
        client.set_model_version_tag(pcfg.model_name, str(version), k, v)
    card["version"] = version
    card["run_id"] = run.info.run_id
    log.info(
        "candidate v%d: hold-out PR-AUC %.3f ROC-AUC %.3f ECE %.3f",
        version,
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["ece"],
    )
    return Candidate(version, run.info.run_id, metrics, card, pipe, ref)


# ----------------------------------------------------------------------------- promotion


def _version_card(client, pcfg: ProductionConfig, version: int) -> dict:
    import mlflow

    mv = client.get_model_version(pcfg.model_name, str(version))
    path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model_card.json", tracking_uri=pcfg.tracking_uri
    )
    with open(path) as f:
        card = json.load(f)
    card["version"], card["run_id"] = version, mv.run_id
    return card


def _alias_version(client, pcfg: ProductionConfig, alias: str) -> int | None:
    try:
        return int(client.get_model_version_by_alias(pcfg.model_name, alias).version)
    except Exception:
        return None


def gate(candidate_card: dict, production_card: dict | None, pcfg: ProductionConfig) -> dict:
    """Decide whether a candidate may replace production; returns the decision and why."""
    c = candidate_card["holdout"]
    checks = {"ece_ok": c["ece"] <= pcfg.max_ece}
    reasons = []
    if not checks["ece_ok"]:
        reasons.append(f"ECE {c['ece']:.3f} above {pcfg.max_ece}")
    if production_card is None:
        checks["pr_auc_ok"] = True
    else:
        p = production_card["holdout"]
        delta = c["pr_auc"] - p["pr_auc"]
        checks["pr_auc_ok"] = delta >= pcfg.min_pr_auc_delta
        checks["pr_auc_delta"] = float(delta)
        if not checks["pr_auc_ok"]:
            reasons.append(
                f"PR-AUC {c['pr_auc']:.3f} trails production v{production_card['version']} "
                f"({p['pr_auc']:.3f}) by {-delta:.3f}, more than {-pcfg.min_pr_auc_delta}"
            )
    return {
        "promote": all(v for k, v in checks.items() if k.endswith("_ok")),
        "checks": checks,
        "reasons": reasons,
    }


def export_version(pcfg: ProductionConfig, version: int, card: dict | None = None) -> Path:
    """Write the registry version to ``export_dir`` as plain files."""
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=pcfg.tracking_uri)
    card = card or _version_card(client, pcfg, version)
    mv = client.get_model_version(pcfg.model_name, str(version))
    mlflow.set_tracking_uri(pcfg.tracking_uri)
    pipe = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model")
    ref_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="reference_stats.json", tracking_uri=pcfg.tracking_uri
    )
    out = pcfg.export_dir
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    with open(ref_path) as f:
        ref = json.load(f)
    with open(out / "reference_stats.json", "w") as f:
        json.dump(ref, f)
    card = {
        **card,
        "version": version,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    (out / "VERSION").write_text(f"{version}\n")
    return out


def promote(
    cfg: Config,
    version: int | None = None,
    *,
    force: bool = False,
    pcfg: ProductionConfig | None = None,
) -> dict:
    """Gate the candidate (or ``version``) against production; on pass, alias + export it."""
    from mlflow import MlflowClient

    pcfg = pcfg or ProductionConfig.from_config(cfg)
    client = MlflowClient(tracking_uri=pcfg.tracking_uri)
    version = version or _alias_version(client, pcfg, ALIAS_CAND)
    if version is None:
        raise RuntimeError("no candidate to promote; run `churnprop train` first")
    prod_v = _alias_version(client, pcfg, ALIAS_PROD)
    cand_card = _version_card(client, pcfg, version)
    prod_card = _version_card(client, pcfg, prod_v) if prod_v is not None else None
    decision = gate(cand_card, prod_card, pcfg)
    decision.update(candidate=version, production_before=prod_v, forced=False)
    if not decision["promote"] and not force:
        log.warning("gate rejected v%d: %s", version, "; ".join(decision["reasons"]))
        client.set_model_version_tag(pcfg.model_name, str(version), "gate", "rejected")
        return decision
    if not decision["promote"]:
        decision["forced"] = True
    if prod_v is not None and prod_v != version:
        client.set_registered_model_alias(pcfg.model_name, ALIAS_PREV, str(prod_v))
    client.set_registered_model_alias(pcfg.model_name, ALIAS_PROD, str(version))
    client.set_model_version_tag(pcfg.model_name, str(version), "gate", "passed")
    cand_card["gate"] = decision
    export_version(pcfg, version, cand_card)
    decision["production_after"] = version
    log.info("promoted v%d to production (was v%s)", version, prod_v)
    return decision


def rollback(cfg: Config, pcfg: ProductionConfig | None = None) -> dict:
    from mlflow import MlflowClient

    pcfg = pcfg or ProductionConfig.from_config(cfg)
    client = MlflowClient(tracking_uri=pcfg.tracking_uri)
    prod_v, prev_v = (
        _alias_version(client, pcfg, ALIAS_PROD),
        _alias_version(client, pcfg, ALIAS_PREV),
    )
    if prev_v is None:
        raise RuntimeError("nothing to roll back to")
    client.set_registered_model_alias(pcfg.model_name, ALIAS_PROD, str(prev_v))
    if prod_v is not None:
        client.set_registered_model_alias(pcfg.model_name, ALIAS_PREV, str(prod_v))
    export_version(pcfg, prev_v)
    log.info("rolled back production v%s -> v%d", prod_v, prev_v)
    return {"production_before": prod_v, "production_after": prev_v}


# ----------------------------------------------------------------------------- serving side


@dataclass
class ProductionModel:
    pipeline: object
    card: dict
    reference: dict

    @property
    def version(self) -> int:
        return int(self.card["version"])

    @property
    def features(self) -> list[str]:
        return list(self.card["features"])


def load_production(
    cfg: Config | None = None, export_dir: Path | str | None = None
) -> ProductionModel:
    """Load the committed export; no MLflow needed."""
    if export_dir is None:
        export_dir = ProductionConfig.from_config(cfg).export_dir
    d = Path(export_dir)
    if not (d / "model.joblib").exists():
        raise FileNotFoundError(
            f"no production model in {d}; run `churnprop train` then `churnprop promote`"
        )
    with open(d / "model_card.json") as f:
        card = json.load(f)
    with open(d / "reference_stats.json") as f:
        ref = json.load(f)
    return ProductionModel(joblib.load(d / "model.joblib"), card, ref)


def registry_table(cfg: Config, pcfg: ProductionConfig | None = None) -> pd.DataFrame:
    """Every registered version with its aliases and hold-out metrics (for the app and README)."""
    from mlflow import MlflowClient

    pcfg = pcfg or ProductionConfig.from_config(cfg)
    client = MlflowClient(tracking_uri=pcfg.tracking_uri)
    rows = []
    aliases = {a: _alias_version(client, pcfg, a) for a in (ALIAS_PROD, ALIAS_PREV, ALIAS_CAND)}
    for mv in client.search_model_versions(f"name='{pcfg.model_name}'"):
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        rows.append(
            {
                "version": int(mv.version),
                "aliases": ",".join(a for a, v in aliases.items() if v == int(mv.version)),
                "gate": mv.tags.get("gate", ""),
                "git_sha": mv.tags.get("git_sha", ""),
                "pr_auc": m.get("holdout_pr_auc", np.nan),
                "roc_auc": m.get("holdout_roc_auc", np.nan),
                "ece": m.get("holdout_ece", np.nan),
                "created": pd.Timestamp(mv.creation_timestamp, unit="ms").strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)
