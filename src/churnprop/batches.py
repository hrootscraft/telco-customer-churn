"""Weekly scoring batches simulated from the book, with drift injected on a schedule (D21).

A real programme scores a fresh cohort every week. This repository has one snapshot,
so each week is a bootstrap sample of the book with event features computed as of
that week (phase 2), and three kinds of drift injected from the weeks named in
``configs/default.yaml: batches.drift`` so the monitors have something to catch:

* ``price_rise``: a covariate shift (a feature's values move; the model has never seen
  them) - ``Monthly Charges`` × factor on the matching rows.
* ``mix_shift``: population drift (no row changes, the mix does) - matching rows are
  over-sampled by ``weight``.
* ``label_shift``: concept drift (features identical, outcome changes) - matching
  non-churners flip to churn with probability ``flip_to_churn``. Only the matured-label
  checks can see this one.

Each batch is written to ``data/batches/week_NN.parquet`` with its raw columns, the
event features as of the batch date, the label and the week the label becomes
available. Regenerable, not committed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop.config import REPO_ROOT, Config
from churnprop.data import load_clean
from churnprop.features_sql.build import build_event_features
from churnprop.pipeline_events import EVENT_FEATURE_COLS, EVENTS_ROOT, ensure_events

log = logging.getLogger(__name__)
BATCHES_ROOT = REPO_ROOT / "data" / "batches"


def batches_config(cfg: Config) -> dict:
    b = dict(cfg.extra.get("batches", {}))
    b.setdefault("n_weeks", 12)
    b.setdefault("size", 2000)
    b.setdefault("drift", [])
    b.setdefault("source", "holdout")
    return b


def batch_path(week: int, root: Path | str = BATCHES_ROOT) -> Path:
    return Path(root) / f"week_{week:02d}.parquet"


def _mask(df: pd.DataFrame, rule: dict) -> np.ndarray:
    return df.eval(rule["where"]).to_numpy(bool) if rule.get("where") else np.ones(len(df), bool)


def make_batch(
    cfg: Config,
    week: int,
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    *,
    events_root=EVENTS_ROOT,
) -> pd.DataFrame:
    b = batches_config(cfg)
    outcome_weeks = int(cfg.extra.get("production", {}).get("outcome_weeks", 4))
    rng = np.random.default_rng(cfg.seed * 1_000_003 + week)
    active = [r for r in b["drift"] if week >= int(r["from_week"])]

    weights = np.ones(len(df))
    for r in active:
        if r["kind"] == "mix_shift":
            weights[_mask(df, r)] *= float(r["weight"])
    idx = rng.choice(len(df), size=int(b["size"]), replace=True, p=weights / weights.sum())
    batch = df.iloc[idx].reset_index(drop=True)
    batch["drift_tags"] = ""

    for r in active:
        m = _mask(batch, r)
        if r["kind"] == "price_rise":
            batch.loc[m, "Monthly Charges"] = (
                batch.loc[m, "Monthly Charges"] * float(r["factor"])
            ).round(2)
        elif r["kind"] == "label_shift":
            flip = (
                m
                & (batch[cfg.data.target] == 0)
                & (rng.random(len(batch)) < float(r["flip_to_churn"]))
            )
            batch.loc[flip, cfg.data.target] = 1
            m = flip
        batch.loc[m, "drift_tags"] = (
            batch.loc[m, "drift_tags"].where(
                batch.loc[m, "drift_tags"] == "", batch.loc[m, "drift_tags"] + "|"
            )
            + r["kind"]
        )

    as_of = snapshot_date + pd.Timedelta(days=7 * week)
    F = build_event_features(
        as_of, events_root=events_root, customer_ids=batch[cfg.data.id_col].unique()
    )
    F = F.set_index("customer_id")[EVENT_FEATURE_COLS]
    batch = pd.concat(
        [batch, F.reindex(batch[cfg.data.id_col].to_numpy()).reset_index(drop=True)], axis=1
    )
    batch.insert(0, "batch_week", week)
    batch.insert(1, "batch_date", as_of)
    batch["churned"] = batch[cfg.data.target].astype(int)
    batch["label_available_week"] = week + outcome_weeks
    return batch.drop(columns=[cfg.data.target])


def make_batches(
    cfg: Config,
    root: Path | str = BATCHES_ROOT,
    *,
    force: bool = False,
    weeks: list[int] | None = None,
) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    b = batches_config(cfg)
    ev = ensure_events(cfg)
    df = load_clean(cfg)
    if b["source"] == "holdout":
        # only customers outside the training split, so the matured-label checks are honest
        from churnprop.data import make_split

        test_ids = set(make_split(df, cfg).test_ids)
        df = df[df[cfg.data.id_col].isin(test_ids)].reset_index(drop=True)
    out = []
    for w in weeks or range(1, int(b["n_weeks"]) + 1):
        p = batch_path(w, root)
        if p.exists() and not force:
            out.append(p)
            continue
        make_batch(cfg, w, df, ev.snapshot_date).to_parquet(p, index=False)
        out.append(p)
        log.info("batch week %2d -> %s", w, p.name)
    return out


def load_batch(week: int, root: Path | str = BATCHES_ROOT) -> pd.DataFrame:
    return pd.read_parquet(batch_path(week, root))


def validate_batch(batch: pd.DataFrame, cfg: Config, required_features: list[str]) -> dict:
    """Schema and sanity checks a DAG runs before scoring; raises on failure."""
    from churnprop.features import add_engineered

    problems = []
    X = add_engineered(batch)
    missing = [c for c in required_features if c not in X.columns]
    if missing:
        problems.append(f"missing features: {missing}")
    nulls = X[[c for c in required_features if c in X.columns]].isna().sum()
    if (nulls > 0).any():
        problems.append(f"nulls in {nulls[nulls > 0].to_dict()}")
    if len(batch) == 0:
        problems.append("empty batch")
    if (batch["Monthly Charges"] <= 0).any():
        problems.append("non-positive Monthly Charges")
    if batch[cfg.data.id_col].isna().any():
        problems.append("null customer id")
    if problems:
        raise ValueError("batch failed validation: " + "; ".join(problems))
    return {
        "n_rows": int(len(batch)),
        "n_unique_customers": int(batch[cfg.data.id_col].nunique()),
        "week": int(batch["batch_week"].iloc[0]),
    }
