"""Batch scoring with the committed production model (D20).

``churnprop score --week N`` loads ``data/batches/week_NN.parquet`` and
``models/production/``, scores every row, and writes ``results/scores/week_NN.parquet``
with the score, its decile, the model version and, optionally, the top three SHAP
reasons per customer in plain words. The same function serves the Streamlit app and
the Airflow DAG.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop.config import REPO_ROOT, Config
from churnprop.features import add_engineered
from churnprop.registry import ProductionModel, load_production

log = logging.getLogger(__name__)
SCORES_ROOT = REPO_ROOT / "results" / "scores"


def feature_frame(df: pd.DataFrame, model: ProductionModel) -> pd.DataFrame:
    X = add_engineered(df)
    missing = [c for c in model.features if c not in X.columns]
    if missing:
        raise KeyError(f"batch is missing model features {missing}")
    return X[model.features]


def score_frame(
    df: pd.DataFrame, model: ProductionModel, *, explain: bool = False, id_col: str = "CustomerID"
) -> pd.DataFrame:
    X = feature_frame(df, model)
    p = model.pipeline.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            id_col: df[id_col].to_numpy(),
            "score": p,
            "decile": score_deciles(p),
            "model_version": model.version,
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    if "batch_week" in df:
        out.insert(1, "batch_week", df["batch_week"].to_numpy())
    if explain:
        out = pd.concat([out, top_reasons(X, model)], axis=1)
    return out


def score_deciles(p: np.ndarray) -> pd.Series:
    """Within-frame decile of each score, 1 = top 10%.

    A decile is a rank among the rows scored together, so it needs at least ten of
    them; with fewer (one customer described in the app, a tiny CSV) it is ``<NA>``
    rather than a number that would read as a population rank.
    """
    n = len(p)
    if n < 10:
        return pd.Series([pd.NA] * n, dtype="Int64")
    ranks = pd.Series(p).rank(method="first")
    return (10 - pd.qcut(ranks, 10, labels=False)).astype("Int64")


def top_reasons(X: pd.DataFrame, model: ProductionModel, k: int = 3) -> pd.DataFrame:
    """The ``k`` raw features pushing each customer's score up the most, as 'feature=value'."""
    from churnprop.explain import shap_values_folded

    folded, _ = shap_values_folded(model.pipeline, X)
    cols = {}
    order = np.argsort(-folded.to_numpy(), axis=1)[:, :k]
    feats = np.array(folded.columns)
    for j in range(k):
        f = feats[order[:, j]]
        vals = [X.iloc[i][f[i]] for i in range(len(X))]
        contrib = folded.to_numpy()[np.arange(len(X)), order[:, j]]
        cols[f"reason_{j + 1}"] = [
            f"{f[i]}={_fmt(vals[i])} ({contrib[i]:+.2f})" if contrib[i] > 0 else ""
            for i in range(len(X))
        ]
    return pd.DataFrame(cols, index=X.index).reset_index(drop=True)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}"
    return str(v)


def score_batch(
    cfg: Config,
    week: int,
    *,
    batches_root=None,
    out_dir: Path | str | None = None,
    explain: bool = False,
) -> Path:
    from churnprop.batches import BATCHES_ROOT, load_batch

    model = load_production(cfg)
    batch = load_batch(week, batches_root or BATCHES_ROOT)
    scored = score_frame(batch, model, explain=explain, id_col=cfg.data.id_col)
    out = Path(out_dir or SCORES_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"week_{week:02d}.parquet"
    scored.to_parquet(p, index=False)
    log.info(
        "scored week %d: %d rows, mean score %.3f, model v%d",
        week,
        len(scored),
        scored["score"].mean(),
        model.version,
    )
    return p
