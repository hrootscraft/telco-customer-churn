"""Shared loaders for the Streamlit app: the committed export, the snapshot, the results.

Everything the app reads is in the repository (``models/production/``, ``data/raw/``,
``results/``), so it runs on Streamlit Cloud from a clone with no registry server,
no event store and no database. Event features for the snapshot customers come from
``results/event_features_snapshot.parquet`` (phase 2's output); a customer typed into
the form or uploaded without them gets the documented default fills.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # Streamlit Cloud runs without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from churnprop.config import load_config  # noqa: E402
from churnprop.data import load_clean, make_split  # noqa: E402
from churnprop.features import add_engineered  # noqa: E402
from churnprop.features_sql.build import EVENT_FEATURES  # noqa: E402
from churnprop.offers.menu import load_offers  # noqa: E402
from churnprop.registry import ProductionModel, load_production  # noqa: E402
from churnprop.scoring import feature_frame, score_frame, top_reasons  # noqa: E402

# the dataviz palette used in the figures
BLUE, ORANGE, GREEN, AMBER, PINK, PURPLE, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#6250d6",
    "#e34948",
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"

EVENT_DEFAULTS = {**EVENT_FEATURES, "has_usage_history": 0}


@st.cache_resource(show_spinner=False)
def cfg():
    return load_config()


@st.cache_resource(show_spinner="Loading the production model…")
def model() -> ProductionModel:
    return load_production(cfg())


@st.cache_data(show_spinner="Loading customers…")
def snapshot() -> pd.DataFrame:
    """The cleaned snapshot with event features as of the snapshot date joined on."""
    c = cfg()
    df = load_clean(c)
    p = ROOT / "results" / "event_features_snapshot.parquet"
    if p.exists():
        ev = pd.read_parquet(p).set_index("customer_id")
        ev = ev[[k for k in EVENT_DEFAULTS if k in ev.columns]]
        df = df.join(ev, on=c.data.id_col)
    for k, v in EVENT_DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna(v)
    return add_engineered(df)


@st.cache_data(show_spinner=False)
def holdout_ids() -> set[str]:
    c = cfg()
    return set(make_split(load_clean(c), c).test_ids)


@st.cache_data(show_spinner="Scoring the hold-out book…")
def holdout_scored() -> pd.DataFrame:
    """Hold-out customers (never used in training) with their production score and CLTV."""
    c = cfg()
    df = snapshot()
    df = df[df[c.data.id_col].isin(holdout_ids())].reset_index(drop=True)
    s = score_frame(df, model(), id_col=c.data.id_col)
    out = df.copy()
    out["score"] = s["score"].to_numpy()
    out["decile"] = s["decile"].to_numpy()
    return out


def score_rows(
    df: pd.DataFrame, *, explain: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Score arbitrary rows (snapshot columns required; event features filled if absent)."""
    c = cfg()
    df = df.copy()
    for k, v in EVENT_DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna(v)
    if c.data.id_col not in df.columns:
        df[c.data.id_col] = [f"row-{i}" for i in range(len(df))]
    s = score_frame(df, model(), id_col=c.data.id_col)
    reasons = None
    if explain:
        X = feature_frame(df, model())
        reasons = top_reasons(X, model(), k=5)
    return s, reasons


def shap_for(df: pd.DataFrame) -> pd.DataFrame:
    """Folded SHAP values (one column per raw feature) for the given rows."""
    from churnprop.explain import shap_values_folded

    X = (
        feature_frame(add_engineered(df), model())
        if "tenure_band" not in df
        else feature_frame(df, model())
    )
    folded, base = shap_values_folded(model().pipeline, X)
    folded.attrs["base"] = base
    return folded


@st.cache_data(show_spinner=False)
def read_json(rel: str) -> dict:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def read_csv(rel: str, **kw) -> pd.DataFrame:
    p = ROOT / rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def offers():
    return load_offers()


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def status_badge(status: str) -> str:
    return {"ok": "🟢 ok", "warn": "🟡 warn", "alert": "🔴 alert"}.get(str(status), str(status))


def segment_label(row: pd.Series) -> str:
    return f"{row['Contract']} · {row['Internet Service']}"


def quantile_rank(score: float) -> float:
    s = holdout_scored()["score"].to_numpy()
    return float(np.mean(s <= score))
