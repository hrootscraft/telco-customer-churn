"""Load and clean the IBM Telco snapshot, and refuse to let post-outcome columns leak.

The raw file is the 33-column IBM Cognos sample ("Telco customer churn"), the same
7,043 customers as the 21-column Kaggle cut, plus geography and four columns that
are recorded *after* a customer has churned: ``Churn Label``, ``Churn Score``,
``Churn Reason`` and IBM's ``CLTV``. Those four are the classic leakage trap on
this dataset; :func:`assert_no_leakage` is called before any model sees a matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churnprop.config import Config


class LeakageError(ValueError):
    """Raised when a post-outcome or identifier column reaches a feature matrix."""


def load_raw(cfg: Config) -> pd.DataFrame:
    """Read the raw xlsx exactly as shipped (no cleaning)."""
    return pd.read_excel(cfg.raw_path)


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fix the two known quirks and type the columns; keep every column.

    * ``Total Charges`` is a string with 11 blank cells (customers with tenure 0, who
      have not been billed yet). We impute monthly charge × tenure (= 0 for them).
    * ``Senior Citizen`` is Yes/No in this version (0/1 in the Kaggle cut).
    """
    out = df.copy()
    tc = pd.to_numeric(out["Total Charges"].astype(str).str.strip(), errors="coerce")
    missing = tc.isna()
    tc[missing] = (out.loc[missing, "Monthly Charges"] * out.loc[missing, "Tenure Months"]).astype(
        float
    )
    out["Total Charges"] = tc.astype(float)
    out["Tenure Months"] = out["Tenure Months"].astype(int)
    out["Monthly Charges"] = out["Monthly Charges"].astype(float)
    out[cfg.data.target] = out[cfg.data.target].astype(int)
    for c in cfg.data.categorical:
        out[c] = out[c].astype(str)
    return out


def assert_no_leakage(columns, cfg: Config) -> None:
    """Fail loudly if any leakage, identifier or target column is in ``columns``."""
    forbidden = set(cfg.data.leakage_cols) | {cfg.data.id_col, cfg.data.target}
    bad = sorted(forbidden & set(columns))
    if bad:
        raise LeakageError(
            f"post-outcome / identifier columns in feature matrix: {bad}. "
            "CLTV may be used in the decision layer only."
        )


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    # Kept for the decision layer and reporting only; never features.
    aux_train: pd.DataFrame
    aux_test: pd.DataFrame


def make_split(df: pd.DataFrame, cfg: Config, feature_cols: list[str] | None = None) -> Split:
    """Stratified hold-out split. The test set is not touched until the final table."""
    from churnprop.features import add_engineered

    df = add_engineered(df)
    cols = list(feature_cols or cfg.data.feature_cols)
    assert_no_leakage(cols, cfg)
    X = df[cols]
    y = df[cfg.data.target]
    aux_cols = [c for c in cfg.data.leakage_cols if c in df.columns]
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)),
        test_size=cfg.benchmark.test_size,
        stratify=y,
        random_state=cfg.seed,
    )
    return Split(
        X_train=X.iloc[idx_tr].reset_index(drop=True),
        X_test=X.iloc[idx_te].reset_index(drop=True),
        y_train=y.iloc[idx_tr].reset_index(drop=True),
        y_test=y.iloc[idx_te].reset_index(drop=True),
        train_ids=df[cfg.data.id_col].iloc[idx_tr].reset_index(drop=True),
        test_ids=df[cfg.data.id_col].iloc[idx_te].reset_index(drop=True),
        aux_train=df[aux_cols].iloc[idx_tr].reset_index(drop=True),
        aux_test=df[aux_cols].iloc[idx_te].reset_index(drop=True),
    )


def load_clean(cfg: Config) -> pd.DataFrame:
    return clean(load_raw(cfg), cfg)


def churn_by(df: pd.DataFrame, col: str, cfg: Config) -> pd.DataFrame:
    """Churn rate, churners and customers by one column (used for the README facts)."""
    g = df.groupby(col)[cfg.data.target].agg(churn_rate="mean", churners="sum", customers="count")
    return g.sort_values("churn_rate", ascending=False)
