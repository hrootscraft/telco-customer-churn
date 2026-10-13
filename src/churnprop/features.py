"""Feature engineering and the one preprocessing object every model shares.

The preprocessor is a :class:`~sklearn.compose.ColumnTransformer` that is fitted
*inside* each cross-validation fold as the first step of each model's pipeline, so
scaling statistics and one-hot categories never see validation rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnprop.config import Config

ADDON_SERVICES = (
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
)

TENURE_BANDS = [
    (-1, 6, "0-6m"),
    (6, 12, "7-12m"),
    (12, 24, "13-24m"),
    (24, 48, "25-48m"),
    (48, 999, "49m+"),
]


def tenure_band(months: pd.Series) -> pd.Series:
    bins = [b[0] for b in TENURE_BANDS] + [TENURE_BANDS[-1][1]]
    labels = [b[2] for b in TENURE_BANDS]
    return pd.cut(months, bins=bins, labels=labels).astype(str)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns (idempotent; safe to call on an already-engineered frame)."""
    out = df.copy()
    out["n_addon_services"] = sum((out[c] == "Yes").astype(int) for c in ADDON_SERVICES)
    out["avg_monthly_charge"] = np.where(
        out["Tenure Months"] > 0,
        out["Total Charges"] / out["Tenure Months"].clip(lower=1),
        out["Monthly Charges"],
    )
    out["tenure_band"] = tenure_band(out["Tenure Months"])
    return out


def make_preprocessor(
    cfg: Config, feature_cols: list[str] | None = None, *, scale: bool = True
) -> ColumnTransformer:
    """Scale numerics, one-hot categoricals (``drop='if_binary'`` keeps Yes/No to one column)."""
    cols = list(feature_cols or cfg.data.feature_cols)
    num = [c for c in cols if c in cfg.data.all_numeric]
    cat = [c for c in cols if c in cfg.data.all_categorical]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_tf, num),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                cat,
            ),
        ],
        verbose_feature_names_out=False,
    )


def feature_names(prep: ColumnTransformer) -> list[str]:
    return [str(n) for n in prep.get_feature_names_out()]


def fit_transform_frame(prep: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    Z = prep.fit_transform(X)
    return pd.DataFrame(Z, columns=feature_names(prep), index=X.index)


def transform_frame(prep: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(prep.transform(X), columns=feature_names(prep), index=X.index)


def preprocessing_only(cfg: Config, feature_cols: list[str] | None = None) -> Pipeline:
    """A pipeline with just the preprocessor, for code that needs the encoded matrix."""
    return Pipeline([("prep", make_preprocessor(cfg, feature_cols))])
