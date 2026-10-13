"""SHAP drivers at three altitudes: the book, a segment, one customer.

SHAP is computed on the LightGBM pipeline (``explain.model`` in the config) with a
``TreeExplainer`` on the *encoded* matrix, then one-hot columns are folded back to
their raw feature so marketing reads "Contract" rather than "Contract_Two year".

Three outputs:

* ``shap_global.csv``      mean |SHAP| per raw feature, plus the direction (sign of the
                           correlation between the feature's SHAP and its value)
* ``shap_by_segment.csv``  top drivers *within* each segment (e.g. fibre customers,
                           0-6 month tenure), which is what an offer owner needs
* ``shap_customers.csv``   the per-feature contributions for a few example customers
                           (waterfall data for the app)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from churnprop.config import Config
from churnprop.data import Split
from churnprop.features import add_engineered, feature_names, transform_frame

log = logging.getLogger(__name__)


def _raw_of(encoded: str, raw_features: list[str]) -> str:
    # longest raw prefix wins ("Online Security" before "Online")
    for f in sorted(raw_features, key=len, reverse=True):
        if encoded == f or encoded.startswith(f + "_"):
            return f
    return encoded


@dataclass
class ShapResult:
    values: pd.DataFrame  # customers × raw features (encoded columns folded)
    base_value: float
    global_: pd.DataFrame
    by_segment: pd.DataFrame
    customers: pd.DataFrame
    scores: pd.Series

    def save(self, out_dir) -> None:
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.global_.to_csv(out / "shap_global.csv", index=False)
        self.by_segment.to_csv(out / "shap_by_segment.csv", index=False)
        self.customers.to_csv(out / "shap_customers.csv", index=False)
        self.values.assign(base_value=self.base_value).to_parquet(out / "shap_values.parquet")


def shap_values_folded(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP of the pipeline's classifier on the encoded matrix, folded back to raw features."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    Z = transform_frame(prep, X)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Z)
    if isinstance(sv, list):  # older API returns [neg, pos]
        sv = sv[1]
    ev = explainer.expected_value
    base = float(ev[1] if np.ndim(ev) else ev)
    enc = pd.DataFrame(sv, columns=feature_names(prep), index=X.index)
    raw = list(X.columns)
    folded = enc.T.groupby(lambda c: _raw_of(c, raw)).sum().T
    return folded[[c for c in raw if c in folded.columns]], base


def run_explain(
    split: Split, pipeline, cfg: Config, *, df_test_full: pd.DataFrame | None = None
) -> ShapResult:
    X = split.X_test
    values, base = shap_values_folded(pipeline, X)
    p = pd.Series(pipeline.predict_proba(X)[:, 1], index=X.index, name="p_churn")

    # Global: mean |SHAP| and direction
    rows = []
    for f in values.columns:
        v = values[f]
        x = X[f]
        if pd.api.types.is_numeric_dtype(x):
            direction = float(np.corrcoef(x.astype(float), v)[0, 1]) if x.nunique() > 1 else 0.0
            level = None
        else:
            # which level pushes churn up the most
            m = v.groupby(x).mean().sort_values(ascending=False)
            direction = float("nan")
            level = f"{m.index[0]} (+{m.iloc[0]:.2f}) vs {m.index[-1]} ({m.iloc[-1]:+.2f})"
        rows.append(
            {
                "feature": f,
                "mean_abs_shap": float(v.abs().mean()),
                "corr_with_value": direction,
                "highest_risk_level": level,
            }
        )
    global_ = (
        pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    )
    global_["share"] = global_["mean_abs_shap"] / global_["mean_abs_shap"].sum()

    # By segment: top 5 drivers within each segment of each configured segment column
    seg_rows = []
    Xe = add_engineered(X) if any(s not in X.columns for s in cfg.explain.segments) else X
    for seg_col in cfg.explain.segments:
        if seg_col not in Xe.columns:
            continue
        for level, idx in Xe.groupby(seg_col).groups.items():
            sub = values.loc[idx]
            mean_signed = sub.mean().sort_values(key=np.abs, ascending=False)
            for rank, (f, s) in enumerate(mean_signed.head(5).items(), start=1):
                seg_rows.append(
                    {
                        "segment_col": seg_col,
                        "segment": level,
                        "n": len(idx),
                        "churn_rate": float(split.y_test.loc[idx].mean()),
                        "mean_p_churn": float(p.loc[idx].mean()),
                        "rank": rank,
                        "feature": f,
                        "mean_shap": float(s),
                        "mean_abs_shap": float(sub[f].abs().mean()),
                    }
                )
    by_segment = pd.DataFrame(seg_rows)

    # Example customers: highest-risk, median, lowest-risk
    order = p.sort_values(ascending=False).index
    k = cfg.explain.n_customers_waterfall
    picks = list(order[: max(1, k - 2)]) + [order[len(order) // 2], order[-1]]
    cust_rows = []
    for i in picks:
        for f in values.columns:
            cust_rows.append(
                {
                    "customer_id": split.test_ids.loc[i],
                    "p_churn": float(p.loc[i]),
                    "feature": f,
                    "value": X.loc[i, f],
                    "shap": float(values.loc[i, f]),
                }
            )
    customers = pd.DataFrame(cust_rows)
    return ShapResult(
        values=values,
        base_value=base,
        global_=global_,
        by_segment=by_segment,
        customers=customers,
        scores=p,
    )
