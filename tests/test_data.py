import numpy as np
import pandas as pd
import pytest

from churnprop.data import LeakageError, assert_no_leakage, churn_by, make_split
from churnprop.features import ADDON_SERVICES, add_engineered, make_preprocessor


def test_shape_and_known_facts(df, cfg):
    assert df.shape[0] == 7043
    assert abs(df[cfg.data.target].mean() - 0.2654) < 1e-3
    by = churn_by(df, "Contract", cfg)
    # The corrected resume facts: 42.7% month-to-month vs 2.8% two-year.
    assert abs(by.loc["Month-to-month", "churn_rate"] - 0.427) < 1e-3
    assert abs(by.loc["Two year", "churn_rate"] - 0.028) < 1e-3


def test_total_charges_imputed(df):
    assert df["Total Charges"].dtype == float
    assert df["Total Charges"].isna().sum() == 0
    zero_tenure = df[df["Tenure Months"] == 0]
    assert len(zero_tenure) == 11
    assert (zero_tenure["Total Charges"] == 0).all()


def test_leakage_gate(cfg):
    with pytest.raises(LeakageError):
        assert_no_leakage(["Tenure Months", "Churn Score"], cfg)
    with pytest.raises(LeakageError):
        assert_no_leakage(["CLTV"], cfg)
    with pytest.raises(LeakageError):
        assert_no_leakage([cfg.data.id_col], cfg)
    assert_no_leakage(cfg.data.feature_cols, cfg)  # no raise


def test_split_is_stratified_and_disjoint(split, cfg):
    assert set(split.train_ids) & set(split.test_ids) == set()
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.01
    assert len(split.X_test) == pytest.approx(7043 * cfg.benchmark.test_size, abs=1)
    for c in cfg.data.leakage_cols:
        assert c not in split.X_train.columns
        assert c in split.aux_test.columns  # kept aside for the decision layer


def test_split_deterministic(df, cfg):
    a = make_split(df, cfg)
    b = make_split(df, cfg)
    assert a.test_ids.equals(b.test_ids)


def test_engineered_features(df):
    e = add_engineered(df)
    assert e["n_addon_services"].between(0, len(ADDON_SERVICES)).all()
    assert set(e["tenure_band"].unique()) <= {"0-6m", "7-12m", "13-24m", "25-48m", "49m+"}
    # avg monthly charge ≈ monthly charge for most customers
    ratio = (e["avg_monthly_charge"] / e["Monthly Charges"]).replace([np.inf], np.nan).dropna()
    assert ratio.median() == pytest.approx(1.0, abs=0.15)
    # idempotent
    pd.testing.assert_frame_equal(add_engineered(e), e)


def test_preprocessor_handles_unknown_category(split, cfg):
    prep = make_preprocessor(cfg)
    prep.fit(split.X_train)
    X = split.X_test.head(3).copy()
    X.loc[X.index[0], "Payment Method"] = "Crypto"
    Z = prep.transform(X)
    assert Z.shape[0] == 3 and not np.isnan(Z).any()
