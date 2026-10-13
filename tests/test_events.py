import numpy as np
import pandas as pd
import pytest

from churnprop.events.simulate import EventsConfig, simulate_events
from churnprop.features_sql.build import EVENT_FEATURES, build_event_features


@pytest.fixture(scope="module")
def small_df(df):
    return df.sample(400, random_state=1).reset_index(drop=True)


@pytest.fixture(scope="module")
def events(small_df, cfg):
    return simulate_events(small_df, cfg, EventsConfig(months_after=3, signal=1.0, seed=3))


def _frames(ev):
    return {
        "billing": ev.billing,
        "usage": ev.usage,
        "tickets": ev.tickets,
        "plan_changes": ev.plan_changes,
    }


def test_one_bill_per_tenure_month_and_totals_match(events, small_df, cfg):
    snap = events.snapshot_date
    hist = events.billing[events.billing["bill_date"] <= snap]
    n = hist.groupby("customer_id").size()
    ten = small_df.set_index(cfg.data.id_col)["Tenure Months"]
    ten = ten[ten > 0]
    pd.testing.assert_series_equal(n.reindex(ten.index), ten.astype("int64"), check_names=False)
    total = hist.groupby("customer_id")["amount"].sum()
    expect = small_df.set_index(cfg.data.id_col)["Total Charges"].reindex(total.index)
    assert (total - expect).abs().max() < 0.5  # rounding to cents over up to 72 bills


def test_future_events_only_for_non_churners(events, small_df, cfg):
    fut = events.billing[events.billing["bill_date"] > events.snapshot_date]
    churned = set(small_df.loc[small_df[cfg.data.target] == 1, cfg.data.id_col])
    assert not (set(fut["customer_id"]) & churned)
    assert fut.groupby("customer_id").size().max() == 3


def test_payment_method_drives_lateness(events, small_df, cfg):
    late = events.billing.groupby("customer_id")["is_late"].mean()
    pm = small_df.set_index(cfg.data.id_col)["Payment Method"]
    auto = late[pm.reindex(late.index).str.contains("automatic")].mean()
    echeck = late[pm.reindex(late.index) == "Electronic check"].mean()
    assert echeck > auto * 3


def test_features_have_every_column_and_no_nulls(events, small_df, cfg):
    F = build_event_features(
        events.snapshot_date, frames=_frames(events), customer_ids=small_df[cfg.data.id_col]
    )
    assert len(F) == len(small_df)
    assert set(EVENT_FEATURES) <= set(F.columns)
    assert F[list(EVENT_FEATURES)].isna().sum().sum() == 0
    # a customer with no internet has no usage history
    no_net = small_df.loc[small_df["Internet Service"] == "No", cfg.data.id_col]
    assert (F.set_index("customer_id").loc[no_net, "has_usage_history"] == 0).all()


def test_point_in_time(events, small_df, cfg):
    """Editing every event dated after the as-of date must not move any feature."""
    as_of = events.snapshot_date - pd.Timedelta(days=45)
    ids = small_df[cfg.data.id_col]
    before = build_event_features(as_of, frames=_frames(events), customer_ids=ids)

    fr = {k: v.copy() for k, v in _frames(events).items()}
    rng = np.random.default_rng(0)
    m = fr["billing"]["bill_date"] > as_of
    fr["billing"].loc[m, "amount"] *= 10
    fr["billing"].loc[m, "is_late"] = True
    fr["billing"].loc[m, "late_days"] = 29
    m = fr["usage"]["month_end"] > as_of
    fr["usage"].loc[m, "data_gb"] = 0.0
    m = fr["tickets"]["ticket_date"] > as_of
    fr["tickets"].loc[m, "csat"] = 1
    fr["tickets"] = pd.concat(
        [
            fr["tickets"],
            pd.DataFrame(
                {
                    "customer_id": rng.choice(ids, 200),
                    "ticket_date": as_of + pd.Timedelta(days=1),
                    "category": "network",
                    "resolution_days": 1,
                    "csat": 1,
                }
            ),
        ],
        ignore_index=True,
    )
    fr["plan_changes"] = pd.concat(
        [
            fr["plan_changes"],
            pd.DataFrame(
                {
                    "customer_id": rng.choice(ids, 200),
                    "change_date": as_of + pd.Timedelta(days=2),
                    "change_type": "downgrade",
                }
            ),
        ],
        ignore_index=True,
    )
    after = build_event_features(as_of, frames=fr, customer_ids=ids)
    pd.testing.assert_frame_equal(before, after)


def test_as_of_moves_the_window(events, small_df, cfg):
    """Features as of an earlier date see fewer bills, never more."""
    ids = small_df[cfg.data.id_col]
    f0 = build_event_features(events.snapshot_date, frames=_frames(events), customer_ids=ids)
    f1 = build_event_features(
        events.snapshot_date - pd.Timedelta(days=120), frames=_frames(events), customer_ids=ids
    )
    assert (f1["n_bills_visible"] <= f0["n_bills_visible"]).all()
    assert (f0["n_bills_visible"] - f1["n_bills_visible"]).max() == 4


def test_signal_zero_removes_churn_gap(small_df, cfg):
    ev = simulate_events(small_df, cfg, EventsConfig(months_after=0, signal=0.0, seed=5))
    F = build_event_features(
        ev.snapshot_date, frames=_frames(ev), customer_ids=small_df[cfg.data.id_col]
    )
    y = small_df[cfg.data.target].to_numpy()
    gap = F.loc[y == 1, "tickets_90d"].mean() - F.loc[y == 0, "tickets_90d"].mean()
    # fibre-without-support customers churn more and raise more tickets, so a small
    # residual gap is real structure; the planted last-3-months spike is gone
    assert gap < 0.15


def test_attach_event_features_aligns_on_customer(events, small_df, cfg):
    from churnprop.data import make_split
    from churnprop.pipeline_events import EVENT_FEATURE_COLS, attach_event_features

    split = make_split(small_df, cfg)
    F = build_event_features(
        events.snapshot_date, frames=_frames(events), customer_ids=small_df[cfg.data.id_col]
    )
    s2 = attach_event_features(split, F)
    assert s2.X_train.shape[1] == split.X_train.shape[1] + len(EVENT_FEATURE_COLS)
    assert len(s2.X_test) == len(split.X_test)
    # row i of X_test belongs to test_ids[i]; check one feature round-trips
    i = 3
    cid = split.test_ids.iloc[i]
    expect = F.set_index("customer_id").loc[cid, "n_bills_visible"]
    assert s2.X_test["n_bills_visible"].iloc[i] == expect
