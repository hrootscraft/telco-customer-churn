"""Run the ``sql/*.sql`` feature families against the event log as of a date.

Every SQL file takes one parameter, ``{as_of}``, and filters its source table to
events dated on or before it *before* any window or aggregate runs. That is the
whole point-in-time rule, and ``tests/test_events.py::test_point_in_time`` checks
it by editing every event after the as-of date and asserting the features do not
move.

Engines: DuckDB (default; reads the hive-partitioned Parquet directly) or Spark
(``engine="spark"``; the same SQL through ``spark.sql``; PySpark is not a
dependency, it is imported only when asked for).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop.config import REPO_ROOT

log = logging.getLogger(__name__)

SQL_DIR = REPO_ROOT / "sql"
FAMILIES = ("billing", "usage", "ticket", "plan_change")
TABLES = {
    "billing": "billing",
    "usage": "usage",
    "ticket": "tickets",
    "plan_change": "plan_changes",
}

# The feature columns the SQL produces, with the fill used when a customer has no events.
EVENT_FEATURES: dict[str, float] = {
    "n_bills_visible": 0,
    "last_bill_amount": 0.0,
    "avg_bill_prior_6m": 0.0,
    "bill_shock": 0.0,
    "late_payments_90d": 0,
    "late_days_90d": 0,
    "late_rate_365d": 0.0,
    "late_streak": 0,
    "usage_gb_last_month": 0.0,
    "usage_gb_last_3m": 0.0,
    "streaming_hours_last_3m": 0.0,
    "usage_trend_3m": 0.0,
    "tickets_90d": 0,
    "tickets_365d": 0,
    "network_tickets_90d": 0,
    "days_since_last_ticket": 9999,
    "avg_csat_180d": 3.5,  # neutral score when there were no tickets
    "avg_resolution_days_180d": 0.0,
    "days_since_last_plan_change": 9999,
    "downgrades_180d": 0,
    "upgrades_180d": 0,
}


def feature_sql(family: str, as_of: str | pd.Timestamp) -> str:
    as_of = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    return (SQL_DIR / f"{family}_features.sql").read_text().replace("{as_of}", as_of)


def _duckdb_connection(events_root: Path | None, frames: dict[str, pd.DataFrame] | None):
    import duckdb

    con = duckdb.connect()
    for fam in FAMILIES:
        table = TABLES[fam]
        if frames is not None:
            con.register(table, frames[table])
        else:
            path = (Path(events_root) / table / "**" / "*.parquet").as_posix()
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}', hive_partitioning = true)"
            )
    return con


def build_event_features(
    as_of: str | pd.Timestamp,
    *,
    events_root: Path | str | None = REPO_ROOT / "data" / "events",
    frames: dict[str, pd.DataFrame] | None = None,
    customer_ids: pd.Series | np.ndarray | None = None,
    engine: str = "duckdb",
) -> pd.DataFrame:
    """One row per customer with every event feature as of ``as_of``.

    ``frames`` (dict of in-memory tables keyed ``billing``/``usage``/``tickets``/``plan_changes``)
    bypasses the Parquet store; tests use it. ``customer_ids`` fixes the output rows (customers
    with no visible events get the documented fills).
    """
    as_of = pd.Timestamp(as_of)
    parts = []
    if engine == "duckdb":
        con = _duckdb_connection(None if frames is not None else Path(events_root), frames)
        try:
            for fam in FAMILIES:
                parts.append(con.execute(feature_sql(fam, as_of)).df())
        finally:
            con.close()
    elif engine == "spark":
        parts = _spark_run(as_of, events_root, frames)
    else:
        raise ValueError(f"unknown engine {engine!r}")

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="customer_id", how="outer")
    if customer_ids is not None:
        base = pd.DataFrame({"customer_id": np.asarray(customer_ids)})
        out = base.merge(out, on="customer_id", how="left")
    for col, fill in EVENT_FEATURES.items():
        if col not in out.columns:
            out[col] = fill
        out[col] = out[col].fillna(fill)
    out["has_usage_history"] = (out["usage_gb_last_3m"] > 0).astype(int)
    out["as_of"] = as_of
    return out[["customer_id", "as_of", *EVENT_FEATURES, "has_usage_history"]]


def _spark_run(as_of, events_root, frames):
    """Same SQL via PySpark. Imported lazily so Spark is never a hard dependency."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("churnprop-features").getOrCreate()
    for fam in FAMILIES:
        table = TABLES[fam]
        if frames is not None:
            spark.createDataFrame(frames[table]).createOrReplaceTempView(table)
        else:
            spark.read.parquet(str(Path(events_root) / table)).createOrReplaceTempView(table)
    out = []
    for fam in FAMILIES:
        sql = (
            feature_sql(fam, as_of)
            .replace("INTERVAL 90 DAY", "INTERVAL 90 DAYS")
            .replace("INTERVAL 180 DAY", "INTERVAL 180 DAYS")
            .replace("INTERVAL 365 DAY", "INTERVAL 365 DAYS")
        )
        out.append(spark.sql(sql).toPandas())
    return out
