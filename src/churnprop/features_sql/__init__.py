"""Point-in-time feature store over the event log, in SQL (DuckDB by default, Spark optional)."""

from churnprop.features_sql.build import EVENT_FEATURES, build_event_features, feature_sql

__all__ = ["EVENT_FEATURES", "build_event_features", "feature_sql"]
