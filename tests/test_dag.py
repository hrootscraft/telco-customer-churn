"""The Airflow DAG parses and has the shape the README describes.

Runs only where Airflow is installed (the ``airflow-dag`` CI job); skipped elsewhere.
Run it alone with ``python -m pytest tests/test_dag.py --noconftest`` so the churnprop
fixtures are not imported.
"""

import os
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag(tmp_path_factory):
    os.environ.setdefault("AIRFLOW_HOME", str(tmp_path_factory.mktemp("airflow_home")))
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "True")
    try:
        from airflow.dag_processing.dagbag import DagBag  # Airflow 3
    except ImportError:  # pragma: no cover - Airflow 2
        from airflow.models.dagbag import DagBag

    return DagBag(dag_folder=str(DAGS))


def test_dag_imports_cleanly(dagbag):
    assert dagbag.import_errors == {}
    assert "retention_weekly" in dagbag.dags


def test_dag_shape(dagbag):
    dag = dagbag.dags["retention_weekly"]
    ids = set(dag.task_ids)
    assert ids == {
        "ingest_batch",
        "validate",
        "score_and_monitor",
        "decide",
        "retrain",
        "publish_plan",
    }
    assert dag.get_task("decide").downstream_task_ids == {"retrain", "publish_plan"}
    assert dag.get_task("retrain").downstream_task_ids == {"publish_plan"}
    assert dag.get_task("publish_plan").trigger_rule == "none_failed_min_one_success"
    assert dag.timetable is not None
    assert dag.catchup is False


def test_week_mapping_wraps():
    import importlib.util

    spec = importlib.util.spec_from_file_location("retention_weekly", DAGS / "retention_weekly.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from datetime import datetime

    assert mod.week_for(datetime(2024, 7, 8)) == 1
    assert mod.week_for(datetime(2024, 9, 23)) == 12
    assert mod.week_for(datetime(2024, 9, 30)) == 1
    assert 1 <= mod.week_for(datetime(2026, 10, 12)) <= 12
