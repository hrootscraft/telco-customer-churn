"""Airflow DAG: the weekly retention job (D23).

Every Monday 06:00: pull this week's batch → validate its schema → score it with
the committed production model → run the drift monitors → if they say retrain,
train a candidate and push it through the gate; otherwise go straight to publishing
the week's offer plan under the budget.

The tasks call the same ``churnprop`` functions the CLI and the GitHub Actions job
use, so the DAG is a schedule around tested code, not a second implementation.
Imports of ``churnprop`` are inside the tasks so the file parses (and the DagBag test
passes) on a scheduler that only has Airflow installed.

Run locally with the UI:  ``docker compose -f docker/airflow/docker-compose.yml up``
(``airflow standalone`` with this folder mounted as the DAG bag). CI validates the
file with ``DagBag`` in ``tests/test_dag.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task

SNAPSHOT = datetime(2024, 6, 30)
N_SIMULATED_WEEKS = 12


def week_for(run_date: datetime) -> int:
    """Map a run date onto one of the simulated batch weeks (wrapping), 1-based."""
    days = (run_date.replace(tzinfo=None) - SNAPSHOT).days
    return max(days // 7, 1) % N_SIMULATED_WEEKS or N_SIMULATED_WEEKS


@dag(
    dag_id="retention_weekly",
    description="Validate, score, monitor the weekly churn batch; retrain through the gate on drift; publish the offer plan",
    schedule="0 6 * * 1",
    start_date=datetime(2024, 7, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10), "owner": "retention-ds"},
    tags=["churnprop", "retention", "weekly"],
)
def retention_weekly():
    @task
    def ingest_batch(**context) -> int:
        from churnprop.batches import make_batches
        from churnprop.config import load_config

        week = week_for(context["logical_date"])
        make_batches(load_config(), weeks=[week])
        return week

    @task
    def validate(week: int) -> int:
        from churnprop.batches import load_batch, validate_batch
        from churnprop.config import load_config
        from churnprop.registry import load_production

        cfg = load_config()
        validate_batch(load_batch(week), cfg, load_production(cfg).features)
        return week

    @task
    def score_and_monitor(week: int) -> dict:
        from churnprop.config import load_config
        from churnprop.pipeline_prod import run_week

        r = run_week(load_config(), week)
        return {
            "week": week,
            "status": r["status"],
            "retrain": r["retrain"],
            "reasons": r["reasons"][:5],
        }

    @task.branch
    def decide(report: dict) -> str:
        return "retrain" if report["retrain"] else "publish_plan"

    @task
    def retrain(report: dict) -> dict:
        from churnprop.config import load_config
        from churnprop.registry import promote, train_candidate

        cfg = load_config()
        cand = train_candidate(
            cfg, note=f"retrain after week {report['week']}: {'; '.join(report['reasons'])}"
        )
        return promote(cfg, cand.version)

    @task(trigger_rule="none_failed_min_one_success")
    def publish_plan(report: dict) -> str:
        """Allocate this week's offers under the budget from the fresh scores and write the plan."""
        import numpy as np
        import pandas as pd

        from churnprop.config import REPO_ROOT, load_config
        from churnprop.offers.menu import load_offers
        from churnprop.uplift.allocate import allocate_budget

        cfg, ocfg = load_config(), load_offers()
        week = report["week"]
        scores = pd.read_parquet(REPO_ROOT / "results" / "scores" / f"week_{week:02d}.parquet")
        batch = pd.read_parquet(REPO_ROOT / "data" / "batches" / f"week_{week:02d}.parquet")
        value = batch[ocfg.value_column].to_numpy(float) * ocfg.value_margin
        costs = np.array(ocfg.costs)
        # risk x value x assumed save rate - cost, as in the loop's warm start; the uplift
        # learner takes over once the loop's log has matured outcomes (phase 3)
        ev = (
            scores["score"].to_numpy()[:, None] * ocfg.assumed_save_rate * value[:, None]
            - costs[None, :]
        )
        ev[:, 0] = 0
        alloc = allocate_budget(ev, costs, ocfg.weekly_budget)
        plan = scores[[cfg.data.id_col, "score"]].assign(
            offer=[ocfg.arms[k] for k in alloc.arm], cost=costs[alloc.arm], week=week
        )
        out = REPO_ROOT / "results" / "plans"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"week_{week:02d}.csv"
        plan[plan["offer"] != "control"].to_csv(path, index=False)
        return str(path)

    week = ingest_batch()
    checked = validate(week)
    report = score_and_monitor(checked)
    branch = decide(report)
    retrained = retrain(report)
    published = publish_plan(report)
    branch >> [retrained, published]
    retrained >> published


retention_weekly()
