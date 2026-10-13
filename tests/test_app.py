"""The Streamlit app renders all three pages headless without raising (AppTest)."""

import sys
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "production" / "model.joblib").exists(),
    reason="needs the committed production export",
)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=600)
    at.query_params["page"] = page  # read by the app to pick the page under test
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_score_page_renders_and_scores_a_holdout_customer():
    at = _run("score")
    assert at.title[0].value == "Score a customer"
    assert at.selectbox  # the customer picker
    metrics = [m.label for m in at.metric]
    assert "4-week churn score" in metrics


def test_offers_page_allocates_within_budget():
    at = _run("offers")
    assert at.title[0].value == "This week's offer plan"
    labels = {m.label: m.value for m in at.metric}
    spend = float(labels["Spend"].replace("$", "").replace(",", ""))
    budget = float(labels["Spend"] and at.sidebar.slider[0].value)
    assert spend <= budget + 1e-6
    assert int(labels["Customers offered"].replace(",", "")) > 0


def test_monitoring_page_shows_latest_verdict():
    at = _run("monitoring")
    assert at.title[0].value == "Monitoring"
    labels = {m.label: m.value for m in at.metric}
    assert "Latest verdict" in labels
    assert "Retrain recommended" in labels


def test_score_page_scores_one_described_customer():
    at = _run("score")
    at.radio[0].set_value("A customer I describe").run()
    assert not at.exception, [e.value for e in at.exception]
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    metrics = [m.label for m in at.metric]
    assert "4-week churn score" in metrics
