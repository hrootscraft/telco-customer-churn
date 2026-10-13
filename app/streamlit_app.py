"""Retention propensity & offer targeting: the app for the people who own the offers.

Three pages: score a customer (and see why), this week's offer plan under a budget,
and monitoring (drift, outcomes, the registry). Everything it reads is committed to
the repository, so it runs on Streamlit Cloud from a clone.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.views import monitoring, offers, score  # noqa: E402

st.set_page_config(
    page_title="Retention propensity & offer targeting",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(score.render, title="Score a customer", icon="🎯", default=True),
    st.Page(offers.render, title="This week's offer plan", icon="💸", url_path="offers"),
    st.Page(monitoring.render, title="Monitoring", icon="📈", url_path="monitoring"),
]
# `?page=score|offers|monitoring` renders one page directly; the headless AppTest uses it
forced = st.query_params.get("page")
views = {"score": score, "offers": offers, "monitoring": monitoring}
nav = st.navigation(pages) if forced not in views else None
with st.sidebar:
    st.markdown(
        "**Telco retention**  \n"
        "Real IBM customers, a calibrated churn model, a budgeted offer plan, and the "
        "monitoring that keeps it honest.  \n"
        "[Repository](https://github.com/hrootscraft/telco-customer-churn) · "
        "[Decisions](https://github.com/hrootscraft/telco-customer-churn/blob/main/docs/DECISIONS.md)"
    )
    st.divider()
if nav is None:
    views[forced].render()
else:
    nav.run()
