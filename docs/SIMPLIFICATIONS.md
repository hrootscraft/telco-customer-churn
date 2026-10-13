# What is simplified, and what a production version would do instead

Every number in the README says whether it comes from the real IBM snapshot or from
something this repository simulated. This page gathers the simulations and the
shortcuts in one place so nobody has to hunt for them.

| Area | What this repository does | What is simplified | What production would do |
|---|---|---|---|
| **Customers and labels** | 7,043 real IBM customers, their contracts, services, charges and churn flags | One snapshot; the churn label is a lifetime flag, not a dated event | A dated churn event per customer, so the label is "churned within N weeks of the scoring date" |
| **Event log** (phase 2) | Bills, payments, usage, tickets and plan changes generated behind each real customer, consistent with the snapshot (bills sum to `Total Charges`, lateness follows payment method, tickets follow services) | Synthetic; the churn signal is planted and calibrated to +0.03 PR-AUC (D9); post-snapshot bills are noisier than pre-snapshot ones (a seam the monitors found, D22) | Real event tables; the same SQL with `{as_of}` runs unchanged |
| **Point-in-time features** | Four SQL families, one `{as_of}` parameter, DuckDB over Parquet, a byte-equality test | Snapshot features (contract, tenure, charges) are taken as of the snapshot, not re-derived per date | A dimension table with effective dates, so contract and price are also as of the date |
| **Offer responses** (phase 3) | Relative 4-week hazard reductions per customer × offer from readable rules, with noise and sleeping dogs (D12) | Entirely invented; no real treatment exists in the data | Learned from the loop's own randomised holdout, which is what the loop is built to produce |
| **Baseline hazard** | Out-of-fold phase-1 score rescaled to a 4-week mean of 0.08 (D13) | About 4× a realistic telco weekly hazard, to compress a year of learning into 12 weeks | The realised rate; expect a longer warm-up before the uplift model beats the heuristic |
| **Budget and holdout** | $32k/week, 8% uniform exploration, 4-week cooldown and outcome window (D14, D15) | Uniform exploration over all arms; one arm per customer per cooldown | Exploration skewed toward control and toward the arms the policy under-serves; holdout concentrated where the policy acts (learn note 05) |
| **Policy comparison** | Six policies on common random numbers, planted-truth expected reward, five replays (D16) | The planted truth is the yardstick because the holdout is underpowered at 12 weeks (D18) | Off-policy evaluation on churn rate, more weeks, a larger targeted holdout |
| **Uplift learner** | T-/S-learner on LightGBM, trained on all logged rows (D17) | No cross-fitting, no X/R/DR learners, no tuning; tierlift benchmarks those | A learner chosen on Qini against the holdout, retrained weekly with the matured rows |
| **Weekly batches** (phase 4) | 2,000-row bootstrap of hold-out customers with event features as of each week, three drifts injected on a schedule (D21) | Resamples of one snapshot; drift is planted; labels arrive exactly four weeks later | Real cohorts; labels arrive when they arrive, with censoring |
| **Monitoring** | PSI/KS per feature, score PSI, matured-outcome checks with a z-test, uplift drift, fixed thresholds (D22) | Thresholds are industry defaults, not tuned to cost; no seasonality model; no per-segment calibration monitor | Thresholds set from the cost of a false page vs a missed drift; segment-level calibration; alerting on trend, not level |
| **Registry** | MLflow on local SQLite + a committed export with a gate (D19, D20) | The gate compares on the training-time hold-out; retraining reuses the snapshot split | A hosted registry; the gate compares on the most recent matured batches; retrain adds them to training |
| **Orchestration** | Airflow 3 DAG validated with `DagBag` in CI; Docker Compose for the UI; GitHub Actions cron runs the weekly step (D23, D24) | The DAG is not scheduled anywhere by default; the Actions job is a cron around the CLI | A scheduler someone operates (Airflow, Databricks Jobs), secrets, alert routing, SLAs |
| **App** (phase 5) | Streamlit Cloud from the committed export and results; scores the hold-out book live | No authentication, no write-back, the offer page uses the heuristic (the learned policy lives in the loop) | A service behind the company's auth that reads the week's plan from the DAG's output |
| **Hyper-parameters** | Sensible defaults for all eight models | Not tuned; the top six are within one CV standard deviation of each other | Tuning once the metric that matters (net value per $ under the budget) is agreed |

Two things are not simplified and are worth saying plainly: nothing from the
customer's future enters a training row (the point-in-time test proves it), and no
number in the README comes from a model that saw its own test data (the gate and the
monitors compare on hold-out customers only).
