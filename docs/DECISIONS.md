# Decisions

Each entry: the choice, the alternatives, and why. Numbered so the README and the
learning notes can cite them. "Phase" refers to the build order in the README.

## Phase 1: data, benchmark, explanation

**D1. The 33-column IBM file, not the 21-column Kaggle cut.** Same 7,043 customers.
The IBM version adds `CLTV`, `Churn Score` and `Churn Reason`. The first is what the
decision layer needs to turn a probability into a dollar value; the other two are a
textbook leakage trap and are the reason `data.assert_no_leakage` exists. Keeping
them in the raw file and *refusing* them in the feature matrix is more instructive
than deleting them.

**D2. One hold-out, repeated CV on the rest, and the hold-out is scored once.**
80/20 stratified. All model selection (regime, hyper-parameters, feature subset)
happens on the 80% with 2 × 5-fold stratified CV. The 20% produces the table the
README quotes. The 2023 notebook reported "87% accuracy" from a SMOTE-balanced
split, which is not a number any customer-facing team could reproduce; the rule
here is that a number is only quotable if it came from rows the model never saw in
any form.

**D3. Resampling regime: `none` for the model that ships; SMOTE and class weights
are compared, not assumed.** SMOTE is on the resume so it stays in the benchmark,
applied inside each CV fold through `imblearn.Pipeline` (synthetic rows are made
from the training fold only). On this data it does not help: for every model the
`none` regime has equal or better CV PR-AUC and clearly better Brier score, because
oversampling shifts the predicted probabilities toward 0.5 and breaks calibration.
Since the decision layer multiplies probabilities by dollars, calibration matters
more than the 0.5-threshold metrics SMOTE was invented to improve. The learning note
`docs/learn/01_imbalance_and_metrics.md` works through why. Class weights behave
like a milder SMOTE. See `results/benchmark_cv.csv` and
`reports/figures/03_resampling_regimes.png`.

**D4. Metrics: rank first, then calibration, accuracy last.** Primary ROC-AUC and
PR-AUC; operational lift @ top decile and recall @ precision ≥ 0.5; calibration via
Brier and ECE; accuracy and F1 at 0.5 reported for comparability with the old
README and with the warning that "nobody churns" scores 73.5%.

**D5. Backward stepwise done as a logistic likelihood-ratio test, by feature
block.** The 2023 code fitted OLS to a 0/1 target and dropped columns one one-hot
level at a time. Here the model is logistic, the test is a likelihood-ratio test on
the whole block of a categorical feature, and it runs on the training split only.
It is compared with L1 regularisation and permutation importance. Finding: dropping
the stepwise-rejected features costs < 0.001 ROC-AUC, so stepwise is a parsimony
tool, not a performance lever.

**D6. PyTorch MLP wrapped as a scikit-learn estimator.** A wrapped estimator shares
the preprocessing, in-fold resampling and CV protocol of the other seven models, so
the comparison is like for like. Two hidden layers (64, 32), dropout 0.2, AdamW,
early stopping on a 10% slice of the training fold. TensorFlow was dropped: PyTorch
is the lighter dependency and the one the target roles name first.

**D7. SHAP on LightGBM, folded back to raw features.** `TreeExplainer` is exact and
fast on gradient-boosted trees; one-hot columns are summed back to their raw
feature so the output reads "Contract: 28% of the model's attention" rather than
"Contract_Two year". Segment drivers are mean *signed* SHAP within a segment, which
is what an offer owner needs ("for fibre customers, missing tech support is what
pushes risk up").

**D8. The engineered features are deliberately few.** `n_addon_services`,
`avg_monthly_charge`, `tenure_band`. Richer behavioural features come from the
event log in phase 2, where they can be computed as of a date; snapshot-only
features that encode the future would defeat that.

## Phase 2: event log and point-in-time features

**D9. The event log is synthetic, consistent with the snapshot, and its churn signal
is planted and calibrated.** The IBM file has one row per customer, so bills,
payments, usage, tickets and plan changes behind each row are invented by
`churnprop.events.simulate`. Hard consistency rules: a customer with tenure *t* has
exactly *t* monthly bills ending at the snapshot date and they sum to `Total Charges`;
payment lateness follows the real payment method; ticket rates follow the real
services. A churn signal (more late payments and tickets, falling usage, downgrades
in the last three months) is planted and scaled by `events.signal`. The scale was
chosen with the fast benchmark over 0.1 / 0.2 / 0.3: at 0.15 the event features add
about 0.03 PR-AUC, which is the modest, believable range for behavioural features on
top of a strong contract/tenure signal; 0.5 would have pushed PR-AUC to 0.87, which
no reader should believe. The README says in its first paragraph that the events
are synthetic and that the reported lift measures the pipeline recovering a planted
signal, not what real telco events would add.

**D10. Features are SQL, computed as of a date, with the date filter applied before
any window or aggregate.** Each family lives in one `sql/*_features.sql` file with
one parameter, `{as_of}`. The first CTE in every file filters its source table to
events dated on or before the as-of date; everything downstream (row numbers,
windows, 90/180/365-day sums) runs on that filtered set. That single discipline is
the point-in-time rule. `tests/test_events.py::test_point_in_time` edits every event
after the as-of date (amounts ×10, all late, zero usage, 400 injected tickets and
downgrades) and asserts the feature table is byte-identical. DuckDB is the engine
because it reads the hive-partitioned Parquet directly with no server; the same SQL
runs through `spark.sql` behind `--engine spark` for the PySpark story without
making Spark a dependency.

**D11. Parquet partitioned by month, zstd, one row group per file.** A first version
wrote one row group per customer (the per-customer frames survive `pd.concat` as
Arrow chunks), which made 460k rows weigh 426 MB and a feature build take 50 s.
`combine_chunks()` before writing takes that to 10 MB and 0.5 s. Kept here because it
is the kind of thing a reviewer asks about and the kind of thing that silently ruins
a production feature store.

## Phase 3: offers, the weekly loop, uplift under a budget

**D12. Offer responses are planted as relative reductions in the 4-week churn
hazard, written as readable rules in `configs/offers.yaml`, with sleeping dogs.**
The IBM file has no treatment, so how a customer responds to an offer has to be
invented. Each offer has a base effect and additive rules in `DataFrame.eval` syntax
(`` `Monthly Charges` > 80 → +0.25 `` for the bill credit) plus per-customer noise
(sd 0.06). Rules were chosen so each offer has a natural segment (price-sensitive
customers for the credit, fibre without support for the tech bundle, settled
month-to-month customers for the contract upgrade) and so each has **sleeping
dogs**, customers the offer pushes out (new customers on promo, two-year customers
reminded to shop). A targeting rule that cannot see sleeping dogs pays for them;
`results/policy_comparison.csv` reports the share of each policy's offers that went
to one. The rules reference event features too (`tickets_90d`), so the learner has a
reason to use the phase-2 pipeline.

**D13. The baseline 4-week hazard is the out-of-fold phase-1 score, rescaled to a
mean of 0.08.** The snapshot label is a lifetime flag (26.5% of customers), not a
weekly rate, so it is scaled: each customer's hazard is proportional to their
out-of-fold churn probability, with the book-wide mean set in config. 0.08 per four
weeks is high for a real telco (roughly 2% a week) and is chosen on purpose: it
gives a 12-week loop ~650 churn events, enough for a learner to find the planted
heterogeneity inside one quarter. At a realistic 2% a month the same loop would see
~250 churners and nothing could be learned from an 8% holdout in 12 weeks; the
reader should read the loop as a compressed year, and `--offers` lets the rate be
changed. The score is out-of-fold (5-fold `cross_val_predict`) so the propensity
the policies use was never fitted on the customer it scores.

**D14. Weekly budget $32,000, exploration 8%.** A first attempt with a $10,000
budget and a 10% holdout spent the whole budget on exploration (10% of 5,174
customers × ~$44 ≈ $23,000), so every policy made identical decisions. The budget is
set so the policy can treat about 12% of the eligible book a week after the holdout
is paid for. The exploration share is uniform over control and the three offers
(probability 1/K each), which is the simplest logging policy whose propensities are
known exactly; a real programme would skew it toward control.

**D15. Cooldown equals the outcome window, and outcomes are read only once
matured.** An offer's outcome is "churned within 4 weeks". A customer who received
an arm (control included) is ineligible for 4 weeks so that one outcome belongs to
one decision, and the learner trains only on rows whose 4 weeks have passed. That is
the delay a real weekly programme carries: the first model can be fitted in week 5.

**D16. Policies are compared on identical potential outcomes and the same
budget, and the realised result is replayed.** Every customer carries one uniform
per week, shared across policies, so a customer churns under policy A and not B only
because the arms differ (common random numbers). Even so, a single 12-week run
decides ~65 extra retained customers out of 4,500 with a standard deviation of
roughly 15, so the realised ranking of risk_top vs uplift flips from one draw to
the next. The pipeline therefore reports three things: the **planted-truth expected
reward** of each policy's decisions (noise-free, the headline), the realised result
of the run, and the mean ± sd over five **replays** of the coin flips with the
hazards fixed. A reader who sees the realised and expected numbers disagree should
conclude the run is short, not that the truth is wrong.

**D17. The shipped uplift learner trains on every logged row, not only the
exploration holdout.** On the holdout alone (unbiased, but ~370 rows a week split
over four arms) the T-learner was data-starved and did not beat the risk × value
heuristic inside 12 weeks (truth $10.1k vs $12.2k a week). Trained on all logged
rows it did ($15.4k a week, and $14.4k vs $6.8k over the second half). Policy rows
are not randomised, but the arm was chosen from the same features the learner sees
(plus the logged propensity score), so assignment is ignorable given the features;
the holdout keeps every arm's support alive for every segment. `uplift.train_on`
switches back to `exploration` for the clean version. Two cheaper fixes a real team
would also try, and that the docs name rather than pretend: skew exploration toward
the arms the policy under-serves, and pool outcomes across weeks for the warm-up.

**D18. Off-policy evaluation is reported with its standard error and the weeks it
would take to be useful, instead of being quietly dropped.** IPS and doubly robust
estimates from the exploration rows are unbiased, and in this setting useless at 12
weeks: the reward is dollars × a rare churn event, the holdout is 8%, and a policy
differs from "do nothing" on about 10% of rows, so the DR 95% interval is roughly
±$300k a week around a true $15k. `results/policy_comparison.csv` gives the
estimate, its standard error, the minimum detectable effect at 80% power and the
weeks of logging needed to detect the policy's true value; figure 16 shows it. This
is the honest version of a finding every retention team meets: a small randomised
holdout proves the programme is not harmful long before it measures its value, and
the simulator's planted truth is what lets this repository say so with numbers.

## Phase 4: the model in production

**D19. Two stores for the model: MLflow for lineage, a committed export for
serving.** Every `churnprop train` logs parameters, hold-out metrics, the data hash,
the git commit, the pipeline and the monitoring reference statistics to an MLflow
tracking server backed by a local SQLite file (`mlflow.db`, gitignored), and
registers the version with the alias `candidate`. Promotion moves the `production`
alias **and** writes the version to `models/production/` as plain files
(`model.joblib`, `model_card.json`, `reference_stats.json`, `VERSION`) that are
committed. Scoring, the Airflow DAG, the GitHub Actions job and the Streamlit app
read only the export, so none of them needs a registry server, and the git history
of `models/production/` is the deployment log. The alternative, a hosted registry
every consumer talks to, is the right call for a team and the wrong one for a
portfolio repository that must run from a clone.

**D20. Promotion runs a gate, and the gate is a function with a test.** A
candidate replaces production only if its hold-out PR-AUC is within 0.005 of
production's and its ECE is at most 0.05 (`registry.gate`). The thresholds sit in
`configs/default.yaml: production.gate`. `churnprop production` demonstrates it by
training a second candidate without event features, which the gate refuses
(hold-out PR-AUC 0.671 against production's 0.704); `--force` records a forced promotion
on the card, and `promote --rollback` swaps the `production` and `previous` aliases
and re-exports. The gate compares like with like: both numbers come from the same
hold-out split, which is also why the training split is fixed by the config seed.

**D21. Weekly batches are bootstrap samples of the book with drift injected on a
schedule, and the schedule names three different kinds.** One snapshot cannot
produce a year of cohorts, so each week is a 2,000-row resample with event features
computed as of that week (phase 2's SQL at a later `{as_of}`). Three drifts are
injected from the weeks named in `batches.drift`, chosen because they are caught by
different monitors: a **price rise** on fibre (covariate shift: PSI/KS on `Monthly
Charges` and on the score), an **acquisition mix shift** toward fibre (population
drift: no row changes, the category shares do), and a **label shift** in the
electronic-check month-to-month segment (concept drift: features identical, more
churn). The third is invisible to every input monitor and surfaces only when the
batch's labels mature four weeks later, which is the point of including it.

**D22. Monitoring compares to reference statistics captured at training time, not
to the training data.** `reference_stats` stores, per feature, the decile edges and
proportions on the training frame, 21 quantiles with the reference ECDF at each (so
the KS statistic is right for discrete features such as ticket counts, where many
quantiles coincide), the category shares, the score histogram and the training churn
rate. It travels with the model card. Thresholds: PSI 0.10 warn / 0.25 alert, KS
0.15, score PSI 0.10, ECE 0.05, observed-vs-predicted churn ±30%, PR-AUC 0.05 below
the card, and a realised-minus-predicted uplift gap of 0.04 on the loop's exploration
rows. Any alert recommends a retrain; the DAG branches on it. These are the usual
industry defaults, not tuned to the injected drift. On the simulated year they warn
at week 5 (the price rise: `Monthly Charges` PSI 0.20, under the alert line, exactly
as a 12% rise on 44% of rows should score), alert from week 8 (the mix shift stacks
on the price rise and `Monthly Charges` crosses 0.25), and raise the outcome alert at
week 11, four weeks after the label shift began (observed churn 0.294 vs predicted
0.262 on batch 7, z = 3.3). There is also an unplanned true positive: `bill_shock`
alerts at week 6 because the event generator's bills *after* the snapshot carry more
noise than the ones before it, which are constrained to sum to `Total Charges`. The
monitor found a seam in the simulator; it is left in, labelled, because that is what
monitors are for and because fixing the generator would change the phase-2 and
phase-3 numbers already reported. Batches are drawn from hold-out customers only
(`batches.source: holdout`), so the matured-label PR-AUC is comparable to the model
card rather than flattered by training rows.

**D23. The Airflow DAG calls the same functions as the CLI, imports them inside the
tasks, and is validated in CI with `DagBag`.** `dags/retention_weekly.py` is ingest →
validate → score + monitor → branch → (retrain through the gate) → publish the offer
plan, Mondays 06:00, one run at a time. Importing `churnprop` inside each task keeps
the file parseable on a scheduler that has only Airflow, which is what the
`airflow-dag` CI job is: Airflow 3.3 with its constraints file, `pytest
tests/test_dag.py --noconftest`, asserting no import errors, the six task ids, the
branch's two children and the `none_failed_min_one_success` trigger rule on the
publish step. The local UI is `docker compose -f docker/airflow/docker-compose.yml
up` (`airflow standalone` with the repo mounted). Airflow is not a dependency of the
package.

**D24. A GitHub Actions schedule runs the same weekly job so the monitoring keeps
moving without anyone hosting Airflow.** `.github/workflows/weekly.yml` rebuilds the
regenerable stores (`churnprop events`, `churnprop batches`), runs `churnprop weekly`
for the week after the last one monitored (`results/monitoring/state.json`, wrapping
around the simulated year), commits `results/monitoring/` and opens an issue when the
week alerts. It is honest about what it is: a cron around the CLI, not an
orchestrator; retries, backfills, task-level visibility and the retrain branch are
why the DAG exists. Both exist so a reader can see the schedule run on GitHub and the
orchestration logic in Airflow.
