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
