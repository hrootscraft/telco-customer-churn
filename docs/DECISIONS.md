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
