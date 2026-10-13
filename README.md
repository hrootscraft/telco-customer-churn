# Retention propensity and offer targeting on the Telco churn data

**Who is about to leave, what is each of them worth, and which ones should a fixed
weekly retention budget reach?**

An end-to-end retention-analytics project on the IBM Telco customer churn data
(7,043 real subscribers, 26.5% churn). It benchmarks eight churn-propensity models
the honest way (resampling inside cross-validation, one untouched hold-out,
calibration reported next to discrimination), explains the scores with SHAP at
three altitudes (book, segment, customer), and is being extended, phase by phase,
into the thing a retention team actually runs: event-level features built in SQL
as of a date, a weekly multi-offer targeting loop under a budget with a randomised
exploration holdout, a model registry, drift monitoring, an Airflow DAG, and a
Streamlit app for the people who own the offers.

> **Status: phases 1 and 2 of 5 are complete** (honest benchmark, feature
> selection, SHAP, tests, CI; then a synthetic event log with point-in-time SQL
> features). The build order and what each phase adds is in [Phases](#phases). The
> 2023 version of this repository (one notebook and a Streamlit script that
> retrained models on every click) is kept in `legacy/`.
>
> **What is real and what is simulated.** The 7,043 customers, their contracts,
> services, charges and churn labels are real (IBM's sample data). The monthly
> bills, payments, usage, tickets and plan changes behind each customer are
> **synthetic**, generated to be consistent with each real row, with a small churn
> signal planted and calibrated (D9). Offer responses in phase 3 will be synthetic
> too. Every number below says which it is.

This is the operations-side companion to
[revenue-tier-and-churn-experiment](https://github.com/hrootscraft/revenue-tier-and-churn-experiment)
(tierlift), which covers experiment design, CUPED, uplift-learner benchmarking and
effect durability on simulated data. The split is deliberate:
[`docs/learn/03_propensity_vs_uplift.md`](docs/learn/03_propensity_vs_uplift.md).

## Headline results (`results/headline_phase1.json`, `results/headline_phase2.json`)

| Question | Method | Result |
|---|---|---|
| How well can churn be ranked? | 8 models, 2 × 5-fold stratified CV on 80%, scored once on the 20% hold-out | **XGBoost ROC-AUC 0.855, PR-AUC 0.671**; six of eight models within 0.007 ROC-AUC of each other (`results/benchmark_test.csv`) |
| What does the top decile buy? | Lift @ top decile; recall @ precision ≥ 0.5 | **2.9× lift**: the riskiest 10% churn at 78% vs 26.5% base; **87% of churners** are caught at 50% precision |
| Does SMOTE help under 26.5% imbalance? | Same folds, three regimes: none / SMOTE / class weights | **No.** SMOTE lowers CV PR-AUC for every model and raises Brier by 3–35%; the shipped model uses no resampling (D3) |
| Can the probabilities be multiplied by dollars? | Brier, ECE, reliability curve | **ECE 0.022** for XGBoost with no post-hoc calibration; the curve sits on the diagonal (figure 05) |
| What about "87% accuracy"? | Accuracy @ 0.5 next to "nobody churns" | **80.3%** vs **73.5%** for predicting nobody churns. Accuracy is reported, not trusted (D4) |
| Does backward stepwise help? | Logistic LR-test by feature block; L1; permutation importance | Keeps **12 of 22** features at ROC-AUC **0.861 vs 0.860**: a parsimony tool, not a performance lever (D5) |
| What drives the score? | SHAP on LightGBM, one-hot columns folded back | **Contract 24%** of total attention, Dependents 17%, Tenure 10%. Month-to-month contracts churn at **42.7%** vs 2.8% on two-year (not the 75% vs 3% in older write-ups) |
| What do event-level features add, computed as of a date? | 461k synthetic events → 22 features in DuckDB SQL with an `{as_of}` filter; same split and folds | **+0.03 to +0.05 PR-AUC** on the hold-out for all four models (XGBoost 0.671 → 0.698, logistic 0.659 → 0.708); event features take 29% of SHAP attention, led by the 3-month usage trend. Lift is from a *planted* signal calibrated to be modest (D9) |
| Does anything read the future? | Point-in-time test: edit every event after the as-of date, rebuild | **Feature table unchanged, byte for byte** (`tests/test_events.py::test_point_in_time`) |

<p align="center"><img src="reports/figures/02_benchmark.png" width="900"></p>

## Contents

1. [The business problem](#1-the-business-problem)
2. [The data, and the leakage trap in it](#2-the-data-and-the-leakage-trap-in-it)
3. [Phase 1 results](#3-phase-1-results)
4. [Phase 2: event-level features in SQL, as of a date](#4-phase-2-event-level-features-in-sql-as-of-a-date)
5. [What I would tell marketing](#5-what-i-would-tell-marketing)
6. [Phases](#phases)
7. [How to run it](#7-how-to-run-it)
8. [Repository layout](#8-repository-layout)
9. [Learning write-ups](#9-learning-write-ups)
10. [Limitations](#10-limitations)

## 1. The business problem

A telco loses about a quarter of its customers a year. Winning a new one costs
several times what keeping an existing one costs, so every quarter marketing has a
retention budget and a question: *who gets an offer?*

The naive answer, "whoever is most likely to leave", is wrong in two ways. A churn
score cannot tell a customer an offer would save from one who is leaving regardless,
and it cannot see the customer who was staying until the offer reminded them to
shop around. A churn score is still the right *first* model: it is what the data
can support today, its probabilities set the stakes for each customer, and its
drivers tell you where the levers might be. This repository builds that model
honestly in phase 1 and then, in phases 2–5, builds the loop that turns it into
offers, measures them, and keeps the model alive in production.

## 2. The data, and the leakage trap in it

`data/raw/Telco_customer.xlsx` is IBM's 33-column sample dataset ("Telco customer
churn", shipped with IBM Cognos Analytics). It has the same 7,043 customers as the
21-column Kaggle version, plus California geography and four columns recorded
*after* the customer churned:

| Column | What it is | Used as a feature? |
|---|---|---|
| `Churn Label` / `Churn Value` | The target | target only |
| `Churn Score` | IBM's own model score | **never** |
| `Churn Reason` | Exit-survey answer, only for churners | **never** |
| `CLTV` | IBM's lifetime-value estimate | **never as a feature**; used in the decision layer (phase 3) to price a customer |

`churnprop.data.assert_no_leakage` raises before any model sees a matrix containing
these. Keeping them in the file and refusing them in code is the point (D1).

Features: 19 raw columns (demographics, services, contract, billing) plus three
engineered ones (`n_addon_services`, `avg_monthly_charge`, `tenure_band`). One
`ColumnTransformer` (scale numerics, one-hot categoricals) is the first step of
every model's pipeline, so it is fitted inside each CV fold.

## 3. Phase 1 results

### 3.1 Eight models, one protocol

80/20 stratified split; 2 × 5-fold stratified CV on the 80% for every choice; the
20% scored once. Models: logistic regression, k-NN, SVM (RBF, Platt-scaled),
random forest, XGBoost, LightGBM, an RF + XGBoost soft-vote ensemble, and a PyTorch
MLP (64-32, dropout, early stopping) wrapped as a scikit-learn estimator so it runs
through the identical pipeline (D6).

| Model | ROC-AUC | PR-AUC | Lift @ top 10% | Recall @ P ≥ 0.5 | Accuracy | Brier | ECE |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0.855** | **0.671** | 2.94 | 0.869 | 0.803 | 0.133 | 0.022 |
| LightGBM | 0.854 | 0.664 | **2.99** | **0.874** | 0.802 | 0.133 | 0.029 |
| RF + XGB soft vote | 0.854 | 0.670 | 2.91 | 0.872 | **0.807** | **0.132** | 0.018 |
| Neural net (PyTorch MLP) | 0.853 | 0.649 | 2.75 | 0.834 | 0.801 | 0.134 | **0.017** |
| Logistic regression | 0.852 | 0.659 | 2.83 | 0.826 | 0.804 | 0.134 | 0.026 |
| Random forest | 0.848 | 0.660 | 2.89 | 0.850 | 0.806 | 0.134 | 0.023 |
| k-nearest neighbours | 0.816 | 0.607 | 2.65 | 0.754 | 0.776 | 0.149 | 0.050 |
| SVM (RBF) | 0.814 | 0.639 | 2.89 | 0.799 | 0.803 | 0.142 | 0.028 |
| *Predict nobody churns* | *0.500* | *0.265* | *1.00* | *0* | *0.735* | | |

Hold-out, no resampling. CV standard deviations are ±0.01 ROC-AUC and ±0.02 PR-AUC
(`results/benchmark_cv.csv`), so the top six are statistically a tie: on 5,600
training rows with 22 tabular features, gradient boosting, a linear model and a
small neural net all find the same signal. The choice between them is made on
calibration, speed and explainability, which is why LightGBM carries the SHAP work
and XGBoost is the reference ranker.

### 3.2 SMOTE, compared rather than assumed

<p align="center"><img src="reports/figures/03_resampling_regimes.png" width="820"></p>

SMOTE is applied inside each fold through `imblearn.Pipeline`, so synthetic rows are
built from training rows only. Done that way, it does not help here: for all eight
models CV PR-AUC is equal or lower and Brier is worse, because oversampling to 50/50
shifts the predicted probabilities toward 0.5. Class weights behave like a milder
SMOTE. The models that ship use no resampling and are well calibrated as a result
(figure 05). Why this is the expected outcome, with worked examples:
[`docs/learn/01_imbalance_and_metrics.md`](docs/learn/01_imbalance_and_metrics.md).

### 3.3 Feature selection: stepwise, L1, permutation

Backward stepwise done as a logistic likelihood-ratio test on whole feature blocks
(D5) removes 10 of 22 features (gender, phone service, device protection, online
backup, senior citizen, total charges, average monthly charge, tech support, online
security, add-on count). Logistic CV ROC-AUC moves from 0.8601 to 0.8608. L1 keeps
19; permutation importance on LightGBM agrees on the top three (Contract,
Dependents, Tenure). `results/feature_selection.csv`, figure 07.

### 3.4 What drives the score

<p align="center"><img src="reports/figures/06_shap_global.png" width="640"></p>

Folding one-hot columns back to their feature, Contract carries 24% of the model's
total |SHAP|, Dependents 17%, Tenure 10%. Within segments the picture changes, which
is the altitude an offer owner works at:

<p align="center"><img src="reports/figures/09_shap_by_internet_service.png" width="900"></p>

For fibre customers (41% churn), being on a month-to-month contract, the fibre
price itself and missing online security push risk up; for DSL customers nearly
everything pulls it down. `results/shap_by_segment.csv`, figures 08–09, and
[`docs/learn/02_shap_three_altitudes.md`](docs/learn/02_shap_three_altitudes.md).

## 4. Phase 2: event-level features in SQL, as of a date

The snapshot has no time in it beyond tenure. Real retention models are built from
event tables (bills, payments, tickets, usage) and the hard part of building them is
not the SQL, it is making sure every feature is computed **as of the scoring date**
so that nothing from the customer's future leaks into their training row.

**The event log (synthetic).** `churnprop events` generates, behind each real
customer, one bill per month of tenure ending at a snapshot date of 2024-06-30
(summing to the real `Total Charges`), a payment with lateness driven by the real
payment method, monthly usage for internet customers, support tickets whose rate
follows the real services (fibre without tech support raises more), and occasional
plan changes; non-churners get three more months past the snapshot so the
point-in-time rule has something to catch. 461k rows, 10 MB of zstd Parquet
partitioned by month (`data/events/`, regenerable, not committed). A churn signal is
planted in the last three months (more late payments and tickets, falling usage,
downgrades) at `events.signal: 0.15`, calibrated so the features add about 0.03
PR-AUC (D9).

**The features (SQL).** Four files in `sql/`, one per family, each with a single
parameter `{as_of}`; the first CTE filters the source table to events on or before
that date and every window and aggregate runs on the filtered set. DuckDB runs them
straight off the Parquet partitions in 0.5 s; `--engine spark` runs the identical
SQL through PySpark. 22 features: bill shock (last bill vs the prior 6-month
average), late payments and late days in 90 days, late rate in a year, late streak,
usage level and 3-month trend, tickets in 90 and 365 days, network tickets, days
since last ticket, CSAT and resolution time, days since last plan change,
downgrades and upgrades in 180 days.

**The test.** `test_point_in_time` builds the table as of a date 45 days before the
snapshot, then multiplies every later bill by 10, marks it late, zeroes later usage,
injects 400 later tickets and downgrades, rebuilds, and asserts equality. A leak in
any feature fails with its column name.

<p align="center"><img src="reports/figures/10_event_feature_lift.png" width="760"></p>

| Model | PR-AUC, snapshot | PR-AUC, + events | ROC-AUC, snapshot | ROC-AUC, + events | Lift @ top 10%, + events |
|---|---|---|---|---|---|
| Logistic regression | 0.659 | **0.708** | 0.852 | 0.871 | 3.05 |
| LightGBM | 0.664 | 0.697 | 0.854 | 0.869 | 3.10 |
| XGBoost | 0.671 | 0.698 | 0.855 | 0.872 | 3.05 |
| Neural net | 0.649 | 0.694 | 0.853 | 0.869 | 3.07 |

Hold-out, same split and folds as phase 1. The gain is largest for the linear model,
which could not build "trend" or "streak" from the snapshot and now gets them
pre-computed; the boosters already squeezed more from tenure and charges. Event
features take 29% of SHAP attention (figure 11), led by the usage trend, usage level
and late days, which is where the signal was planted; nothing unplanted lit up, which
is the check that matters. Why the lift is honest only as a statement about the
pipeline, and what a real lift would need:
[`docs/learn/04_point_in_time_features.md`](docs/learn/04_point_in_time_features.md).

## 5. What I would tell marketing

1. **The model is ready to rank, and its probabilities can be trusted as
   probabilities.** The riskiest 10% of customers churn at 78%. A list of the top
   700 would contain about 550 real churners.
2. **Do not read the drivers as levers yet.** Month-to-month contracts churn 15×
   more than two-year ones, but the people who chose two-year contracts differ from
   those who did not. Whether a contract-upgrade *offer* reduces churn is a
   question for a holdout, which is what phase 3 builds.
3. **Stop quoting accuracy.** 80% sounds good until you learn that 73.5% is free.
   The numbers to put on a slide are lift (2.9×), recall at a precision floor
   (87% of churners at 50% precision) and the calibration curve.
4. **SMOTE is not a feature of this model, and that is a decision.** It was
   benchmarked and lost on the metrics that matter when scores are multiplied by
   dollars.

## Phases

| Phase | Adds | Status |
|---|---|---|
| **1. Honest benchmark** | `src/churnprop` package, config, tests, CI; leakage gates; 8 models with resampling inside CV; calibration; stepwise/L1/permutation; SHAP at three altitudes | **done** |
| **2. Event-level features in SQL** | Synthetic monthly bills, payments, usage, tickets and plan changes behind each real customer; DuckDB SQL feature families computed *as of* a date; a point-in-time leakage test; optional Spark path; benchmark re-run with event features | **done** |
| 3. Offers and the weekly loop | Three offers with different costs plus control, responses simulated on the real customers; a 12-week loop with a 10% randomised exploration holdout; multi-treatment uplift; Lagrangian allocation under a weekly budget; off-policy evaluation | next |
| 4. Production | MLflow registry with a committed production export; batch scoring CLI; PSI/KS, calibration and uplift drift monitoring with injected drift weeks; Airflow DAG (validated in CI) and a GitHub Actions weekly run of the same steps | |
| 5. App and narrative | Three-page Streamlit app (score, this week's offer plan by segment, monitoring); learning notes; model card | |

## 7. How to run it

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

churnprop benchmark          # phase 1: ~10 min on 4 cores; writes results/ and reports/figures/
churnprop benchmark --fast   # 3 folds, 100 trees, 10 epochs: ~2 min
churnprop events             # phase 2: simulate the event log into data/events/ (~25 s)
churnprop features --as-of 2024-03-31   # point-in-time features as of any date -> results/
churnprop benchmark-events   # phase 2: four models with vs without event features (~3 min)
make test                    # unit tests (~1 min)
make test-all                # + end-to-end fast run
```

`configs/default.yaml` holds every tunable (split, folds, regime, model list, SHAP
segments). Results are deterministic for a given seed.

## 8. Repository layout

```
configs/default.yaml        every tunable, commented
data/raw/                   the IBM xlsx (unchanged)
data/events/                synthetic event log, Parquet by month (generated, gitignored)
sql/                        one point-in-time feature family per file, parameter {as_of}
src/churnprop/
  config.py                 YAML -> frozen dataclasses
  data.py                   load, clean, leakage gate, stratified split
  features.py               engineered columns, the shared ColumnTransformer
  events/simulate.py        the event-log generator, consistent with each real row
  features_sql/build.py     runs sql/ through DuckDB (or Spark) as of a date
  pipeline_events.py        phase 2 end to end
  propensity/
    models.py               the 8-model zoo: prep -> [smote] -> clf pipelines
    nn.py                   PyTorch MLP as a scikit-learn estimator
    metrics.py              ranking + calibration metrics
    benchmark.py            repeated CV, hold-out table, calibration bins
    stepwise.py             backward stepwise (logistic LR test), L1, permutation
  explain.py                SHAP: global, by segment, by customer
  plots.py                  all figures
  pipeline.py               phase 1 end to end
  cli.py                    `churnprop benchmark | events | features | benchmark-events`
tests/                      data facts, leakage gate, SMOTE-in-pipeline, every model, stepwise, SHAP,
                            event consistency, point-in-time, e2e
results/                    tables the README quotes
reports/figures/            figures 01-11
docs/DECISIONS.md           D1-D8, the choices and why
docs/learn/                 learning notes with worked examples and exercises
legacy/                     the 2023 notebook and Streamlit script, unchanged
```

## 9. Learning write-ups

Each note is written to build intuition first, with worked examples and exercises
(answers folded):

1. [Class imbalance, SMOTE, and which metric to believe](docs/learn/01_imbalance_and_metrics.md)
2. [SHAP at three altitudes](docs/learn/02_shap_three_altitudes.md)
3. [Propensity is not uplift](docs/learn/03_propensity_vs_uplift.md)
4. [Point-in-time features from an event log](docs/learn/04_point_in_time_features.md)

## 10. Limitations

* The event log is synthetic. Its consistency with the snapshot is enforced, but
  its churn signal is planted, so the +0.03 PR-AUC says the pipeline works, not what
  real events would add. Real bills and tickets would also carry seasonality,
  outages and re-engagement that the generator does not model.
* No treatment: nothing in the file says what anyone was offered, so no causal claim
  is made from it. Phase 3 simulates offer responses on the real customers and runs
  the loop that would generate real treatment data.
* The hold-out is 1,409 customers, so differences below ~0.01 ROC-AUC between models
  are noise; the README says "tie" where it is one.
* Hyper-parameters are sensible defaults, not tuned; tuning would move the top six
  by less than their CV standard deviation.

**References:** IBM Cognos Analytics sample data, *Telco customer churn*; Kaggle
mirror by Blastchar. Methods: Chawla et al. 2002 (SMOTE); Lundberg & Lee 2017 (SHAP);
van den Goorbergh et al. 2022 (harm of imbalance corrections for risk models).
