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

> **Status: phase 1 of 5 is complete** (data, honest benchmark, feature selection,
> SHAP, tests, CI). The build order and what each phase adds is in
> [Phases](#phases). The 2023 version of this repository (one notebook and a
> Streamlit script that retrained models on every click) is kept in `legacy/`.

This is the operations-side companion to
[revenue-tier-and-churn-experiment](https://github.com/hrootscraft/revenue-tier-and-churn-experiment)
(tierlift), which covers experiment design, CUPED, uplift-learner benchmarking and
effect durability on simulated data. The split is deliberate:
[`docs/learn/03_propensity_vs_uplift.md`](docs/learn/03_propensity_vs_uplift.md).

## Headline results (phase 1; `results/headline_phase1.json`)

| Question | Method | Result |
|---|---|---|
| How well can churn be ranked? | 8 models, 2 × 5-fold stratified CV on 80%, scored once on the 20% hold-out | **XGBoost ROC-AUC 0.855, PR-AUC 0.671**; six of eight models within 0.007 ROC-AUC of each other (`results/benchmark_test.csv`) |
| What does the top decile buy? | Lift @ top decile; recall @ precision ≥ 0.5 | **2.9× lift**: the riskiest 10% churn at 78% vs 26.5% base; **87% of churners** are caught at 50% precision |
| Does SMOTE help under 26.5% imbalance? | Same folds, three regimes: none / SMOTE / class weights | **No.** SMOTE lowers CV PR-AUC for every model and raises Brier by 3–35%; the shipped model uses no resampling (D3) |
| Can the probabilities be multiplied by dollars? | Brier, ECE, reliability curve | **ECE 0.022** for XGBoost with no post-hoc calibration; the curve sits on the diagonal (figure 05) |
| What about "87% accuracy"? | Accuracy @ 0.5 next to "nobody churns" | **80.3%** vs **73.5%** for predicting nobody churns. Accuracy is reported, not trusted (D4) |
| Does backward stepwise help? | Logistic LR-test by feature block; L1; permutation importance | Keeps **12 of 22** features at ROC-AUC **0.861 vs 0.860**: a parsimony tool, not a performance lever (D5) |
| What drives the score? | SHAP on LightGBM, one-hot columns folded back | **Contract 24%** of total attention, Dependents 17%, Tenure 10%. Month-to-month contracts churn at **42.7%** vs 2.8% on two-year (not the 75% vs 3% in older write-ups) |

<p align="center"><img src="reports/figures/02_benchmark.png" width="900"></p>

## Contents

1. [The business problem](#1-the-business-problem)
2. [The data, and the leakage trap in it](#2-the-data-and-the-leakage-trap-in-it)
3. [Phase 1 results](#3-phase-1-results)
4. [What I would tell marketing](#4-what-i-would-tell-marketing)
5. [Phases](#phases)
6. [How to run it](#6-how-to-run-it)
7. [Repository layout](#7-repository-layout)
8. [Learning write-ups](#8-learning-write-ups)
9. [Limitations](#9-limitations)

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

## 4. What I would tell marketing

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
| 2. Event-level features in SQL | Synthetic monthly bills, payments, usage, tickets and plan changes behind each real customer; DuckDB SQL feature families computed *as of* a date; a point-in-time leakage test; optional Spark path; benchmark re-run with event features | next |
| 3. Offers and the weekly loop | Three offers with different costs plus control, responses simulated on the real customers; a 12-week loop with a 10% randomised exploration holdout; multi-treatment uplift; Lagrangian allocation under a weekly budget; off-policy evaluation | |
| 4. Production | MLflow registry with a committed production export; batch scoring CLI; PSI/KS, calibration and uplift drift monitoring with injected drift weeks; Airflow DAG (validated in CI) and a GitHub Actions weekly run of the same steps | |
| 5. App and narrative | Three-page Streamlit app (score, this week's offer plan by segment, monitoring); learning notes; model card | |

## 6. How to run it

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

churnprop benchmark          # full phase 1: ~10 min on 4 cores; writes results/ and reports/figures/
churnprop benchmark --fast   # 3 folds, 100 trees, 10 epochs: ~2 min
make test                    # unit tests (~1 min)
make test-all                # + end-to-end fast run
```

`configs/default.yaml` holds every tunable (split, folds, regime, model list, SHAP
segments). Results are deterministic for a given seed.

## 7. Repository layout

```
configs/default.yaml        every tunable, commented
data/raw/                   the IBM xlsx (unchanged)
src/churnprop/
  config.py                 YAML -> frozen dataclasses
  data.py                   load, clean, leakage gate, stratified split
  features.py               engineered columns, the shared ColumnTransformer
  propensity/
    models.py               the 8-model zoo: prep -> [smote] -> clf pipelines
    nn.py                   PyTorch MLP as a scikit-learn estimator
    metrics.py              ranking + calibration metrics
    benchmark.py            repeated CV, hold-out table, calibration bins
    stepwise.py             backward stepwise (logistic LR test), L1, permutation
  explain.py                SHAP: global, by segment, by customer
  plots.py                  all figures
  pipeline.py               phase 1 end to end
  cli.py                    `churnprop benchmark`
tests/                      data facts, leakage gate, SMOTE-in-pipeline, every model, stepwise, SHAP, e2e
results/                    tables the README quotes
reports/figures/            figures 01-09
docs/DECISIONS.md           D1-D8, the choices and why
docs/learn/                 learning notes with worked examples and exercises
legacy/                     the 2023 notebook and Streamlit script, unchanged
```

## 8. Learning write-ups

Each note is written to build intuition first, with worked examples and exercises
(answers folded):

1. [Class imbalance, SMOTE, and which metric to believe](docs/learn/01_imbalance_and_metrics.md)
2. [SHAP at three altitudes](docs/learn/02_shap_three_altitudes.md)
3. [Propensity is not uplift](docs/learn/03_propensity_vs_uplift.md)

## 9. Limitations

* One snapshot, no time dimension: tenure is the only history. Phase 2 adds a
  (synthetic, clearly labelled) event log so that features can be computed as of a
  date.
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
