# Resume bullets for this project (drafts, with the numbers they rest on)

Every number below is in `results/` and reproducible with one command. The old
bullet's "87% accuracy" is not among them: the honest accuracy is 80–82% against a
73.5% floor for predicting nobody churns (D4), and accuracy is the wrong metric for
a score that gets multiplied by dollars. Use lift, recall at a precision floor, and
calibration instead; for the production half, use the detection weeks.

## Option A: one project, two bullets (recommended)

**Retention Propensity & Offer Targeting** · Python, scikit-learn, XGBoost, LightGBM,
PyTorch, SHAP, DuckDB/SQL, MLflow, Airflow, GitHub Actions, Streamlit

- Built an end-to-end retention system on 7k telco customers: benchmarked 8 models
  (logistic to a PyTorch MLP) under repeated stratified CV with resampling inside
  the pipeline, shipping a calibrated XGBoost (hold-out ROC-AUC 0.87, PR-AUC 0.70,
  ECE 0.02; 2.9× lift in the top decile, 87% of churners caught at 50% precision)
  and showing SMOTE lowered PR-AUC and worsened calibration on every model; added
  22 point-in-time SQL features from an event log (+0.03–0.05 PR-AUC) with a
  byte-equality leakage test; explained drivers with SHAP at book, segment and
  customer level.
- Turned the score into a weekly offer programme and owned it in production:
  three-offer uplift targeting with a Lagrangian budget allocator (within 0.04% of
  the LP bound) that earned 2.1× the net value of risk-ranked targeting in
  simulation while sending 19% fewer offers; MLflow registry with a gated,
  committed promotion path and rollback; PSI/KS, score and matured-outcome drift
  monitors that flagged an injected price rise in week 1 and concept drift four
  weeks after onset; Airflow DAG validated in CI plus a scheduled GitHub Actions job
  that files issues on alert; three-page Streamlit app for the offers team.

## Option B: three shorter bullets

- Benchmarked 8 churn models (logistic, KNN, SVM, RF, XGBoost, LightGBM, ensemble,
  PyTorch MLP) with resampling inside cross-validation; shipped a calibrated XGBoost
  with ROC-AUC 0.87 / PR-AUC 0.70 / ECE 0.02 and 2.9× top-decile lift, and
  documented why SMOTE and accuracy were dropped from the headline.
- Designed a weekly multi-offer retention loop with an 8% randomised holdout,
  a T-learner uplift model and a budget-constrained allocator; in simulation it
  earned $14.4k vs $6.8k per week against the risk × value heuristic and halved
  offers sent to "sleeping dogs"; reported IPS/DR off-policy estimates with their
  power so the holdout's limits were explicit.
- Productionised the model: MLflow registry with a promotion gate (refused a
  candidate 0.03 PR-AUC below production), point-in-time SQL feature pipeline in
  DuckDB, drift monitoring (PSI/KS, calibration, outcome z-tests) over 12 weekly
  batches, an Airflow DAG with a retrain branch validated by `DagBag` in CI, and a
  Monday GitHub Actions job that commits reports and opens issues.

## Numbers and where they come from

| Claim | Value | Source |
|---|---|---|
| Hold-out ROC-AUC / PR-AUC / ECE (production XGBoost + events) | 0.869 / 0.704 / 0.023 | `models/production/model_card.json`, `results/headline_phase4.json` |
| Snapshot-only XGBoost | 0.855 / 0.671 / 0.022 | `results/benchmark_test.csv` |
| Lift @ top decile; recall @ precision ≥ 0.5 | 2.9× (78% vs 26.5%); 87% | `results/headline_phase1.json` |
| Accuracy vs "nobody churns" | 80.3% vs 73.5% | `results/benchmark_baselines.csv` |
| SMOTE effect | lower CV PR-AUC for all 8 models; Brier +3–35% | `results/benchmark_cv.csv`, figure 03 |
| Event features | +0.03 to +0.05 PR-AUC, 29% of SHAP attention | `results/benchmark_event_features.csv` |
| Month-to-month vs two-year churn | 42.7% vs 2.8% | `results/headline_phase1.json` |
| Uplift vs risk × value, second half | $14.4k vs $6.8k per week (2.1×), 19% fewer offers, sleeping-dog share 15% vs 18% (risk_top 19%, random 30%) | `results/policy_comparison.csv` |
| Allocator vs LP bound | gap 0.04% | `results/headline_phase3.json` |
| Gate refusal | v2 PR-AUC 0.671 vs v1 0.704 | `results/registry.csv` |
| Drift detection | price rise warns week 1 after onset (batch week 5), mix shift alerts week 1 after onset (week 8), concept drift alerts 4 weeks after onset (week 11, z = 3.3) | `results/monitoring/summary.csv` |

## What not to claim

- "87% accuracy": the old number came from SMOTE applied before the split.
- Any dollar figure as a real-world result: offer responses are planted (D12). Say
  "in simulation" or quote the ratio (2.1×), which is the honest comparison.
- "Deployed Airflow": the DAG is validated in CI and runs locally in Docker; the
  scheduled job that actually runs is GitHub Actions. "Airflow DAG with a retrain
  branch, validated in CI, plus a scheduled GitHub Actions job" is accurate.
