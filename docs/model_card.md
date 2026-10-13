# Model card: `churnprop-propensity` v1

Rendered from [`models/production/model_card.json`](../models/production/model_card.json),
which `churnprop promote` writes on every promotion. The JSON is the source of truth;
this page is the readable copy.

| | |
|---|---|
| Family | XGBoost (`xgboost`), scikit-learn pipeline: preprocessing → classifier, no resampling (D3) |
| Version / run | v1 · MLflow run `928b31607bd5…` · git `96c4690` · data hash `1bcbc0ccc9b35217` |
| Trained | 2026-10-13T03:55:04 UTC on 5,634 customers (80% stratified split, seed 42); hold-out 1,409 |
| Features | 44: the snapshot columns (contract, tenure, charges, services, demographics, three engineered) plus 22 event features computed as of the scoring date (phase 2) |
| Target | `Churn Value` from the IBM snapshot; in production, read as "churns within 4 weeks" (D13) |
| Gate at promotion | passed: first production version, ECE under 0.05 |

## Hold-out performance

| Metric | Value | Read it as |
|---|---|---|
| ROC-AUC | 0.869 | ranking quality over the whole book |
| PR-AUC | 0.704 | ranking quality where it matters (26.5% base rate) |
| Lift @ top decile | 2.99× | the riskiest 10% churn at ~78% |
| Recall @ precision ≥ 0.5 | 89% | share of churners caught when half the list is right |
| Brier | 0.126 | mean squared error of the probability |
| ECE | 0.023 | calibration: the probability means what it says |
| Accuracy @ 0.5 | 81.5% | reported, not trusted: 73.5% is free (D4) |

## What drives the score (share of mean |SHAP| at training)

| Feature | Share |
|---|---|
| Contract | 18% |
| Dependents | 11% |
| Tenure Months | 8% |
| usage_trend_3m | 7% |
| Online Security | 5% |
| Internet Service | 5% |
| Tech Support | 4% |
| usage_gb_last_month | 3% |

## Intended use

Rank active customers by 4-week churn risk for the weekly offer loop; the score is multiplied by CLTV in the decision layer, so calibration matters as much as ranking.

## Limitations

Trained on the IBM snapshot plus a synthetic event log (D9). Event features are computed as of the scoring date; the model has not seen prices or mixes outside the training book, which the drift monitors watch for. The event features and the offer responses used downstream are
synthetic ([SIMPLIFICATIONS](SIMPLIFICATIONS.md)). Not for decisions about individual
people outside the retention programme it was built for.

## Monitoring

Reference statistics for drift (`reference_stats.json`) travel with this version.
Thresholds and what each check can see: D22 and [learn note 06](learn/06_drift_monitoring.md).
Weekly reports: [`results/monitoring/`](../results/monitoring/).
