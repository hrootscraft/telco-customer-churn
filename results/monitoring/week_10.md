# Monitoring, week 10 (model v1): **ALERT**

Scored 2,000 customers. Retrain recommended: **yes**.

Alerts:
- Monthly Charges: PSI 0.23, KS 0.20

| Check | Status |
|---|---|
| features | alert |
| scores | ok |
| outcomes | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.232 | 0.197 | alert |
| bill_shock | 0.151 | 0.086 | warn |
| Internet Service | 0.082 |  | ok |
| streaming_hours_last_3m | 0.076 | 0.081 | ok |
| avg_bill_prior_6m | 0.060 | 0.097 | ok |

Score PSI 0.020; mean score 0.265 at training, 0.293 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
Batch 3 matured: observed churn 0.258 vs predicted 0.263, ECE 0.015, PR-AUC 0.706 (card 0.704).
Batch 4 matured: observed churn 0.264 vs predicted 0.262, ECE 0.029, PR-AUC 0.703 (card 0.704).
Batch 5 matured: observed churn 0.274 vs predicted 0.262, ECE 0.035, PR-AUC 0.714 (card 0.704).
Batch 6 matured: observed churn 0.269 vs predicted 0.261, ECE 0.027, PR-AUC 0.706 (card 0.704).
