# Monitoring, week 12 (model v1): **ALERT**

Scored 2,000 customers. Retrain recommended: **yes**.

Alerts:
- Monthly Charges: PSI 0.23, KS 0.19
- batch 7: observed churn 0.294 vs predicted 0.262 (+12%, z=3.3)
- batch 8: observed churn 0.328 vs predicted 0.296 (+11%, z=3.1)

| Check | Status |
|---|---|
| features | alert |
| scores | ok |
| outcomes | alert |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.234 | 0.191 | alert |
| bill_shock | 0.149 | 0.088 | warn |
| days_since_last_ticket | 0.073 | 0.061 | ok |
| streaming_hours_last_3m | 0.070 | 0.090 | ok |
| Internet Service | 0.070 |  | ok |

Score PSI 0.018; mean score 0.265 at training, 0.288 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
Batch 3 matured: observed churn 0.258 vs predicted 0.263, ECE 0.015, PR-AUC 0.706 (card 0.704).
Batch 4 matured: observed churn 0.264 vs predicted 0.262, ECE 0.029, PR-AUC 0.703 (card 0.704).
Batch 5 matured: observed churn 0.274 vs predicted 0.262, ECE 0.035, PR-AUC 0.714 (card 0.704).
Batch 6 matured: observed churn 0.269 vs predicted 0.261, ECE 0.027, PR-AUC 0.706 (card 0.704).
Batch 7 matured: observed churn 0.294 vs predicted 0.262, ECE 0.040, PR-AUC 0.705 (card 0.704).
Batch 8 matured: observed churn 0.328 vs predicted 0.296, ECE 0.041, PR-AUC 0.746 (card 0.704).
