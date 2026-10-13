# Monitoring, week 8 (model v1): **ALERT**

Scored 2,000 customers. Retrain recommended: **yes**.

Alerts:
- Monthly Charges: PSI 0.26, KS 0.21

| Check | Status |
|---|---|
| features | alert |
| scores | ok |
| outcomes | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.257 | 0.213 | alert |
| bill_shock | 0.231 | 0.149 | warn |
| Internet Service | 0.086 |  | ok |
| last_bill_amount | 0.079 | 0.131 | ok |
| streaming_hours_last_3m | 0.079 | 0.097 | ok |

Score PSI 0.022; mean score 0.265 at training, 0.296 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
Batch 3 matured: observed churn 0.258 vs predicted 0.263, ECE 0.015, PR-AUC 0.706 (card 0.704).
Batch 4 matured: observed churn 0.264 vs predicted 0.262, ECE 0.029, PR-AUC 0.703 (card 0.704).
