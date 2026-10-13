# Monitoring, week 11 (model v1): **ALERT**

Scored 2,000 customers. Retrain recommended: **yes**.

Alerts:
- Monthly Charges: PSI 0.21, KS 0.19
- batch 7: observed churn 0.294 vs predicted 0.262 (+12%, z=3.3)

| Check | Status |
|---|---|
| features | alert |
| scores | ok |
| outcomes | alert |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.208 | 0.185 | alert |
| bill_shock | 0.175 | 0.094 | warn |
| avg_bill_prior_6m | 0.072 | 0.110 | ok |
| last_bill_amount | 0.069 | 0.111 | ok |
| days_since_last_ticket | 0.066 | 0.053 | ok |

Score PSI 0.021; mean score 0.265 at training, 0.289 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
Batch 3 matured: observed churn 0.258 vs predicted 0.263, ECE 0.015, PR-AUC 0.706 (card 0.704).
Batch 4 matured: observed churn 0.264 vs predicted 0.262, ECE 0.029, PR-AUC 0.703 (card 0.704).
Batch 5 matured: observed churn 0.274 vs predicted 0.262, ECE 0.035, PR-AUC 0.714 (card 0.704).
Batch 6 matured: observed churn 0.269 vs predicted 0.261, ECE 0.027, PR-AUC 0.706 (card 0.704).
Batch 7 matured: observed churn 0.294 vs predicted 0.262, ECE 0.040, PR-AUC 0.705 (card 0.704).
