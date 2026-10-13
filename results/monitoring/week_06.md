# Monitoring, week 6 (model v1): **ALERT**

Scored 2,000 customers. Retrain recommended: **yes**.

Alerts:
- bill_shock: PSI 0.26, KS 0.13

| Check | Status |
|---|---|
| features | alert |
| scores | ok |
| outcomes | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| bill_shock | 0.263 | 0.129 | alert |
| Monthly Charges | 0.132 | 0.122 | warn |
| n_bills_visible | 0.033 | 0.043 | ok |
| days_since_last_ticket | 0.028 | 0.034 | ok |
| avg_bill_prior_6m | 0.019 | 0.030 | ok |

Score PSI 0.015; mean score 0.265 at training, 0.261 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
