# Monitoring, week 7 (model v1): **WARN**

Scored 2,000 customers. Retrain recommended: **no**.

| Check | Status |
|---|---|
| features | warn |
| scores | ok |
| outcomes | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| bill_shock | 0.232 | 0.128 | warn |
| Monthly Charges | 0.106 | 0.109 | warn |
| n_bills_visible | 0.035 | 0.046 | ok |
| avg_bill_prior_6m | 0.030 | 0.032 | ok |
| days_since_last_ticket | 0.027 | 0.045 | ok |

Score PSI 0.006; mean score 0.265 at training, 0.262 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
Batch 2 matured: observed churn 0.271 vs predicted 0.272, ECE 0.024, PR-AUC 0.733 (card 0.704).
Batch 3 matured: observed churn 0.258 vs predicted 0.263, ECE 0.015, PR-AUC 0.706 (card 0.704).
