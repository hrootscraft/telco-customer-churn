# Monitoring, week 5 (model v1): **WARN**

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
| bill_shock | 0.200 | 0.104 | warn |
| Monthly Charges | 0.107 | 0.098 | warn |
| n_bills_visible | 0.029 | 0.046 | ok |
| usage_gb_last_month | 0.022 | 0.043 | ok |
| days_since_last_ticket | 0.021 | 0.041 | ok |

Score PSI 0.008; mean score 0.265 at training, 0.262 now.
Batch 1 matured: observed churn 0.268 vs predicted 0.260, ECE 0.049, PR-AUC 0.703 (card 0.704).
