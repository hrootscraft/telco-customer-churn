# Monitoring, week 1 (model v1): **OK**

Scored 2,000 customers. Retrain recommended: **no**.

| Check | Status |
|---|---|
| features | ok |
| scores | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.014 | 0.020 | ok |
| avg_csat_180d | 0.011 | 0.013 | ok |
| bill_shock | 0.011 | 0.020 | ok |
| Total Charges | 0.010 | 0.022 | ok |
| Tenure Months | 0.009 | 0.025 | ok |

Score PSI 0.013; mean score 0.265 at training, 0.260 now.
