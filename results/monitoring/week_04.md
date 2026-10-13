# Monitoring, week 4 (model v1): **OK**

Scored 2,000 customers. Retrain recommended: **no**.

| Check | Status |
|---|---|
| features | ok |
| scores | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Monthly Charges | 0.017 | 0.032 | ok |
| streaming_hours_last_3m | 0.015 | 0.040 | ok |
| Total Charges | 0.015 | 0.034 | ok |
| Tenure Months | 0.014 | 0.027 | ok |
| n_bills_visible | 0.014 | 0.027 | ok |

Score PSI 0.013; mean score 0.265 at training, 0.262 now.
