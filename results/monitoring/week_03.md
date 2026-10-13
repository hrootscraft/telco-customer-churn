# Monitoring, week 3 (model v1): **OK**

Scored 2,000 customers. Retrain recommended: **no**.

| Check | Status |
|---|---|
| features | ok |
| scores | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| Tenure Months | 0.017 | 0.032 | ok |
| n_bills_visible | 0.017 | 0.032 | ok |
| usage_trend_3m | 0.016 | 0.032 | ok |
| Monthly Charges | 0.016 | 0.033 | ok |
| streaming_hours_last_3m | 0.014 | 0.051 | ok |

Score PSI 0.010; mean score 0.265 at training, 0.263 now.
