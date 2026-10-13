# Monitoring, week 2 (model v1): **OK**

Scored 2,000 customers. Retrain recommended: **no**.

| Check | Status |
|---|---|
| features | ok |
| scores | ok |
| uplift | ok |

Most drifted features:

| Feature | PSI | KS | Status |
|---|---|---|---|
| bill_shock | 0.013 | 0.019 | ok |
| streaming_hours_last_3m | 0.013 | 0.029 | ok |
| Total Charges | 0.011 | 0.035 | ok |
| days_since_last_ticket | 0.011 | 0.022 | ok |
| avg_bill_prior_6m | 0.009 | 0.025 | ok |

Score PSI 0.016; mean score 0.265 at training, 0.272 now.
