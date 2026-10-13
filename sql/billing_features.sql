-- Billing and payment features as of {as_of}.
-- Only bills dated on or before the as-of date are visible (point-in-time rule).
WITH visible AS (
    SELECT customer_id, bill_date, amount, is_late, late_days,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY bill_date DESC) AS recency
    FROM billing
    WHERE bill_date <= DATE '{as_of}'
),
last_bill AS (
    SELECT customer_id, amount AS last_bill_amount
    FROM visible WHERE recency = 1
),
prior_6 AS (
    SELECT customer_id, AVG(amount) AS avg_bill_prior_6m
    FROM visible WHERE recency BETWEEN 2 AND 7
    GROUP BY customer_id
),
late_90 AS (
    SELECT customer_id,
           SUM(CASE WHEN is_late THEN 1 ELSE 0 END) AS late_payments_90d,
           SUM(late_days) AS late_days_90d
    FROM visible
    WHERE bill_date > DATE '{as_of}' - INTERVAL 90 DAY
    GROUP BY customer_id
),
late_365 AS (
    SELECT customer_id,
           AVG(CASE WHEN is_late THEN 1.0 ELSE 0.0 END) AS late_rate_365d
    FROM visible
    WHERE bill_date > DATE '{as_of}' - INTERVAL 365 DAY
    GROUP BY customer_id
),
-- consecutive late bills counting back from the most recent one
streak AS (
    SELECT customer_id,
           COALESCE(MIN(CASE WHEN NOT is_late THEN recency END) - 1, COUNT(*)) AS late_streak
    FROM visible
    GROUP BY customer_id
),
counts AS (
    SELECT customer_id, COUNT(*) AS n_bills_visible
    FROM visible GROUP BY customer_id
)
SELECT c.customer_id,
       c.n_bills_visible,
       lb.last_bill_amount,
       p6.avg_bill_prior_6m,
       CASE WHEN p6.avg_bill_prior_6m > 0
            THEN lb.last_bill_amount / p6.avg_bill_prior_6m - 1 ELSE 0 END AS bill_shock,
       COALESCE(l90.late_payments_90d, 0) AS late_payments_90d,
       COALESCE(l90.late_days_90d, 0)     AS late_days_90d,
       COALESCE(l365.late_rate_365d, 0)   AS late_rate_365d,
       COALESCE(s.late_streak, 0)         AS late_streak
FROM counts c
LEFT JOIN last_bill lb USING (customer_id)
LEFT JOIN prior_6 p6 USING (customer_id)
LEFT JOIN late_90 l90 USING (customer_id)
LEFT JOIN late_365 l365 USING (customer_id)
LEFT JOIN streak s USING (customer_id)
