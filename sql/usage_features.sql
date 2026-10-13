-- Usage features as of {as_of}: level and 3-month trend. Customers without internet have no rows
-- and get NULLs here, which the feature builder fills with 0 and a "no internet" flag.
WITH visible AS (
    SELECT customer_id, month_end, data_gb, streaming_hours,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month_end DESC) AS recency
    FROM usage
    WHERE month_end <= DATE '{as_of}'
),
recent AS (
    SELECT customer_id,
           AVG(CASE WHEN recency <= 3 THEN data_gb END)             AS usage_gb_last_3m,
           AVG(CASE WHEN recency BETWEEN 4 AND 6 THEN data_gb END)  AS usage_gb_prior_3m,
           MAX(CASE WHEN recency = 1 THEN data_gb END)              AS usage_gb_last_month,
           AVG(CASE WHEN recency <= 3 THEN streaming_hours END)     AS streaming_hours_last_3m
    FROM visible
    GROUP BY customer_id
)
SELECT customer_id,
       usage_gb_last_month,
       usage_gb_last_3m,
       streaming_hours_last_3m,
       CASE WHEN usage_gb_prior_3m > 0 THEN usage_gb_last_3m / usage_gb_prior_3m - 1 ELSE 0 END AS usage_trend_3m
FROM recent
