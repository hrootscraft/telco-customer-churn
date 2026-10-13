-- Plan-change features as of {as_of}.
WITH visible AS (
    SELECT customer_id, change_date, change_type
    FROM plan_changes
    WHERE change_date <= DATE '{as_of}'
)
SELECT customer_id,
       DATE_DIFF('day', MAX(change_date), DATE '{as_of}') AS days_since_last_plan_change,
       SUM(CASE WHEN change_date > DATE '{as_of}' - INTERVAL 180 DAY
                 AND change_type IN ('downgrade', 'remove_service') THEN 1 ELSE 0 END) AS downgrades_180d,
       SUM(CASE WHEN change_date > DATE '{as_of}' - INTERVAL 180 DAY
                 AND change_type IN ('upgrade', 'add_service') THEN 1 ELSE 0 END) AS upgrades_180d
FROM visible
GROUP BY customer_id
