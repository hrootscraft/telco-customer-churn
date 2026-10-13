-- Support-ticket features as of {as_of}.
WITH visible AS (
    SELECT customer_id, ticket_date, category, resolution_days, csat
    FROM tickets
    WHERE ticket_date <= DATE '{as_of}'
)
SELECT customer_id,
       SUM(CASE WHEN ticket_date > DATE '{as_of}' - INTERVAL 90 DAY  THEN 1 ELSE 0 END) AS tickets_90d,
       SUM(CASE WHEN ticket_date > DATE '{as_of}' - INTERVAL 365 DAY THEN 1 ELSE 0 END) AS tickets_365d,
       SUM(CASE WHEN ticket_date > DATE '{as_of}' - INTERVAL 90 DAY AND category = 'network' THEN 1 ELSE 0 END) AS network_tickets_90d,
       DATE_DIFF('day', MAX(ticket_date), DATE '{as_of}') AS days_since_last_ticket,
       AVG(CASE WHEN ticket_date > DATE '{as_of}' - INTERVAL 180 DAY THEN csat END) AS avg_csat_180d,
       AVG(CASE WHEN ticket_date > DATE '{as_of}' - INTERVAL 180 DAY THEN resolution_days END) AS avg_resolution_days_180d
FROM visible
GROUP BY customer_id
