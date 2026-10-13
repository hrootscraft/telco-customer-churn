# 04 · Point-in-time features from an event log

*What this note covers:* why a feature store needs an "as of" date, the two ways
future information leaks into training data without anyone noticing, how the SQL
in `sql/` enforces the rule with one CTE, how to test it, and how to read the lift
the event features produce here.

## 1. The problem in one picture

```
customer A      ──bills──┬──ticket──┬──────────┬── churns ──▶
                         │          │          │
                     as-of date   (future)   label date
```

You want to predict at the **as-of date** whether A churns by the **label date**.
Anything to the right of the as-of line is the future. If a feature such as
"tickets in the last 90 days" is computed over the whole table, it counts the ticket
A raised *after* the as-of date, in the run-up to leaving. The model learns "people
with a recent ticket churn" and is right in training, because it was shown the
answer. In production, at the as-of date, that ticket does not exist yet.

**Intuition.** It is the difference between a weather forecast made on Monday for
Friday, and one "made on Monday" by someone who has already seen Friday.

## 2. Two leaks, one rule

**Leak 1: window computed over the full table.** `COUNT(*) WHERE ticket_date >
as_of - 90 days` without `AND ticket_date <= as_of` counts future tickets.

**Leak 2: "latest" rows.** `ROW_NUMBER() OVER (ORDER BY bill_date DESC) = 1` picks
the *last bill in the table*, which for a customer with future rows is a future
bill. Any "last", "most recent", "current" feature is exposed.

The one rule that closes both: **filter the source table to `event_date <= as_of`
in the first CTE, and let nothing downstream touch the raw table.** Every file in
`sql/` is shaped that way:

```sql
WITH visible AS (
    SELECT ... , ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY bill_date DESC) AS recency
    FROM billing
    WHERE bill_date <= DATE '{as_of}'      -- the only place the raw table is read
), ...
```

`recency = 1` is now "the latest bill *the business could have seen*", and every
90/180/365-day window is a window ending at the as-of date.

**Exercise 1.** This query is meant to give "late payments in the last 90 days as of
`{as_of}`". Find the leak.

```sql
SELECT customer_id,
       SUM(CASE WHEN is_late THEN 1 ELSE 0 END) AS late_90d
FROM billing
WHERE bill_date > DATE '{as_of}' - INTERVAL 90 DAY
GROUP BY customer_id
```

<details><summary>Answer</summary>

The window has a lower bound but no upper bound: a bill dated after `{as_of}` is
still `> as_of − 90 days`, so future late payments are counted. Add
`AND bill_date <= DATE '{as_of}'`, or (the house style) filter in a `visible` CTE
first and window over that.
</details>

## 3. Testing it

`tests/test_events.py::test_point_in_time` builds the feature table as of a date 45
days before the snapshot, then **edits every event after that date** (multiplies
amounts by 10, marks them late, zeroes usage, sets CSAT to 1, injects 400 new
tickets and downgrades), rebuilds, and asserts the two tables are identical. If any
feature anywhere reads the future, the test fails with the column name.

Why edit rather than delete? Deleting future rows would also pass a query that
reads the whole table but happens to see no future rows. Editing them to extreme
values makes a leak loud.

**Exercise 2.** The test asserts the tables are *identical*. Suppose a new feature
`avg_bill_all_time` were added without the `visible` filter. Which rows of the test
would change, and why would they change even for a customer with no injected
tickets?

<details><summary>Answer</summary>

Every customer with at least one bill after the as-of date (non-churners with
future months, plus anyone whose last history bill falls in the 45-day window),
because the test multiplied those amounts by 10, so an all-time average that reads
them moves. The injected tickets are irrelevant to a billing feature; the amount
edit is what catches it.
</details>

## 4. Why SQL, and why DuckDB

Features over event tables are joins, windows and date arithmetic, which is what SQL
is for, and the SQL reads the same whether the engine is DuckDB on a laptop, Spark
on a cluster or a warehouse. `features_sql.build` runs the files through DuckDB by
default (it reads the Parquet partitions directly, no server, 0.5 s for 460k rows)
and through `spark.sql` behind a flag; the SQL is identical apart from
`INTERVAL 90 DAY` vs `DAYS`.

The alternative, pandas groupbys in Python, works at this size but does not travel:
the production feature pipeline at a telco or a rewards platform is SQL or Spark,
and the as-of discipline has to live where the data lives.

## 5. Reading the lift honestly

Figure 10 and `results/benchmark_event_features.csv` show hold-out PR-AUC with and
without the event features. The lift is real in the sense that the pipeline recovered
it at the right date; it is not evidence about real customers, because the signal
was planted (D9). Two things the comparison *does* tell you:

1. The planted size matters enormously. At `signal: 0.5` the lift was +0.20
   PR-AUC; at 0.15 it is about +0.03. Anyone showing a huge lift from behavioural
   features on a public dataset should be asked how the behaviour got there.
2. Which event features the model used (figure 11) is informative about the
   *pipeline*: the usage trend and late-payment count carry most of the added
   attention, which matches where the signal was planted. If a feature nobody
   planted a signal into lit up, that would be a leak.

**Exercise 3.** You add a feature `months_until_churn` by mistake (it is in the
simulator's internals). Which of the following would catch it, and which would not:
(a) the point-in-time test, (b) the leakage gate `assert_no_leakage`, (c) a SHAP
share chart, (d) hold-out PR-AUC?

<details><summary>Answer</summary>

(b) only if the column name is in `leakage_cols`; it is a name-based gate, so a new
column slips through until someone adds it. (a) catches it only if the value depends
on events after the as-of date in a way the test's edits change; a precomputed
column would pass. (c) would show one feature with an absurd share (50%+), which is
the practical catch. (d) would show PR-AUC near 1.0, which is the other practical
catch. Moral: name-based gates and date filters are necessary; a human looking at
the attribution and the headline number is the last line.
</details>
