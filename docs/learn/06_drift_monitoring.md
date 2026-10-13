# 06 · Drift: what moves, what you can see, and when

*What this note covers:* the three kinds of drift and why each needs a different
monitor; PSI and KS with the numbers worked by hand; why a feature can drift without
the model caring and the model can break without any feature drifting; why the only
honest check arrives four weeks late; and how the retrain decision becomes a branch
in a DAG. Everything here runs in `churnprop.monitor` on the simulated year in
`results/monitoring/`.

## 1. Three things that can move

Take the production model: `P(churn in 4 weeks | features)`. Between training and
this week's batch, three different things can change.

| Name | What changed | Example in this repository (`configs/default.yaml: batches.drift`) | Which monitor sees it |
|---|---|---|---|
| **Covariate shift** | The distribution of the features, `P(x)`, for rows that exist | Fibre bills rise 12% from week 5 | Feature PSI/KS, score PSI |
| **Population drift** | The mix of customers; no row changes | Acquisition tilts to fibre from week 8 (`weight: 1.8`) | Category-share PSI, score PSI |
| **Concept drift** | `P(churn | x)` itself | A competitor campaign: 25% of electronic-check month-to-month non-churners flip from week 7 | Only the matured-label checks |

The first two are visible in the batch the moment it lands. The third is invisible
in the inputs by construction, and that is the one that costs money.

## 2. PSI, worked by hand

The population stability index compares two histograms on the same bins:

```
PSI = Σ_bins (cur_b − ref_b) · ln(cur_b / ref_b)
```

Reference bins are **fixed at training time** (deciles of the training frame), so
each training bin holds 10% of customers. Take `Monthly Charges` with a 12% price
rise on fibre customers (44% of the book). Roughly, fibre customers in each bin move
up by one bin:

| bin | ref | cur (after rise) | (cur−ref)·ln(cur/ref) |
|---|---|---|---|
| 1 (lowest) | 0.10 | 0.10 | 0 (DSL and no-internet bins unaffected) |
| 5 | 0.10 | 0.08 | (−0.02)·ln(0.8) = 0.0045 |
| 6 | 0.10 | 0.07 | (−0.03)·ln(0.7) = 0.0107 |
| 7 | 0.10 | 0.09 | 0.0011 |
| 8 | 0.10 | 0.11 | 0.0010 |
| 9 | 0.10 | 0.13 | 0.0079 |
| 10 (highest) | 0.10 | 0.15 | 0.0203 |

Sum ≈ **0.05**: a *warn* at best under the usual 0.10/0.25 thresholds. That is what
the fast run shows for weeks 5–7. When the mix shift stacks on top (more fibre rows
*and* each paying more) the top bins fill further and PSI on `Monthly Charges`
crosses 0.25 at week 8: an alert.

Two things to notice. PSI is symmetric in the sense that it does not know which
direction is bad; and it is blind to the *meaning* of the feature. A PSI of 0.3 on a
feature the model barely uses is noise; a PSI of 0.1 on the top SHAP driver is not.
That is why figure 17 orders rows by maximum PSI and the README reads it next to the
SHAP table.

**Exercise 1.** The reference has 10 equal bins. This week, every customer's value is
exactly as before, but the batch is drawn from a new acquisition channel so 30% of
rows fall in the top bin and the rest spread evenly over the other nine. Compute PSI.
Is this covariate shift or population drift, and does the distinction matter to the
score?

<details><summary>Answer</summary>

Top bin: (0.30 − 0.10)·ln(3) = 0.220. Each other bin: cur = 0.70/9 = 0.0778,
(0.0778 − 0.10)·ln(0.778) = 0.0056, times nine = 0.050. PSI ≈ **0.27**: an alert.
It is population drift (rows are unchanged). To the *score* the distinction does not
matter: the model is being asked about customers it rarely saw, and its calibration
in that region is untested either way. To the *remedy* it matters: for population
drift you may only need to re-weight or widen the training sample; for covariate
shift the feature's meaning (a $90 bill) may have changed.
</details>

## 3. KS, and the trap with discrete features

The Kolmogorov–Smirnov statistic is the largest vertical gap between two empirical
CDFs. Monitoring stores 21 reference quantiles and evaluates both CDFs there. For a
feature like `tickets_90d`, where 80% of customers have 0, the 1st to 16th quantiles
are all 0 and the naive version of the check compares `F_cur(0) = 0.8` against the
*grid level* 0.05, 0.10, …: a KS of 0.75 on a feature that did not move. The first
smoke run of this repository alerted on exactly that. The fix is to store the
reference ECDF at each quantile (`quantile_cdf` in `reference_stats.json`) and
compare `F_cur(q)` with `F_ref(q)`, which is 0.8 against 0.8.

**Exercise 2.** The reference for `late_payments_90d` is 85% zeros, 12% ones, 3% twos.
This week it is 75% zeros, 18% ones, 7% twos. What is KS, and roughly what PSI on
three bins {0, 1, ≥2}?

<details><summary>Answer</summary>

CDFs at 0: 0.85 vs 0.75 (gap 0.10); at 1: 0.97 vs 0.93 (gap 0.04). **KS = 0.10**,
under the 0.15 alert. PSI = (0.75−0.85)·ln(0.882) + (0.18−0.12)·ln(1.5) +
(0.07−0.03)·ln(2.33) = 0.0126 + 0.0243 + 0.0339 ≈ **0.07**: under warn. A 50% rise in
late payers does not trip the input monitors. If this feature matters (it does: the
planted churn signal lives in it), the *score* and the *matured outcomes* are where
it shows up, which is the next point.
</details>

## 4. Inputs can move without the score moving, and the reverse

Panel 1 of figure 18: the worst feature PSI climbs to 0.33 while the **score PSI stays
near 0.04**. The price rise moves `Monthly Charges` but the model's view of churn
barely changes, because most of the moved customers were already in the high-risk
region and tree splits are coarse there. Feature drift was an alert; the score said
"so what". Conversely, the concept drift in week 7 changes no input and no score,
and the first evidence is a batch whose observed churn comes in above its predicted
churn four weeks later (panel 2), with a z-score on the gap so a 2,000-row batch's
noise does not page anyone.

This is why the monitors are ordered the way they are in `monitor.py`: inputs (cheap,
immediate, noisy in meaning), scores (immediate, closer to the decision), outcomes
(late, the truth), uplift (later still, the money).

## 5. Why the retrain decision is a branch and not a cron

Retraining on every alert is wrong in both directions. Retrain on the price rise and
you fit a model to a book where the most expensive customers are the fibre ones who
just got a price rise and have not yet had time to churn: the labels are stale for
exactly the segment that moved. Do not retrain on the concept drift and you keep
under-predicting churn in the electronic-check segment for as long as nobody looks.

So the DAG (`dags/retention_weekly.py`) does three things: it always scores and
monitors; it branches to `retrain` only when a check alerts; and `retrain` goes
through the **gate** (D20), which refuses the candidate if it is not at least as good
on the hold-out. The retrain is a proposal the gate can decline, and the week's offer
plan is published either way (`publish_plan` runs on `none_failed_min_one_success`).

**Exercise 3.** Week 8 alerts on `Monthly Charges` PSI 0.27. The DAG retrains; the
candidate's hold-out PR-AUC is 0.694 against production's 0.698; the gate threshold is
−0.005. What happens, and is it the right outcome?

<details><summary>Answer</summary>

−0.004 ≥ −0.005, so the gate passes and the candidate is promoted. Whether that is
right depends on what the retrain used: the hold-out is the *training-time* split,
so a model retrained on the same data with a new seed is not "fresher", it is just
different. A real retrain adds the matured weeks 1–4 batches (with their labels) to
the training set and holds out the most recent matured week; then the gate compares
on data that includes the drift. The repository's `train_candidate` fits on the
snapshot split, so the DAG's retrain demonstrates the mechanics (train → gate →
promote → export → commit) and the docstring says what a production version adds.
</details>

## Further reading

* Gama et al. (2014), *A survey on concept drift adaptation*: the covariate /
  prior / concept taxonomy used above.
* Yurdakul (2018), *Statistical properties of population stability index*: where
  the 0.10 / 0.25 rules of thumb come from, and how badly they depend on sample size.
* Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*:
  why the monitoring, the gate and the committed export exist at all.
