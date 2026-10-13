# 05 · Spending a weekly budget, and finding out afterwards whether it worked

*What this note covers:* turning "which offer, if any, for each customer" into a
budget problem; why one number (a Lagrange multiplier) solves it; how a randomised
holdout lets you value a policy you never ran (IPS and doubly robust estimation);
and the power calculation that says when the holdout will actually tell you
anything. Everything here runs in `churnprop.uplift` and the weekly loop.

## 1. The decision each week

For every active customer *i* and every arm *k* (control plus three offers) the
learner gives an estimate of the **expected value** of that arm versus doing nothing:

```
ev[i, k] = (P(churn | control) − P(churn | offer k)) × value_i − cost_k
         =  predicted uplift_k(i)  × CLTV_i             − cost_k
```

`ev[i, 0] = 0` by definition. Examples from the simulator at week 6:

| customer | bill credit ($45) | tech bundle ($60) | contract upgrade ($70) | best arm |
|---|---|---|---|---|
| A: fibre, no support, $95/month, 14 months | 0.06 × 4,100 − 45 = **201** | 0.11 × 4,100 − 60 = **391** | 0.07 × 4,100 − 70 = 217 | tech bundle |
| B: DSL, 2 months, $45/month | −0.05 × 3,200 − 45 = −205 | 0.01 × 3,200 − 60 = −28 | −0.06 × 3,200 − 70 = −262 | **nothing** (sleeping dog) |
| C: month-to-month, 20 months, $70/month | 0.03 × 5,000 − 45 = 105 | 0.02 × 5,000 − 60 = 40 | 0.09 × 5,000 − 70 = **380** | contract upgrade |

With no budget you would give each customer their best arm if its `ev` is positive.
With a budget of $32,000 a week and ~5,000 eligible customers you cannot: the sum of
the best arms' costs is far more than that.

## 2. One number that solves it: the Lagrange multiplier

The exact problem is an integer programme (one arm per customer, total cost ≤ B).
The trick is to charge every dollar of cost a **price λ** and let each customer be
solved on their own:

```
score[i, k] = ev[i, k] − λ · cost_k           pick arm_i = argmax_k score[i, k], or 0 if all ≤ 0
```

* λ = 0: everyone with a positive `ev` gets an offer (the unconstrained answer).
* λ large: only offers whose value is many times their cost survive.
* Spend is a decreasing function of λ, so **bisection on λ** finds the value where
  spend just fits the budget. That is all `allocate_budget` does.

**Worked example.** Three customers, one offer costing $50, `ev` = 300, 120, 20,
budget $100 (two offers at most).

| λ | scores | offers given | spend |
|---|---|---|---|
| 0 | 300, 120, 20 | 3 | 150 ✗ |
| 1 | 250, 70, −30 | 2 | 100 ✓ |
| 3 | 150, −30, −130 | 1 | 50 (under-spent) |

λ = 1 is the answer: "an offer must return at least **$1 of value per $1 of cost
beyond its own cost**", and the cheapest way to say "we are out of money" is to raise
that bar. λ has a meaning you can report to finance: in `results/budget_frontier.csv`
λ is 5.6 at a $10k budget and 0.37 at $80k, which reads as "at $10k we are turning
away offers that return $5 per $1; at $80k the marginal offer barely pays".

Why it is (almost) optimal: relaxing "one arm per customer" to "fractions of arms"
gives a linear programme, and the λ rule is exactly its dual solution; the two can
differ by at most one customer's worth of value. `lp_allocation` checks this each
run (gap 0.2% on 2,500 customers, `headline_phase3.json: lp_gap`).

**Exercise 1.** Budget $100, one offer costing $50, `ev` = 300, 120, 80, 20. Give the
λ that spends exactly the budget, and the one customer whose offer the LP relaxation
would split.

<details><summary>Answer</summary>

Any λ in (1.6, 2.4] gives offers to the first two only (scores 300−50λ, 120−50λ,
80−50λ: the third goes non-positive at λ = 1.6, the second at 2.4). Spend $100. The
LP would give the third customer a fractional offer only if money were left, and
here none is, so the gap is zero. With budget $125 the LP gives customer 3 half an
offer (value 40) that the integer rule cannot: that half-offer is the whole gap.
</details>

**Exercise 2.** The uplift policy in figure 14 sends *fewer* offers than the budget
allows in later weeks, while risk_top always spends everything. Why, and which of
the two is the mistake?

<details><summary>Answer</summary>

The allocator never gives an arm whose `ev − λ·cost` is ≤ 0, and λ cannot go below
0, so when the book of positive-`ev` customers is smaller than the budget it stops.
Spending the rest would mean paying to make churn *more* likely (sleeping dogs) or
paying for customers who stay anyway. risk_top has no notion of "this offer does not
pay", so spending everything is the mistake; the honest report is "we under-spent
because the remaining offers had negative expected value", which is a finding, not
a failure.
</details>

## 3. Valuing a policy you did not run: IPS

Each week 8% of eligible customers are in the **exploration holdout**: their arm is
drawn uniformly at random (probability 1/4 each) and logged with that probability.
Four weeks later we know the reward: `r = −cost − churned × CLTV`.

Suppose you now want the value of a different rule π (say, "risk_top") without
running it. For each exploration row ask: *did the coin happen to give what π would
have given?* If so, count the reward, scaled up by 1/p because such rows are only a
quarter of the ones π would have treated:

```
V_IPS(π) = mean over rows of  1{a == π(x)} / p · r
```

**Toy example.** Four rows, two arms, p = ½ each:

| row | a | r | π(x) | matches? | contribution |
|---|---|---|---|---|---|
| 1 | 0 | −1,000 | 1 | no | 0 |
| 2 | 1 | −45 | 1 | yes | −45 / 0.5 = −90 |
| 3 | 1 | −3,045 | 1 | yes | −3,045 / 0.5 = −6,090 |
| 4 | 0 | 0 | 0 | yes | 0 / 0.5 = 0 |

V_IPS = (0 − 90 − 6,090 + 0) / 4 = **−1,545 per row**. Unbiased: in expectation the
matching rows are a random half of each arm's rows. But look at row 3: one churner
with a $3,000 CLTV moved the estimate by $1,500. That is the whole problem.

## 4. Doubly robust: borrow strength from a model

Fit an outcome model `m(x, k) ≈ E[r | x, k]` on the holdout rows (cross-fitted so a
row never scores itself), then

```
V_DR(π) = mean of  m(x, π(x))  +  1{a == π(x)} / p · ( r − m(x, a) )
```

Read it as "the model's guess for the policy's arm, corrected by the residual on rows
where we actually observed that arm". If the model is right the correction averages
to zero and the variance is the model's; if the model is wrong the correction fixes
it (because the propensities are right). Here the propensities are known exactly, so
DR is purely a variance play.

**Exercise 3.** On the toy table, with a perfect model m(x, 1) = r for rows 2–3 and
m(x, 1) = −1,000 for row 1, m(x, 0) = r for rows 1 and 4, compute V_DR(π).

<details><summary>Answer</summary>

Row 1: m(x,1) = −1,000, no match, correction 0 → −1,000. Row 2: m = −45, match,
residual 0 → −45. Row 3: −3,045 + 0 → −3,045. Row 4: m(x,0) = 0, match, residual 0 →
0. Mean = (−1,000 − 45 − 3,045 + 0)/4 = **−1,022.5**. Different from IPS (−1,545)
because the model supplies row 1's counterfactual instead of leaving it at zero and
inflating rows 2–3. With a perfect model the estimate is exact; the IPS one is noise
around it.
</details>

## 5. Why the holdout cannot see a $15k/week effect at 12 weeks

Run the numbers for this repository (`results/policy_comparison.csv`, uplift row):

* ~2,600 matured exploration rows over 9 weeks, ~650 per arm.
* The policy differs from "do nothing" on about 10% of rows (it treats ~330 of
  ~3,200 eligible customers a week). All other rows contribute exactly zero to
  V(π) − V(no offers). So the comparison rests on ~260 rows, ~65 of which happen to
  match the policy's arm.
* Per-row reward has a standard deviation around $800 (a ~3.5% churn chance times a
  ~$4,500 CLTV). The DR residual cannot remove that: it is the coin flip itself.
* Standard error of the per-week difference ≈ 800 × √(2/65) × 4 (the 1/p weight)
  × (330 treated rows a week) ≈ **$150k**. The truth is $15k.

So the 95% interval on the uplift policy's value is roughly ±$300k a week. A
two-sided test at 80% power needs the true effect to be about 2.8 standard errors,
and the standard error falls with √weeks, so detecting $15k/week takes on the order
of (2.8 × 150 / 15)² × 9 ≈ **7,000 weeks** at this holdout size. Figure 16 shows the
estimates with their intervals next to the planted truth.

What a real team does with this (all three are in the backlog, none is pretend):

1. **Evaluate churn, not dollars**, for the significance test (removes the CLTV
   spread), and report dollars descriptively.
2. **Enlarge the holdout where the policy acts**: randomise 30% of the customers the
   policy *wants* to treat, instead of 8% of everyone. Most exploration rows above
   were control-vs-control, which carry no information about the policy.
3. **Pool across weeks and run longer**: the simulator can be run for 26 weeks
   (`--offers` with `horizon.n_weeks: 26`, events already cover it).

**Exercise 4.** The programme switches to randomising 30% of the ~330 customers the
policy wants to treat each week (and nobody else), still uniform over 4 arms. By what
factor does the standard error of V(π) − V(no offers) shrink, roughly, at the same
number of weeks?

<details><summary>Answer</summary>

Informative rows go from ~260 (8% of 3,200, then the ~10% the policy would treat) to
~100 a week × 9 weeks ≈ 900 over the same period, all of them on customers the policy
would treat, so the rows that matter triple-and-a-half: standard error falls by about
√(900/260) ≈ 1.9. Still not enough on its own (≈ $80k against $15k); combined with
testing on churn rather than dollars (another factor of 2–3) and 26 weeks (√2.9 ≈ 1.7)
you approach a detectable effect. The lesson is that evaluation design is a
sample-size problem before it is an estimator problem.
</details>

## Further reading

* Dudík, Langford & Li (2011), *Doubly Robust Policy Evaluation and Learning*: the DR
  estimator in §4 above.
* Boyd & Vandenberghe, *Convex Optimization*, §5: Lagrangian duality; the one-knob
  allocation is the knapsack special case.
* The companion repo's `docs/` on CUPED and effect durability, which attack the same
  variance problem from the experiment-design side.
