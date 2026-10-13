# 02 · SHAP at three altitudes: the book, a segment, one customer

*What this note covers:* what a SHAP value is (with a two-feature example you can
do by hand), why one-hot columns must be folded back before anyone reads them, what
"mean SHAP within a segment" buys an offer owner that global importance does not,
and the two mistakes people make when they read SHAP as causality.

## 1. The idea, by hand

A model's prediction for one customer is the base value (the average prediction)
plus one contribution per feature. SHAP (SHapley Additive exPlanations) defines
each feature's contribution as its average marginal effect over every order in
which the features could have been "revealed" to the model.

**Two-feature example.** Say the model's output (in log-odds) depends on
`Contract` (M = month-to-month, else 2-year) and `Fibre` (yes/no):

| Contract | Fibre | model output |
|---|---|---|
| 2-year | no | −2.0 |
| 2-year | yes | −1.5 |
| M | no | −0.5 |
| M | yes | +0.5 |

Base value (average over the four cells, equal weights) = −0.875.

Customer: month-to-month, fibre. Output +0.5. Total to explain: 0.5 − (−0.875) = 1.375.

* Order "Contract first, then Fibre": revealing Contract = M moves the expectation
  from −0.875 to the average of the two M rows, 0.0 (Δ = +0.875). Then revealing
  Fibre = yes moves it from 0.0 to +0.5 (Δ = +0.5).
* Order "Fibre first": revealing Fibre = yes moves −0.875 to the average of the two
  fibre rows, −0.5 (Δ = +0.375). Then Contract = M moves −0.5 to +0.5 (Δ = +1.0).

SHAP(Contract) = mean(0.875, 1.0) = **0.9375**. SHAP(Fibre) = mean(0.5, 0.375) =
**0.4375**. They sum to 1.375, as they must.

**Exercise 1.** Compute the two SHAP values for the customer *2-year, fibre*.
Check they sum to the output minus the base value.

<details><summary>Answer</summary>

Output −1.5; total to explain −0.625. Contract first: 2-year → avg(−2.0, −1.5) =
−1.75 (Δ −0.875); then fibre → −1.5 (Δ +0.25). Fibre first: → −0.5 (Δ +0.375); then
2-year → −1.5 (Δ −1.0). SHAP(Contract) = mean(−0.875, −1.0) = −0.9375;
SHAP(Fibre) = mean(0.25, 0.375) = 0.3125. Sum = −0.625. ✓
</details>

With 22 features there are 22! orders; `TreeExplainer` computes the exact average
for tree ensembles in polynomial time by walking the trees, which is why SHAP is
computed on LightGBM here (D7) rather than on the neural net (which would need the
slower, approximate `KernelExplainer` or `DeepExplainer`).

## 2. Fold the one-hot columns back

The model sees `Contract_Month-to-month`, `Contract_One year`, `Contract_Two year`
as three columns and gives each a SHAP value. Shown that way, "Contract" looks like
three medium features instead of one dominant one, and a reader who ranks by column
will underrate it. SHAP values are additive, so summing the three columns gives the
exact contribution of the *feature* `Contract`. `explain.shap_values_folded` does
this for every categorical. After folding, Contract carries about 28% of the total
|SHAP| on the hold-out (`results/shap_global.csv`), which is the figure the README
quotes.

## 3. Three altitudes

**Global** (`shap_global.csv`, figure 06): mean |SHAP| per feature. Answers "what
does the model pay attention to?" Useful for a model card and for catching leakage
(a feature with 60% of the attention is a red flag).

**Segment** (`shap_by_segment.csv`, figures 08–09): mean *signed* SHAP per feature
*within* a segment. This is the altitude an offer owner works at. Within
month-to-month customers the Contract feature is a constant, so it is no longer what
separates the risky from the safe; what remains is payment method, paperless
billing, tenure and add-ons. Those are the levers an offer can pull (you cannot
retroactively give someone a two-year contract, but you can offer one). Within
two-year customers almost everything pulls risk *down*; they are not where a budget
goes.

**Customer** (`shap_customers.csv`, the app's waterfall): the full contribution
vector for one person. Answers "why is *this* customer at 0.81?" and gives the
retention agent a script: "you are paying $95/month on month-to-month with no tech
support".

**Exercise 2.** Global importance says `Dependents` is the second feature. Within
the month-to-month segment its mean signed SHAP is small and negative. Reconcile
the two.

<details><summary>Answer</summary>

Global importance is mean |SHAP|: the feature moves predictions a lot in *both*
directions (customers with dependents down, without up). Within month-to-month the
signed mean nets those out and reports the segment's average push, which is close
to zero. Both are true; the segment view is what tells you Dependents is not a
*lever* in that segment (it is a fact about the household, not something an offer
changes).
</details>

## 4. Two things SHAP is not

**Not causal.** SHAP explains the model, not the customer. "Contract has the largest
SHAP" means the model leans on it, and the model leans on it because month-to-month
customers churn at 42.7% vs 2.8%. It does *not* say that moving a customer to a
two-year contract would cut their churn probability by that gap: the people who
chose two-year contracts differ in ways the model cannot see. That is the question
phase 3 (offers, holdout, uplift) answers, and it is why
[`03_propensity_vs_uplift.md`](03_propensity_vs_uplift.md) exists.

**Not stable under correlated features.** `Tenure Months` and `tenure_band` carry
the same signal; SHAP will split it between them in a way that depends on tree
structure. Read correlated features as a group. (The stepwise table, where tenure
survives as one or the other, shows the same thing from the statistics side.)

**Exercise 3.** A stakeholder sees the segment chart and says: "Electronic check is
the top risk driver for month-to-month customers, so let's move everyone to
auto-pay." What would you check before agreeing, and what experiment would settle it?

<details><summary>Answer</summary>

Check whether electronic-check customers differ in ways that also predict churn
(newer, no partner, higher charges) so the payment method is a *marker* rather
than a cause. Then run a randomised test: offer an auto-pay incentive to a random
half of eligible month-to-month customers, hold the rest out, and compare churn
over the next 90 days. The SHAP chart tells you where to *look* for a lever; only
the holdout tells you whether it *is* one.
</details>
