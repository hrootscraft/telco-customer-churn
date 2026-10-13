# 03 · Propensity is not uplift: who is leaving vs who an offer will save

*What this note covers:* the four customer types every retention programme has,
why ranking by churn probability targets the wrong two of them, the quantity you
actually want (the uplift, or conditional average treatment effect), why it cannot
be estimated from this snapshot alone, and how phases 2–3 of this repository and the
companion project [tierlift](https://github.com/hrootscraft/revenue-tier-and-churn-experiment)
divide the problem.

## 1. Four kinds of customer

Think about one customer and one offer (say, a $45 bill credit). There are two
worlds: the one where they get the offer and the one where they do not. In each
world they either stay or leave.

| | Leaves without offer | Stays without offer |
|---|---|---|
| **Stays with offer** | **Persuadable** (the offer saved them) | **Sure thing** (would have stayed anyway; offer is wasted) |
| **Leaves with offer** | **Lost cause** (offer is wasted) | **Sleeping dog** (the offer *caused* them to leave) |

A churn-propensity model ranks customers by P(leave without offer). The top of that
list is full of **lost causes and persuadables mixed together**, and it cannot tell
them apart: both have high churn probability. The bottom of the list hides the
**sleeping dogs**, customers who were fine until a "we noticed you might leave"
email reminded them to shop around. On a real telco book, two-year customers nearing
renewal are the classic sleeping dogs.

**Intuition.** Propensity answers "how sick is this patient?" Uplift answers "will
*this drug* help *this patient*?" A hospital that gives the drug to the sickest
patients regardless of whether it works on them is not running a treatment
programme; it is running a triage list.

**Exercise 1.** A retention team has budget for 1,000 offers. The propensity model's
top 1,000 are 40% persuadable, 55% lost cause, 5% sure thing. The uplift model's top
1,000 are 70% persuadable, 15% lost cause, 15% sure thing. Each saved customer is
worth $400 of margin; each offer costs $45. Compute the net value of each list.

<details><summary>Answer</summary>

Only persuadables generate value. Propensity list: 400 × $400 − 1,000 × $45 =
$160k − $45k = **$115k**. Uplift list: 700 × $400 − $45k = $280k − $45k = **$235k**.
Same budget, roughly double the return, and the uplift list *also* contacted fewer
people who were about to leave anyway, which the propensity model would have counted
as "high-risk customers reached".
</details>

## 2. The quantity you want

For customer *i* with features *x*, define Y(1) = stays if offered, Y(0) = stays if
not. The **uplift** (CATE, conditional average treatment effect) is

τ(x) = E[Y(1) − Y(0) | x] = P(stay | offer, x) − P(stay | no offer, x).

The fundamental problem: you observe one of Y(1), Y(0) per customer, never both. So
τ cannot be computed row by row; it has to be *estimated* from groups of similar
customers, some of whom were offered and some not, **where the offer was assigned
in a way unrelated to the outcome**. That last clause is why a randomised holdout
is not a nice-to-have but the data-generating mechanism for an uplift model.

## 3. Why this snapshot cannot give you τ

The Telco data has no treatment column. Nobody was offered anything (or if they
were, it is not recorded). So from this file alone you can estimate P(churn | x)
and nothing about what changes it. Any claim of the form "customers with tech
support churn 15 points less, so giving tech support saves 15 points" is the
sleeping-dog mistake in reverse: the people who bought tech support differ from
those who did not in ways the model does not see.

**Exercise 2.** In the data, customers paying by electronic check churn at 45%;
those on automatic bank transfer churn at 17%. Give two different stories, one
causal and one not, that both produce this gap. What single piece of data would
tell them apart?

<details><summary>Answer</summary>

Causal: electronic check requires a manual monthly action, each one a moment to
reconsider; auto-pay removes those moments and so reduces churn. Non-causal:
customers who are already uncommitted (new, no partner, planning to leave) avoid
auto-pay precisely because they expect to leave; the payment method is a symptom.
To tell them apart you need an *assignment* that is independent of commitment: a
randomised auto-pay incentive offered to half of eligible customers, with churn
compared against the untreated half.
</details>

## 4. How the two repositories divide the problem

| | tierlift | this repository (phases 2–3) |
|---|---|---|
| Data | Simulated clients, one randomised experiment, planted truth | 7,043 real customers; only their *responses to offers* are simulated |
| Treatment | One incentive (plus an information-only arm) vs holdout | Three offers with different costs plus control |
| Question | Which uplift learner ranks clients best? For whom and how long does the effect last? | Which offer, if any, does each customer get this week, under a fixed budget? |
| Methods | S/T/X/R/DR meta-learners vs causal forest; Qini, AUUC; CUPED, AIPW; survival | One multi-treatment learner; Lagrangian budget allocation; off-policy evaluation from the exploration holdout |
| Timing | One experiment, analysed afterwards | Twelve weekly cycles, each retrained on the logged data so far |

tierlift is the *measurement* project: it shows that a heterogeneous effect can be
estimated and which estimator does it best. This repository is the *operations*
project: it takes a propensity model that is honest about what it can and cannot
say, puts a decision layer on top that needs an uplift estimate, and runs the loop
that produces the data to estimate it. The 10% random exploration holdout in every
weekly cycle is the price of being able to learn τ at all; without it the targeting
policy only ever sees outcomes for the customers it chose, and its training data
becomes a mirror of its own biases.

**Exercise 3.** Week 1's policy sends the bill credit only to customers with
P(churn) > 0.6. In week 2 you train an uplift model on week 1's outcomes with no
holdout. What can the model learn about the bill credit's effect on customers with
P(churn) = 0.3? What does the exploration holdout fix?

<details><summary>Answer</summary>

Nothing. No customer below 0.6 received the offer, so there is no treated-vs-control
contrast in that region; any estimate there is extrapolation from the tree
structure, not evidence. The holdout assigns a random 10% of *all* customers to
arms (including control) regardless of their score, so every region of feature
space has some treated and some untreated customers, and the weights 1/P(assignment)
make those rows representative. That is what lets the week-2 model discover that
the bill credit works best at 0.3–0.5 (price-sensitive but not yet decided), which
is in fact where it is planted in `configs/offers.yaml`.
</details>

## Further reading

* Radcliffe & Surry (2011), *Real-World Uplift Modelling with Significance-Based
  Uplift Trees*: the four-quadrant picture and the Qini curve.
* Künzel et al. (2019), *Metalearners for estimating heterogeneous treatment effects
  using machine learning*: S/T/X-learners (benchmarked in tierlift).
* Dudík, Langford & Li (2011), *Doubly Robust Policy Evaluation and Learning*: how
  phase 3 values a targeting policy from logged data.
