# 01 · Class imbalance, SMOTE, and which metric to believe

*What this note covers:* why "87% accuracy" was never a real number on this data,
what SMOTE actually does to a classifier, why it hurt here, and how to pick metrics
when the scores feed a budget decision. Worked on the numbers in `results/`.

## 1. The 73.5% model

26.5% of the 7,043 customers churned. A model that says "nobody churns" is right
73.5% of the time. So accuracy starts at 73.5 before the model has learned anything,
and the 80% the best models reach on the hold-out is really "6.5 points better than
doing nothing".

**Intuition.** Accuracy weighs every customer equally. The business does not: a
missed churner costs the margin on that customer for the rest of their life; a false
alarm costs one offer. When the costs differ, a metric that counts both errors as one
"mistake" cannot rank models for you.

**Exercise 1.** In the hold-out there are 1,409 customers, 374 of whom churn. A model
flags 300 customers, 200 of whom actually churn. Compute accuracy, precision, recall.
Then compute them for "flag nobody". Which pair of numbers tells you the model is
useful, and which hides it?

<details><summary>Answer</summary>

Model: TP = 200, FP = 100, FN = 174, TN = 935. Accuracy = (200 + 935) / 1409 = 80.6%.
Precision = 200/300 = 66.7%. Recall = 200/374 = 53.5%.
Flag nobody: accuracy = 1035/1409 = 73.5%, precision undefined, recall = 0.
Accuracy moved 7 points; recall moved from 0 to 53%. Recall and precision describe
what marketing would actually get (a list of 300 names, two thirds of them right).
</details>

## 2. What SMOTE does, mechanically

SMOTE (Synthetic Minority Over-sampling TEchnique) takes a minority-class row, picks
one of its *k* nearest minority neighbours, and creates a new row on the straight
line between them. Repeat until the classes are balanced. Two consequences:

1. **The training class prior becomes 50/50.** A model trained on balanced data
   learns P(churn | x) *as if* half of customers churn. Its probabilities are shifted
   up. On the real 26.5% population, a customer the model calls "0.6" might churn
   35% of the time. The *ranking* can survive this; the *probabilities* do not.
2. **If you oversample before splitting, synthetic copies of a test customer leak
   into training.** Two points on the line between customer A (train) and customer B
   (test) look almost exactly like B. The model has effectively seen the test set.
   That is where numbers like 87% come from.

**Intuition for 1.** Imagine a doctor trained in a hospital where half the patients
have the disease, then sent to a clinic where 1 in 4 do. Their *ordering* of who looks
sickest is still fine; their stated "60% chance" is now too high.

**What the benchmark found** (`results/benchmark_cv.csv`, figure 03): for every one
of the eight models, SMOTE *lowered* CV PR-AUC a little and *raised* the Brier score
a lot (worse calibration). Class weights sit in between. Gradient boosting with no
resampling at all was the best-calibrated ranker. So the shipped model uses `none`
(decision D3) and SMOTE stays in the benchmark as the control that shows why.

**Exercise 2.** Model A has CV Brier 0.129 and PR-AUC 0.687 (no resampling). Model
A′ is the same model with SMOTE: Brier 0.142, PR-AUC 0.675. The decision layer will
compute *expected revenue saved = P(churn) × CLTV × save-rate − cost* for each
customer and only contact those with positive value. Which model do you ship, and
what goes wrong if you ship the other one?

<details><summary>Answer</summary>

Ship A. The decision multiplies the probability by dollars, so a probability that is
systematically too high (A′) makes marginal customers look profitable to contact
when they are not; you overspend the budget on false positives. The slightly worse
PR-AUC of A′ is a second, smaller reason. If you only ever used the scores to sort a
list and cut at a fixed size, A′ would be nearly as good; it is the *dollar*
conversion that makes calibration decisive.
</details>

## 3. Choosing metrics by what the score is used for

| The score is used to… | Metric | Why |
|---|---|---|
| Sort everyone, contact the top *k* | **Lift @ top decile**, ROC-AUC | Only the ordering matters |
| Build a list with a quality floor ("at least half must be real churners") | **Recall @ precision ≥ 0.5** | That is literally the constraint |
| Compare models when positives are rare | **PR-AUC** | ROC-AUC is flattered by the many easy negatives |
| Multiply by dollars | **Brier, ECE, reliability curve** | The number must mean what it says |
| Talk to the old README | Accuracy, F1 @ 0.5 | Only with "nobody churns = 73.5%" next to it |

**Why PR-AUC and not ROC-AUC on imbalanced data.** ROC-AUC asks: pick a random
churner and a random non-churner, how often does the churner score higher? With
three non-churners for every churner, most of those pairs are easy. PR-AUC asks: as
I walk down the ranked list, what share of the names so far are churners? It
punishes the false positives near the top of the list, which are exactly the ones
that waste offer budget.

**Exercise 3.** Two models have identical ROC-AUC (0.85). Model X puts 60 churners
in its top 100; model Y puts 45 churners in its top 100 but does better in the
middle of the ranking. Which has the higher PR-AUC, and which would a retention
team with budget for 100 offers prefer?

<details><summary>Answer</summary>

X on both counts. PR-AUC is dominated by precision at the top of the list; the
middle of the ranking barely moves it. A team that can only make 100 offers lives in
the top 100 and does not care about the middle. ROC-AUC averages over the whole
ranking and so cannot see the difference.
</details>

## 4. Reading the reliability curve

Figure 05 bins hold-out customers by predicted probability and plots the observed
churn rate per bin. A calibrated model sits on the diagonal. Under-confidence bows
the curve above the diagonal at the low end (predicting 0.1, observing 0.2);
SMOTE-trained models bow *below* it at the high end (predicting 0.7, observing 0.5).
ECE in the tables is the bin-weighted average distance from the diagonal.

**Exercise 4.** A model's reliability table shows: bin 0.6–0.7 has 120 customers,
mean predicted 0.65, observed rate 0.48. If CLTV averages $4,400 and the save rate
is 30%, by how much does the model overstate the expected revenue saved per
customer in that bin?

<details><summary>Answer</summary>

Predicted: 0.65 × 4,400 × 0.30 = $858. Actual: 0.48 × 4,400 × 0.30 = $634.
Overstated by $224 per customer, about 35%. Across 120 customers that is $27k of
imaginary value in one bin, which is the kind of gap that makes a campaign "beat
its forecast" on paper and miss in the ledger.
</details>

## Further reading

* Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*.
* Saito & Rehmsmeier (2015), *The precision-recall plot is more informative than the
  ROC plot when evaluating binary classifiers on imbalanced datasets*.
* van den Goorbergh et al. (2022), *The harm of class imbalance corrections for risk
  prediction models*, JAMIA. The clinical-prediction literature reached the same
  conclusion as this benchmark: resampling damages calibration and rarely helps
  discrimination.
