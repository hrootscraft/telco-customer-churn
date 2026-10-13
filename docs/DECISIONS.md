# Decisions

Each entry: the choice, the alternatives, and why. Numbered so the README and the
learning notes can cite them. "Phase" refers to the build order in the README.

## Phase 1: data, benchmark, explanation

**D1. The 33-column IBM file, not the 21-column Kaggle cut.** Same 7,043 customers.
The IBM version adds `CLTV`, `Churn Score` and `Churn Reason`. The first is what the
decision layer needs to turn a probability into a dollar value; the other two are a
textbook leakage trap and are the reason `data.assert_no_leakage` exists. Keeping
them in the raw file and *refusing* them in the feature matrix is more instructive
than deleting them.

**D2. One hold-out, repeated CV on the rest, and the hold-out is scored once.**
80/20 stratified. All model selection (regime, hyper-parameters, feature subset)
happens on the 80% with 2 × 5-fold stratified CV. The 20% produces the table the
README quotes. The 2023 notebook reported "87% accuracy" from a SMOTE-balanced
split, which is not a number any customer-facing team could reproduce; the rule
here is that a number is only quotable if it came from rows the model never saw in
any form.

**D3. Resampling regime: `none` for the model that ships; SMOTE and class weights
are compared, not assumed.** SMOTE is on the resume so it stays in the benchmark,
applied inside each CV fold through `imblearn.Pipeline` (synthetic rows are made
from the training fold only). On this data it does not help: for every model the
`none` regime has equal or better CV PR-AUC and clearly better Brier score, because
oversampling shifts the predicted probabilities toward 0.5 and breaks calibration.
Since the decision layer multiplies probabilities by dollars, calibration matters
more than the 0.5-threshold metrics SMOTE was invented to improve. The learning note
`docs/learn/01_imbalance_and_metrics.md` works through why. Class weights behave
like a milder SMOTE. See `results/benchmark_cv.csv` and
`reports/figures/03_resampling_regimes.png`.

**D4. Metrics: rank first, then calibration, accuracy last.** Primary ROC-AUC and
PR-AUC; operational lift @ top decile and recall @ precision ≥ 0.5; calibration via
Brier and ECE; accuracy and F1 at 0.5 reported for comparability with the old
README and with the warning that "nobody churns" scores 73.5%.

**D5. Backward stepwise done as a logistic likelihood-ratio test, by feature
block.** The 2023 code fitted OLS to a 0/1 target and dropped columns one one-hot
level at a time. Here the model is logistic, the test is a likelihood-ratio test on
the whole block of a categorical feature, and it runs on the training split only.
It is compared with L1 regularisation and permutation importance. Finding: dropping
the stepwise-rejected features costs < 0.001 ROC-AUC, so stepwise is a parsimony
tool, not a performance lever.

**D6. PyTorch MLP wrapped as a scikit-learn estimator.** A wrapped estimator shares
the preprocessing, in-fold resampling and CV protocol of the other seven models, so
the comparison is like for like. Two hidden layers (64, 32), dropout 0.2, AdamW,
early stopping on a 10% slice of the training fold. TensorFlow was dropped: PyTorch
is the lighter dependency and the one the target roles name first.

**D7. SHAP on LightGBM, folded back to raw features.** `TreeExplainer` is exact and
fast on gradient-boosted trees; one-hot columns are summed back to their raw
feature so the output reads "Contract: 28% of the model's attention" rather than
"Contract_Two year". Segment drivers are mean *signed* SHAP within a segment, which
is what an offer owner needs ("for fibre customers, missing tech support is what
pushes risk up").

**D8. The engineered features are deliberately few.** `n_addon_services`,
`avg_monthly_charge`, `tenure_band`. Richer behavioural features come from the
event log in phase 2, where they can be computed as of a date; snapshot-only
features that encode the future would defeat that.
