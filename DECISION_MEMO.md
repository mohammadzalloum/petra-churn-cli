# Decision Memo

## Recommendation

I recommend deploying **RF_default** as the primary churn model for Petra Telecom. It delivered the strongest cross-validated ranking performance among the evaluated candidates, achieving the highest **PR-AUC mean of 0.5137**, which makes it the best option for prioritizing at-risk customers under class imbalance.

## Why This Model

The project compares six classification pipelines and selects the best one using **cross-validated PR-AUC** by default. That choice is appropriate for this problem because the dataset is imbalanced and Petra Telecom needs a model that ranks churn risk well rather than relying on accuracy alone.

The final selected model is:

- **Model:** `RF_default`
- **Selection metric:** `pr_auc_mean`
- **Threshold selection strategy:** `best_f1`
- **Selected threshold:** `0.20`

## Threshold Decision

The operating threshold was selected using **out-of-fold training predictions**, not the held-out test set. This is important because it avoids using test data to make deployment decisions and keeps the test set reserved for final evaluation.

At threshold **0.20**, the out-of-fold training threshold sweep produced:

- Precision: **0.3690**
- Recall: **0.6961**
- F1: **0.4824**
- Alert rate: **0.3086**
- Alerts per 1,000 customers: **308.6**

This threshold was chosen because it maximized out-of-fold F1 among the tested options.

## Final Test-Set Performance

Using **RF_default** at the selected threshold **0.20**, the held-out test set produced:

- Accuracy: **0.7389**
- Precision: **0.3406**
- Recall: **0.6395**
- F1: **0.4444**
- PR-AUC: **0.4484**
- Brier score: **0.1129**
- Predicted positives: **276 / 900**
- Alert rate: **0.3067**

Confusion matrix:

- **TN:** 571
- **FP:** 182
- **FN:** 53
- **TP:** 94

## Business Interpretation

This operating point is clearly **recall-oriented**. The model catches a substantial share of true churners, which is valuable if Petra Telecom wants broader retention coverage. The trade-off is that the model also generates a large number of false positives, which increases outreach cost and operational workload.

In practice, this means:

- the current setup is a good choice if the business prefers to catch more potential churners even at the cost of extra outreach
- the current setup is less attractive if the retention team has tight capacity constraints or if unnecessary interventions are expensive

A more aggressive alternative also exists:

- **Threshold 0.15** reaches about **0.815 recall** on out-of-fold training predictions
- however, it raises the alert rate to about **0.3983**, which would increase intervention volume even further

No capacity-constrained threshold candidate was identified under the current settings, so the recommended threshold is statistically strong but may still require operational review before production use.

## Model Behavior Insights

Error analysis shows that many high-confidence false positives share patterns such as:

- `contract_months = 1`
- relatively high `num_support_calls`
- short or moderate tenure

This suggests the random forest is learning a meaningful churn signal around short contracts and support friction, but sometimes reacts too strongly to it.

The tree-vs-linear disagreement analysis supports this interpretation. In one highlighted test example, the random forest assigned a much higher churn probability than logistic regression because it captured a more rule-like interaction pattern, especially around short contract length.

## Data Quality Caveat

The data quality report identified **278 suspicious rows** where:

- `tenure > 3`
- `total_charges == 0`

These cases may represent data entry problems, placeholder values, or edge cases. Because similar patterns appear in some false-positive examples, improving this field could reduce avoidable prediction errors and strengthen deployment reliability.

## Final Decision

**Deploy `RF_default` as the primary model, using threshold `0.20` as the recommended operating point.**

This is the best current choice because it combines the strongest ranking performance with a threshold selected using training-only evidence. However, rollout should be paired with a business review of intervention capacity, since the selected threshold produces a relatively high alert volume. If Petra Telecom needs fewer alerts, the next step should be to raise the operating threshold rather than replace the model.
