# Heart Disease Decision Tree Model Card

**Model version:** v1  
**Training date:** 2025-10-22  

### Best Params
{'dt__max_depth': 10, 'dt__min_samples_leaf': 100, 'dt__min_samples_split': 200, 'dt__min_impurity_decrease': 0.0001, 'dt__class_weight': 'balanced'}

### Metrics (Test Set)
{
  "roc_auc": 0.6983439234974984,
  "pr_auc": 0.8315550664357139,
  "precision": 0.746,
  "recall": 1.0,
  "f1": 0.854524627720504,
  "specificity": 0.0,
  "confusion_matrix": {
    "tn": 0,
    "fp": 254,
    "fn": 0,
    "tp": 746
  }
}

### Notes
Educational demo only — not medical advice.
