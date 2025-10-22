# Heart Disease Decision Tree Model Card

**Model version:** v1  
**Training date:** 2025-10-22  

### Best Params
{'dt__max_depth': 10, 'dt__min_samples_leaf': 100, 'dt__min_samples_split': 200, 'dt__min_impurity_decrease': 0.0001, 'dt__class_weight': 'balanced'}

### Metrics (Test Set)
{
  "roc_auc": 0.7032321205104476,
  "pr_auc": 0.833267652907268,
  "precision": 0.7451029759806637,
  "recall": 1.0,
  "f1": 0.8539358264081256,
  "specificity": 0.0,
  "confusion_matrix": {
    "tn": 0,
    "fp": 35434,
    "fn": 0,
    "tp": 103579
  }
}

### Notes
Educational demo only — not medical advice.
