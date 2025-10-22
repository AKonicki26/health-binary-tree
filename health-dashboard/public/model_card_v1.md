# Heart Disease Decision Tree Model Card

**Model version:** v1  
**Training date:** 2025-10-22  

### Best Params
{'dt__max_depth': 10, 'dt__min_samples_leaf': 100, 'dt__min_samples_split': 200, 'dt__min_impurity_decrease': 0.0001, 'dt__class_weight': 'balanced'}

### Metrics (Test Set)
{
  "roc_auc": 0.6996258195354896,
  "pr_auc": 0.8315825062838861,
  "precision": 0.7452252667016754,
  "recall": 1.0,
  "f1": 0.8540161329546719,
  "specificity": 0.0,
  "confusion_matrix": {
    "tn": 0,
    "fp": 35417,
    "fn": 0,
    "tp": 103596
  }
}

### Notes
Educational demo only — not medical advice.
