
import pandas as pd
import os
import warnings
import json, math
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
import kagglehub
warnings.filterwarnings('ignore')


def download_dataset():
    """Download the COVID-19 dataset from Kaggle"""
    print("=== Downloading Dataset ===")
    path = kagglehub.dataset_download("abhilashjash/covid-19-simulated-dataset-by-abhilash-jash")
    print("Path to dataset files:", path)
    
    raw_data_path = os.path.join(path, "covid19_simulated_dataset_by_Abhilash_Jash.csv")
    return raw_data_path


def load_and_clean_data(raw_data_path):
    """Load and clean the raw dataset"""
    print("\n=== Loading RAW data ===")
    raw_data = pd.read_csv(raw_data_path)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Missing values: {raw_data.isnull().sum().sum()} total")
    
    print("\n=== Cleaning Data ===")
    
    # Drop rows with missing values
    data = raw_data.copy()
    data.dropna(inplace=True)
    print(f"After dropping NaN: {data.shape}")
    
    # Replace boolean values
    data = data.replace(False, 0)
    data = data.replace(True, 1)
    
    # Create gender dummy variables
    data["Is_Male"] = (data["Gender"] == "Male").astype(int)
    data["Is_Female"] = (data["Gender"] == "Female").astype(int)
    data["Gender_Other"] = (data["Gender"] == "Other").astype(int)
    
    # Drop unnecessary columns
    data = data.drop(["Patient_ID", "Gender", "Name"], axis=1)
    
    print(f"Final cleaned data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    return data


def save_cleaned_data(data):
    """Save cleaned data to project root and dashboard public folder"""
    print("\n=== Saving cleaned data ===")
    
    # Location 1: Project root
    root_input_path = "../../input.csv"
    data.to_csv(root_input_path, index=False)
    print(f"✓ Saved to: {root_input_path}")
    
    # Location 2: Dashboard public folder
    dashboard_input_path = "../public/input.csv"
    data.to_csv(dashboard_input_path, index=False)
    print(f"✓ Saved to: {dashboard_input_path}")
    
    print("\n=== Data cleaning complete! ===")
    print(f"Both input.csv files now contain {len(data)} cleaned records")
    
    return root_input_path


def generate_metadata(data):
    """Generate and save metadata statistics"""
    print("\n=== Cleaned Data Preview ===")
    print(data.head(2))
    
    print("\n=== Summary Statistics ===")
    print(data.describe())
    
    # Export metadata to both locations
    # Location 1: Project root
    root_metadata_path = '../../metadata.csv'
    data.describe().T.to_csv(root_metadata_path)
    print(f"\n✓ Metadata exported to: {root_metadata_path}")
    
    # Location 2: Dashboard public folder
    dashboard_metadata_path = '../public/metadata.csv'
    data.describe().T.to_csv(dashboard_metadata_path)
    print(f"✓ Metadata exported to: {dashboard_metadata_path}")
    
    print("\n=== Data Info ===")
    data.info()


def to_py(o):
    """Helper to make objects JSON-safe"""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o


def prepare_features(data, target_col):
    """Prepare features and target for training"""
    print(f"\n=== Preparing Features ===")
    print(f"Target column: {target_col}")
    
    y = data[target_col].astype(int).values
    X = data.drop(columns=[target_col])
    
    # Split categorical vs numeric
    categorical_cols, numeric_cols = [], []
    for c in X.columns:
        if X[c].dtype == "object":
            categorical_cols.append(c)
        elif pd.api.types.is_integer_dtype(X[c]):
            (categorical_cols if X[c].nunique() <= 10 else numeric_cols).append(c)
        else:
            numeric_cols.append(c)
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    return X, y, categorical_cols, numeric_cols


def create_preprocessor(categorical_cols, numeric_cols):
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    return preprocessor


def train_decision_tree(X, y, preprocessor):
    """Train decision tree with hyperparameter search"""
    print("\n=== Training Decision Tree ===")
    
    # Split train/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Decision Tree with hyperparameter search
    dt = DecisionTreeClassifier(random_state=42)
    
    pipe = Pipeline([
        ("pre", preprocessor),
        ("dt", dt)
    ])
    
    param_grid = {
        "dt__max_depth": [3, 4, 5, 6, None],
        "dt__min_samples_leaf": [1, 2, 5, 10]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="f1", n_jobs=-1
    )
    grid.fit(X_trainval, y_trainval)
    
    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    
    print(f"Best parameters: {best_params}")
    
    return best_pipe, best_params, X_trainval, X_test, y_trainval, y_test


def choose_threshold(best_pipe, X_trainval, y_trainval):
    """Choose optimal threshold favoring recall"""
    print("\n=== Choosing Threshold ===")
    
    val_proba = best_pipe.predict_proba(X_trainval)[:, 1]
    prec, rec, thr = precision_recall_curve(y_trainval, val_proba)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    mask = rec >= 0.85
    if mask.any():
        idx = np.nanargmax(f1s[mask])
        chosen_threshold = thr[max(0, np.where(mask)[0][idx] - 1)]
    else:
        chosen_threshold = thr[np.nanargmax(f1s)] if len(thr) > 0 else 0.5
    chosen_threshold = float(chosen_threshold)
    
    print(f"Chosen threshold: {chosen_threshold:.3f}")
    
    return chosen_threshold


def evaluate_model(best_pipe, X_test, y_test, chosen_threshold):
    """Evaluate model on test set"""
    print("\n=== Evaluating Model ===")
    
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= chosen_threshold).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba)),
        "precision": float(precision_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred))
    }
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    metrics["specificity"] = float(tn / (tn + fp + 1e-12))
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return metrics, test_proba, test_pred


def export_preprocessing_manifest(preprocessor, X_trainval, categorical_cols, numeric_cols):
    """Export preprocessing configuration"""
    print("\n=== Exporting Preprocessing Manifest ===")
    
    preprocessor.fit(X_trainval)
    
    # Handle numeric imputation values and ranges safely
    if numeric_cols:
        num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
        num_impute_values = {
            c: float(v)
            for c, v in zip(numeric_cols, num_imputer.statistics_.tolist())
        }
        num_ranges = {
            c: {"min": float(X_trainval[c].min()), "max": float(X_trainval[c].max())}
            for c in numeric_cols
        }
    else:
        num_impute_values = {}
        num_ranges = {}
    
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_vocabs = {}
    for c, cats in zip(categorical_cols, onehot.categories_):
        cat_vocabs[c] = [None if (isinstance(v, float) and math.isnan(v)) else v for v in cats.tolist()]
    
    final_feature_order = []
    for col in numeric_cols:
        final_feature_order.append({"source": col, "kind": "numeric"})
    for col, cats in cat_vocabs.items():
        for cat in cats:
            final_feature_order.append({"source": col, "kind": "onehot", "category": cat})
    
    preproc_manifest = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "numeric_imputation": num_impute_values,
        "numeric_ranges_train": num_ranges,
        "categorical_vocabulary": cat_vocabs,
        "final_feature_order": final_feature_order,
        "preproc_version": "v1",
        "trained_on": datetime.utcnow().strftime("%Y-%m-%d")
    }
    
    return preproc_manifest


def export_tree(tree_estimator):
    """Export decision tree structure"""
    tree = tree_estimator.tree_
    nodes = []
    for i in range(tree.node_count):
        nodes.append({
            "feature_index": int(tree.feature[i]),
            "threshold": float(tree.threshold[i]),
            "left": int(tree.children_left[i]),
            "right": int(tree.children_right[i]),
            "value": [float(x) for x in tree.value[i][0].tolist()],
            "is_leaf": bool(tree.children_left[i] == -1 and tree.children_right[i] == -1),
        })
    return {"nodes": nodes}


def export_model(best_pipe, best_params, chosen_threshold, metrics):
    """Export decision tree model"""
    print("\n=== Exporting Model ===")
    
    dt_est = best_pipe.named_steps["dt"]
    dt_export = {
        "params": best_params,
        "threshold": chosen_threshold,
        "metrics": metrics,
        "model_version": "v1",
        "trained_on": datetime.utcnow().strftime("%Y-%m-%d"),
        "tree": export_tree(dt_est)
    }
    
    return dt_export


def create_model_card(best_params, metrics):
    """Create model card documentation"""
    model_card = f"""# Heart Disease Decision Tree Model Card

**Model version:** v1  
**Training date:** {datetime.utcnow().strftime("%Y-%m-%d")}  

### Best Params
{best_params}

### Metrics (Test Set)
{json.dumps(metrics, indent=2)}

### Notes
Educational demo only — not medical advice.
"""
    return model_card


def create_golden_examples(X_test, y_test, test_proba, test_pred):
    """Create golden examples for testing"""
    golden = []
    for i in range(min(10, len(X_test))):
        row = X_test.iloc[i].to_dict()
        golden.append({
            "input": to_py(row),
            "pred_proba_high": float(test_proba[i]),
            "label": int(test_pred[i]),
            "true": int(y_test[i])
        })
    return golden


def save_artifacts(preproc_manifest, dt_export, model_card, golden):
    """Save all artifacts to dashboard public folder"""
    print("\n=== Saving Artifacts ===")
    
    # Save to dashboard public folder
    with open("../public/preproc_v1.json", "w") as f:
        json.dump(to_py(preproc_manifest), f, indent=2)
    with open("../public/dt_model_v1.json", "w") as f:
        json.dump(to_py(dt_export), f, indent=2)
    with open("../public/model_card_v1.md", "w") as f:
        f.write(model_card)
    with open("../public/golden_examples_v1.json", "w") as f:
        json.dump(to_py(golden), f, indent=2)
    
    print("✓ Artifacts written to ../public/:")
    print("  - preproc_v1.json")
    print("  - dt_model_v1.json")
    print("  - model_card_v1.md")
    print("  - golden_examples_v1.json")


def main():
    """Main pipeline"""
    target_col = "Is_Covid_True"

    # Download and prepare data
    raw_data_path = download_dataset()
    data = load_and_clean_data(raw_data_path)
    root_input_path = save_cleaned_data(data)
    generate_metadata(data)

    # Prepare features
    X, y, categorical_cols, numeric_cols = prepare_features(data, target_col)

    # Create preprocessor
    preprocessor = create_preprocessor(categorical_cols, numeric_cols)

    # Train model
    print("\n=== Training Decision Tree Model ===")
    best_pipe, best_params, X_trainval, X_test, y_trainval, y_test = train_decision_tree(
        X, y, preprocessor
    )

    # Choose threshold and evaluate
    print("\n=== Optimizing Threshold & Evaluating Performance ===")
    chosen_threshold = choose_threshold(best_pipe, X_trainval, y_trainval)
    metrics, test_proba, test_pred = evaluate_model(best_pipe, X_test, y_test, chosen_threshold)

    # Export preprocessing manifest
    print("\n=== Creating Preprocessing Configuration ===")
    preproc_manifest = export_preprocessing_manifest(
        preprocessor, X_trainval, categorical_cols, numeric_cols
    )

    # Export model
    print("\n=== Exporting Model Structure ===")
    dt_export = export_model(best_pipe, best_params, chosen_threshold, metrics)

    # Create documentation and examples
    print("\n=== Generating Documentation & Examples ===")
    model_card = create_model_card(best_params, metrics)
    golden = create_golden_examples(X_test, y_test, test_proba, test_pred)

    # Save all artifacts
    print("\n=== Saving All Artifacts ===")
    save_artifacts(preproc_manifest, dt_export, model_card, golden)

    print("\n=== Pipeline Complete! ===")


if __name__ == "__main__":
    main()