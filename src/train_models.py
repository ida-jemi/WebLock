"""
WebLock - Model Training
---------------------------
Trains TWO models on the same data split so we can prove the value of
graph features with a clean, fair comparison:

  1. BASELINE  -> tabular features only (amount, time, category, velocity)
  2. GRAPH     -> baseline features + graph features (degree, pagerank,
                  component_size, clustering_coefficient)

Uses LightGBM (fast, handles imbalance well, industry-standard for
tabular fraud/risk models). Evaluated on PR-AUC (NOT accuracy -- with
5.6% fraud rate, a model that predicts "not fraud" for everything
would already be 94% "accurate" and completely useless).
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "transactions_with_graph_features.csv"))

# ---------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------
TABULAR_FEATURES = [
    "amount", "hour_of_day", "day", "is_weekend",
    "time_since_last_txn_sec", "txn_count_last_24h"
]
GRAPH_FEATURES = ["degree", "pagerank", "clustering_coefficient", "component_size"]

# one-hot encode merchant_category for both feature sets
df = pd.get_dummies(df, columns=["merchant_category"], prefix="cat")
cat_cols = [c for c in df.columns if c.startswith("cat_")]

TABULAR_FEATURES = TABULAR_FEATURES + cat_cols
GRAPH_ENHANCED_FEATURES = TABULAR_FEATURES + GRAPH_FEATURES

TARGET = "is_fraud"

# ---------------------------------------------------------------
# Train/test split (stratified to preserve fraud ratio in both sets)
# ---------------------------------------------------------------
X = df[GRAPH_ENHANCED_FEATURES]  # superset; we'll slice per model
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean():.3%}  Test fraud rate: {y_test.mean():.3%}")

# ---------------------------------------------------------------
# Helper to train + evaluate a LightGBM model on a given feature set
# ---------------------------------------------------------------
def train_and_eval(feature_cols, model_name):
    Xtr, Xte = X_train[feature_cols], X_test[feature_cols]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=15,
        class_weight="balanced",   # crucial for imbalanced fraud data
        random_state=42,
        verbosity=-1
    )
    model.fit(Xtr, y_train)

    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"ROC-AUC:                    {roc_auc:.4f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, target_names=["Normal", "Fraud"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return model, pr_auc, roc_auc

# ---------------------------------------------------------------
# Train both models
# ---------------------------------------------------------------
baseline_model, baseline_pr_auc, baseline_roc_auc = train_and_eval(
    TABULAR_FEATURES, "BASELINE MODEL (tabular features only)"
)

graph_model, graph_pr_auc, graph_roc_auc = train_and_eval(
    GRAPH_ENHANCED_FEATURES, "GRAPH-ENHANCED MODEL (tabular + graph features)"
)

# ---------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------
print(f"\n{'='*60}\nHEAD-TO-HEAD COMPARISON\n{'='*60}")
print(f"{'Metric':<15}{'Baseline':<15}{'Graph-Enhanced':<15}{'Lift':<10}")
print(f"{'PR-AUC':<15}{baseline_pr_auc:<15.4f}{graph_pr_auc:<15.4f}{(graph_pr_auc-baseline_pr_auc):+.4f}")
print(f"{'ROC-AUC':<15}{baseline_roc_auc:<15.4f}{graph_roc_auc:<15.4f}{(graph_roc_auc-baseline_roc_auc):+.4f}")

# ---------------------------------------------------------------
# Feature importance for the graph-enhanced model (sanity check that
# graph features actually matter, not just along for the ride)
# ---------------------------------------------------------------
importances = pd.Series(
    graph_model.feature_importances_, index=GRAPH_ENHANCED_FEATURES
).sort_values(ascending=False)
print("\nTop 10 most important features (graph-enhanced model):")
print(importances.head(10))

# ---------------------------------------------------------------
# Save models + test set for the next steps (evaluation/explainability/API)
# ---------------------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(baseline_model, os.path.join(BASE_DIR, "models", "baseline_model.pkl"))
joblib.dump(graph_model, os.path.join(BASE_DIR, "models", "graph_model.pkl"))
X_test.to_csv(os.path.join(BASE_DIR, "data", "X_test.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "data", "y_test.csv"), index=False)
joblib.dump(TABULAR_FEATURES, os.path.join(BASE_DIR, "models", "tabular_features.pkl"))
joblib.dump(GRAPH_ENHANCED_FEATURES, os.path.join(BASE_DIR, "models", "graph_features.pkl"))

print("\nSaved models to models/, test set to data/")
