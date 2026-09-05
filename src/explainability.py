"""
WebLock - Explainability Layer (SHAP)
----------------------------------------
Turns raw fraud probability scores into human-readable explanations,
e.g. "Flagged mainly because: shared device with 6 other accounts (+0.31),
unusually high transaction amount (+0.12)".

This is the single most important piece for making the project usable
by an actual fraud analyst, and for demonstrating you understand that
"a model that works" and "a model people can trust and act on" are
different problems.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------
# Load model, feature list, and test data
# ---------------------------------------------------------------
model = joblib.load(os.path.join(BASE_DIR, "models", "graph_model.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "models", "graph_features.pkl"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "data", "X_test.csv"))[feature_cols]
y_test = pd.read_csv(os.path.join(BASE_DIR, "data", "y_test.csv"))["is_fraud"]

# ---------------------------------------------------------------
# Build SHAP explainer (TreeExplainer is fast + exact for LightGBM)
# ---------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values can be a list (per-class) or single array depending on version;
# normalize to "positive class" (fraud=1) contribution array
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# ---------------------------------------------------------------
# Global feature importance (mean |SHAP value|) -- which features
# matter most ACROSS the whole test set
# ---------------------------------------------------------------
mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print("=== Global feature importance (mean |SHAP value|) ===")
print(mean_abs_shap.head(10))

# ---------------------------------------------------------------
# Per-transaction explanation helper -- this is what the API/dashboard
# will call to generate the "why flagged" text
# ---------------------------------------------------------------
def explain_transaction(idx, top_k=3):
    row = X_test.iloc[idx]
    proba = model.predict_proba(X_test.iloc[[idx]])[0, 1]
    contributions = pd.Series(sv[idx], index=feature_cols).sort_values(key=abs, ascending=False)

    print(f"\nTransaction #{idx} | Fraud probability: {proba:.1%} | Actual label: {'FRAUD' if y_test.iloc[idx]==1 else 'normal'}")
    print("Top contributing factors:")
    for feat, val in contributions.head(top_k).items():
        direction = "increases" if val > 0 else "decreases"
        print(f"  - {feat} = {row[feat]:.2f}  ->  {direction} fraud score by {abs(val):.3f}")

# Show a few example explanations: a true fraud caught, and a normal txn
fraud_indices = y_test[y_test == 1].index[:2]
normal_indices = y_test[y_test == 0].index[:2]

print("\n" + "="*60)
print("EXAMPLE EXPLANATIONS")
print("="*60)
for i in list(fraud_indices) + list(normal_indices):
    pos = X_test.index.get_loc(i)
    explain_transaction(pos)

# ---------------------------------------------------------------
# Save SHAP explainer + values for reuse in the API/dashboard
# ---------------------------------------------------------------
joblib.dump(explainer, os.path.join(BASE_DIR, "models", "shap_explainer.pkl"))
print("\nSaved SHAP explainer to models/shap_explainer.pkl")
