"""
WebLock - Fraud Scoring API
------------------------------
FastAPI service that takes transaction features and returns:
  - fraud probability
  - risk tier (low/medium/high)
  - top contributing factors (SHAP-based, human readable)

Run locally with:
    uvicorn api.main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json

Or visit http://localhost:8000/docs for interactive Swagger UI (auto-generated).
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

app = FastAPI(
    title="WebLock Fraud Scoring API",
    description="Real-time transaction fraud scoring with graph-based risk features and explainability.",
    version="1.0.0"
)

# ---------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "graph_model.pkl"))
explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "graph_features.pkl"))

MERCHANT_CATEGORIES = [
    "electronics", "gift_cards", "travel", "groceries", "apparel",
    "digital_goods", "jewelry", "crypto_exchange", "money_transfer", "gaming"
]


class TransactionRequest(BaseModel):
    amount: float = Field(..., example=245.50, description="Transaction amount")
    hour_of_day: int = Field(..., ge=0, le=23, example=2)
    day: int = Field(..., ge=1, le=31, example=15)
    is_weekend: int = Field(..., ge=0, le=1, example=0)
    merchant_category: str = Field(..., example="gift_cards")
    time_since_last_txn_sec: int = Field(..., example=300, description="Seconds since this account's last transaction")
    txn_count_last_24h: int = Field(..., example=12, description="Number of transactions by this account in last 24h")
    # graph features -- in a real system these would be computed live from the
    # shared-entity graph service; here the caller supplies them directly
    degree: int = Field(..., example=6, description="Number of other accounts sharing this account's device/IP/card")
    pagerank: float = Field(..., example=0.0012)
    clustering_coefficient: float = Field(..., example=0.4)
    component_size: int = Field(..., example=7, description="Size of the connected cluster this account belongs to")


class FraudPrediction(BaseModel):
    fraud_probability: float
    risk_tier: str
    top_factors: list[dict]


def build_feature_row(txn: TransactionRequest) -> pd.DataFrame:
    row = {c: 0 for c in feature_cols}
    row["amount"] = txn.amount
    row["hour_of_day"] = txn.hour_of_day
    row["day"] = txn.day
    row["is_weekend"] = txn.is_weekend
    row["time_since_last_txn_sec"] = txn.time_since_last_txn_sec
    row["txn_count_last_24h"] = txn.txn_count_last_24h
    row["degree"] = txn.degree
    row["pagerank"] = txn.pagerank
    row["clustering_coefficient"] = txn.clustering_coefficient
    row["component_size"] = txn.component_size

    cat_col = f"cat_{txn.merchant_category}"
    if cat_col in row:
        row[cat_col] = 1

    return pd.DataFrame([row])[feature_cols]


def risk_tier(proba: float) -> str:
    if proba >= 0.7:
        return "HIGH"
    elif proba >= 0.3:
        return "MEDIUM"
    return "LOW"


@app.get("/")
def root():
    return {"service": "WebLock Fraud Scoring API", "status": "running"}


@app.post("/predict", response_model=FraudPrediction)
def predict(txn: TransactionRequest):
    X = build_feature_row(txn)
    proba = float(model.predict_proba(X)[0, 1])

    shap_values = explainer.shap_values(X)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = pd.Series(sv, index=feature_cols).sort_values(key=abs, ascending=False)
    top_factors = [
        {
            "feature": feat,
            "value": float(X.iloc[0][feat]),
            "impact": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk"
        }
        for feat, val in contributions.head(3).items()
    ]

    return FraudPrediction(
        fraud_probability=round(proba, 4),
        risk_tier=risk_tier(proba),
        top_factors=top_factors
    )
