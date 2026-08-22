#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import shap

# Initialize FastAPI app
app = FastAPI(
    title="Glass-Box Bank Fraud Detection API",
    description="Infers account fraud probability using tuned LightGBM and explains decisions via TreeSHAP.",
    version="1.0.0"
)

# Load the champion model and preprocessors (adjust path as needed)
# model = joblib.load("tuned_lightgbm_fraud_model.pkl")
# explainer = shap.TreeExplainer(model)

# Define optimal classification threshold derived from business cost analysis
OPTIMAL_THRESHOLD = 0.3763

# Define input schema matching your engineered feature set
class TransactionInput(BaseModel):
    prev_address_months_count: int = Field(..., description="Months at previous address")
    current_address_months_count: int = Field(..., description="Months at current address")
    bank_months_count: int = Field(..., description="Months with the bank")
    income_x_credit_score: float = Field(..., description="Interaction term of income and credit risk score")
    velocity_6h: float = Field(..., description="Transaction velocity over the last 6 hours")
    velocity_24h: float = Field(..., description="Transaction velocity over the last 24 hours")
    velocity_4w: float = Field(..., description="Transaction velocity over the last 4 weeks")
    customer_age: int = Field(..., description="Age of the customer")
    intended_balcon_amount: float = Field(..., description="Intended balanced amount")
    session_length_in_minutes: float = Field(..., description="Length of the active session")

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud_predicted: bool
    threshold_used: float
    top_contributing_factors: dict

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput):
    try:
        # Convert input payload to DataFrame for model inference
        input_data = pd.DataFrame([transaction.dict()])

        # 1. Predict Fraud Probability using tuned LightGBM
        # proba = model.predict_proba(input_data)[:, 1][0]

        # Mocking inference output for demonstration structure based on your capstone metrics:
        proba = 0.4120  # Example score exceeding the 0.3763 threshold

        is_fraud = bool(proba >= OPTIMAL_THRESHOLD)

        # 2. Local Glass-Box Interpretation using TreeSHAP
        # shap_values = explainer.shap_values(input_data)
        # Extract top feature drivers for this specific inference call

        top_factors = {
            "prev_address_months_count": -0.65,
            "income_x_credit_score": +0.42,
            "velocity_24h": +0.18
        }

        return {
            "fraud_probability": round(float(proba), 4),
            "is_fraud_predicted": is_fraud,
            "threshold_used": OPTIMAL_THRESHOLD,
            "top_contributing_factors": top_factors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "LightGBM Tuned", "auc": 0.8325}

