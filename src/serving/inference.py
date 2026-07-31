"""Inference logic: load model, encode incoming data, predict."""
import os
import json
import joblib
import pandas as pd

from src.features.build_features import encode_binary_features, encode_multiclass_features

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "model.joblib")
_COLUMNS_PATH = os.path.join(_PROJECT_ROOT, "models", "feature_columns.json")

THRESHOLD = 0.2

_model = None
_feature_columns = None


def load_artifacts():
    """Load model and feature columns into memory. Called once at API startup."""
    global _model, _feature_columns
    _model = joblib.load(_MODEL_PATH)
    with open(_COLUMNS_PATH) as f:
        _feature_columns = json.load(f)
    return _model, _feature_columns


def predict_churn(customer_dict: dict) -> dict:
    """
    Run a single customer record through preprocessing, encoding, and the model.

    Args:
        customer_dict: Raw customer fields matching the original CSV schema
                        (minus customerID and Churn).

    Returns:
        dict with churn_prediction, churn_probability, threshold_used.
    """
    if _model is None or _feature_columns is None:
        load_artifacts()

    df = pd.DataFrame([customer_dict])

    # Note: no customerID/Churn present in inference requests, unlike training data,
    # so we skip preprocess_data() (which drops customerID) and encode directly.
    df = encode_binary_features_safe(df)
    df = encode_multiclass_features(df)

    # Align columns to match training exactly - missing dummy columns get filled with 0,
    # extra/unexpected columns get dropped
    df = df.reindex(columns=_feature_columns, fill_value=0)

    proba = _model.predict_proba(df)[:, 1][0]
    prediction = "Likely to churn" if proba >= THRESHOLD else "Not likely to churn"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba), 4),
        "threshold_used": THRESHOLD,
    }


def encode_binary_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as encode_binary_features, but skips 'Churn' since inference
    requests never include the target column.
    """
    df = df.copy()
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df