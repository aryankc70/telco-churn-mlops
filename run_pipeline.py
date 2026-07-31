"""
End-to-end training pipeline for the Telco churn model.
Run with: python run_pipeline.py
"""
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.validation.validate_data import validate_telco_data
from src.models.train import train_model

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Best hyperparameters found via Optuna tuning (see notebooks/eda.ipynb, Cell 22)
BEST_PARAMS = {
    "n_estimators": 313,
    "max_depth": 5,
    "learning_rate": 0.010165842770028277,
    "subsample": 0.8393524083955952,
    "colsample_bytree": 0.7415518994257162,
    "random_state": 42,
}

THRESHOLD = 0.2


def run_pipeline():
    print("Step 1/5: Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH)
    print(f"  Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    print("Step 2/5: Validating data quality...")
    validate_telco_data(df_raw)
    print("  Validation passed")

    print("Step 3/5: Preprocessing...")
    df_clean = preprocess_data(df_raw)
    print(f"  Cleaned shape: {df_clean.shape}")

    print("Step 4/5: Building features...")
    df_final = build_features(df_clean)
    print(f"  Final feature shape: {df_final.shape}")

    print("Step 5/5: Training model with MLflow tracking...")
    model = train_model(df_final, params=BEST_PARAMS, threshold=THRESHOLD)
    print("Pipeline complete.")

    return model


if __name__ == "__main__":
    run_pipeline()