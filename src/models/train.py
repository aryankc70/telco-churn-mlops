"""Model training with MLflow experiment tracking."""
import os
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
import joblib
import json


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_model(df: pd.DataFrame, params: dict = None, threshold: float = 0.3,
                 experiment_name: str = "telco-churn"):
    """
    Train an XGBoost classifier on the Telco churn dataset with MLflow tracking.

    Args:
        df: Feature-engineered dataframe with 'Churn' as target column.
        params: XGBoost hyperparameters. Uses sensible defaults if None.
        threshold: Classification threshold (lower = higher recall, lower precision).
        experiment_name: MLflow experiment name.

    Returns:
        Trained XGBClassifier model.
    """
    mlflow.set_tracking_uri(f"sqlite:///{_PROJECT_ROOT}/mlflow.db")

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
        }

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.xgboost.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred))

    return model, X.columns.tolist()


def tune_model(df: pd.DataFrame, n_trials: int = 30, threshold: float = 0.3):
    """
    Use Optuna to find XGBoost hyperparameters that maximize recall.

    Args:
        df: Feature-engineered dataframe with 'Churn' as target column.
        n_trials: Number of Optuna trials to run.
        threshold: Classification threshold used during evaluation.

    Returns:
        dict of best hyperparameters found.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        return recall_score(y_test, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best recall: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params

def save_model(model, feature_columns, model_dir: str = None):
    """
    Persist the trained model and the exact feature column order to disk.

    Args:
        model: Trained XGBClassifier.
        feature_columns: List of column names in the order the model expects.
        model_dir: Directory to save into. Defaults to <project_root>/models.
    """
    if model_dir is None:
        model_dir = os.path.join(_PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    columns_path = os.path.join(model_dir, "feature_columns.json")

    joblib.dump(model, model_path)
    with open(columns_path, "w") as f:
        json.dump(list(feature_columns), f)

    print(f"Model saved to {model_path}")
    print(f"Feature columns saved to {columns_path}")