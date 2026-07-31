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

    return model