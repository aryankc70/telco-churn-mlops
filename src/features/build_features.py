"""Feature engineering: binary and one-hot encoding for Telco churn data."""
import pandas as pd

BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
MULTICLASS_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map 2-category columns to 0/1."""
    df = df.copy()
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df


def encode_multiclass_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode 3+ category columns, dropping first to avoid the dummy trap."""
    return pd.get_dummies(df, columns=MULTICLASS_COLS, drop_first=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = encode_binary_features(df)
    df = encode_multiclass_features(df)
    return df