"""Cleaning and preprocessing for the Telco churn dataset."""
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data: drop irrelevant columns, fix types, handle NAs."""
    df = df.copy()

    # Drop identifier column - not predictive
    df = df.drop(columns=['customerID'])

    # TotalCharges arrives as string with blank entries for tenure=0 customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    return df