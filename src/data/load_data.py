"""Load raw Telco churn data from CSV."""
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV into a DataFrame."""
    return pd.read_csv(file_path)