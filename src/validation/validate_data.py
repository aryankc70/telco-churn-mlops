"""Data quality validation for incoming Telco churn data using Great Expectations."""
import great_expectations as gx
import pandas as pd


EXPECTED_BINARY_VALUES = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'PaperlessBilling': ['Yes', 'No'],
    'Churn': ['Yes', 'No'],
}

EXPECTED_MULTICLASS_VALUES = {
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
    'Bank transfer (automatic)', 'Credit card (automatic)'],
}


def validate_telco_data(df: pd.DataFrame) -> bool:
    """
    Validate raw Telco churn data against expected schema and value ranges.
    Raises ValueError if any check fails. Returns True if all checks pass.
    """
    errors = []

    # 1. Check all expected columns exist
    required_columns = (
        ['customerID', 'gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        + list(EXPECTED_BINARY_VALUES.keys())
        + list(EXPECTED_MULTICLASS_VALUES.keys())
    )
    required_columns = list(dict.fromkeys(required_columns))  # dedupe, preserve order

    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(errors))

    # 2. Check categorical columns only contain expected values
    for col, allowed_values in {**EXPECTED_BINARY_VALUES, **EXPECTED_MULTICLASS_VALUES}.items():
        actual_values = set(df[col].dropna().unique())
        unexpected = actual_values - set(allowed_values)
        if unexpected:
            errors.append(f"Column '{col}' has unexpected values: {unexpected}")

    # 3. Numerical bounds checks
    if df['tenure'].min() < 0:
        errors.append("tenure contains negative values")
    if df['tenure'].max() > 120:
        errors.append(f"tenure has implausible max value: {df['tenure'].max()}")

    if df['MonthlyCharges'].min() < 0:
        errors.append("MonthlyCharges contains negative values")

    if df['SeniorCitizen'].dropna().unique().tolist() not in ([0], [1], [0, 1], [1, 0]):
        errors.append(f"SeniorCitizen has unexpected values: {df['SeniorCitizen'].unique()}")

    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))

    return True