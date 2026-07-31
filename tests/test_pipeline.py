"""Basic unit tests for the Telco churn pipeline components."""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.validation.validate_data import validate_telco_data

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_load_data_returns_dataframe():
    df = load_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 7043
    assert df.shape[1] == 21


def test_validate_data_passes_on_clean_data():
    df = load_data(RAW_DATA_PATH)
    assert validate_telco_data(df) is True


def test_validate_data_catches_bad_gender():
    df = load_data(RAW_DATA_PATH)
    df.loc[0, "gender"] = "Unknown"
    with pytest.raises(ValueError, match="unexpected values"):
        validate_telco_data(df)


def test_validate_data_catches_negative_tenure():
    df = load_data(RAW_DATA_PATH)
    df.loc[0, "tenure"] = -5
    with pytest.raises(ValueError, match="negative values"):
        validate_telco_data(df)


def test_preprocess_removes_customer_id():
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    assert "customerID" not in df_clean.columns


def test_preprocess_total_charges_is_numeric():
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
    assert df_clean["TotalCharges"].isnull().sum() == 0


def test_build_features_encodes_churn_as_binary():
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    df_final = build_features(df_clean)
    assert set(df_final["Churn"].unique()).issubset({0, 1})


def test_build_features_no_object_columns_remain():
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    df_final = build_features(df_clean)
    object_cols = df_final.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Unexpected non-numeric columns: {object_cols}"