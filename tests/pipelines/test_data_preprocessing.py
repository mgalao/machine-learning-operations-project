"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append("src")

from src.fraud_project.pipelines.preprocessing.preprocessing_train.nodes import feature_engineering_pipeline, clean_data
from fraud_project.pipelines.preprocessing.utils_preprocessing.encoding import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.cleaning import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.feature_engineering import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.missing_values import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.outliers import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.scaling import *

@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv("tests/pipelines/sample/sample.csv")


# --- CLEANING ---
def test_convert_zip_to_string(sample_df):
    df_out = convert_strings(sample_df.copy())
    assert df_out['zip'].dtype == object

def test_cap_min_age_applies_threshold(sample_df):
    df_out = cap_min_age(sample_df.copy(), min_age=16)
    assert df_out['age'].min() >= 16

def test_clean_data_fills_missing(sample_df):
    df_out, _ = clean_data(sample_df.copy())
    assert not df_out.isnull().any().any()


# --- TRANSFORMATIONS ---
def test_add_log_features(sample_df):
    df_transformed, log_cols = add_log_features(sample_df.copy())
    for col in log_cols:
        assert col in df_transformed.columns
        assert (df_transformed[col] >= 0).all()

def test_add_amt_per_km(sample_df):
    df = add_distance_feature(sample_df.copy())
    df = add_amt_per_km(df)
    assert "amt_per_km" in df.columns
    assert not df["amt_per_km"].isnull().any()

def test_outlier_treatment_amt(sample_df):
    treated_df, cap = outlier_treatment_amt(sample_df.copy(), quantile=0.90)
    assert treated_df["amt"].max() <= cap
    assert "log_amt" in treated_df.columns


# --- ENCODING ---
def test_encode_low_cardinality_output_shape(sample_df):
    df = sample_df.copy()
    df["is_weekend"] = np.random.randint(0, 2, len(df))
    features = {'categorical_features_low_cardinality': ['is_weekend'], 'categorical': ['is_weekend']}
    df_out, encoder, new_features = encode_low_cardinality(df, features)
    assert all(col.startswith('is_weekend') for col in new_features['categorical_features_low_cardinality'])

def test_rare_label_encoder(sample_df):
    df = rare_label_encoder(sample_df.copy(), "state", threshold=0.3)
    assert "state_grouped" in df.columns
    assert "Other" in df["state_grouped"].values


# --- TEMPORAL ENCODING ---
def test_encode_temporal(sample_df):
    df = sample_df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    df["day"] = pd.to_datetime(df["datetime"]).dt.day

    features = {"temporal": []}
    df_encoded, features_updated = encode_temporal(df, features)

    expected_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    for col in expected_cols:
        assert col in df_encoded.columns
    assert len(features_updated["temporal"]) == 8


# --- SCALING ---
def test_scale_numerical_order_preserved(sample_df):
    df = sample_df.copy()
    features = ["amt", "city_pop"]
    df_scaled, scaler = scale_numerical(df, features)
    df_scaled_2, _ = scale_numerical(df, features, scaler=scaler)
    assert df_scaled.columns.equals(df_scaled_2.columns)

def test_scaled_columns_are_zero_mean_unit_var(sample_df):
    df_cleaned, params = clean_data(sample_df.copy())
    df_feat, _ = feature_engineering_pipeline(df_cleaned, params)

    params["features"] = {"numerical": ["amt", "lat", "long", "city_pop", "age", "merch_lat", "merch_long"]}
    
    numerical = params["features"]["numerical"]
    means = df_feat[numerical].mean().abs()
    stds = df_feat[numerical].std().round()

    assert all(means < 1e-6), "Means not zero after scaling"
    assert all((stds == 1) | (stds == 0)), "Standard deviations not 1 or 0"


# --- COLUMNS / FEATURE TRACKING ---
def test_drop_columns_and_update_features(sample_df):
    df = sample_df.copy()
    features = {
        'numerical': ['amt', 'lat'],
        'categorical': ['merchant', 'state', 'job']
    }
    df_out, updated_features = drop_columns(df, features, features_to_drop=['merchant', 'job'])
    assert 'merchant' not in df_out.columns
    assert 'merchant' not in updated_features['categorical']
    assert 'job' not in updated_features['categorical']