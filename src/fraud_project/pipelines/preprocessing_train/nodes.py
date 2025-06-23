"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
from typing import List, Tuple

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings


conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
params_from_yml = conf_loader.get("parameters", "parameters*")
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


# --------------- initial cleaning and incoherences tretament ---------------------

def convert_datetime(data: pd.DataFrame) -> pd.DataFrame:
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], errors='coerce')
    data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
    return data

def convert_strings(data: pd.DataFrame) -> pd.DataFrame:
    data['zip'] = data['zip'].astype(str)
    return data

def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['day'] = data['trans_date_trans_time'].dt.day
    data['weekday'] = data['trans_date_trans_time'].dt.weekday
    data['month'] = data['trans_date_trans_time'].dt.month
    return data

# def calculate_age(data: pd.DataFrame) -> pd.DataFrame:
#     data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year
#     return data

def cap_min_age(data: pd.DataFrame, min_age: int = 16) -> pd.DataFrame:
    data = data.copy()

    data.loc[data['age'] < min_age, 'age'] = min_age

    return data

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.drop(columns=['merchant', 'first'], inplace=True)

    return data

def drop_columns(data: pd.DataFrame, features: Dict[str, List[str]], features_to_drop: List=['trans_date_trans_time', 'dob', 'merchant', 'first']) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    data = data.copy()

    data = data.drop(columns=features_to_drop, errors='ignore')
    for group in features:
        features[group] = [col for col in features[group] if col not in features_to_drop]

    return data, features




# --------------- missing values tretament ---------------------

def impute_merch_zipcode(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # impute using most frequent zip per city
    city_zip_lookup = (
        data[~data['merch_zipcode'].isna()]
        .groupby('city')['merch_zipcode']
        .agg(lambda x: x.mode().iloc[0])
        .to_dict())
    mask_city = data['merch_zipcode'].isna() & data['city'].notna()
    data.loc[mask_city, 'merch_zipcode'] = data.loc[mask_city, 'city'].map(city_zip_lookup)

    # impute remaining using most frequent zip per state
    state_zip_lookup = (
        data[~data['merch_zipcode'].isna()]
        .groupby('state')['merch_zipcode']
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    mask_state = data['merch_zipcode'].isna() & data['state'].notna()
    data.loc[mask_state, 'merch_zipcode'] = data.loc[mask_state, 'state'].map(state_zip_lookup)

    # final flag for rows still missing after imputations
    data['zip_missing'] = data['merch_zipcode'].isna().astype(int)

    return data


# --------------- feature engineering ---------------------

# distance
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def add_distance_feature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["distance"] = haversine_vectorized(
        data["lat"], data["long"], data["merch_lat"], data["merch_long"])
    
    return data

def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['day'] = data['trans_date_trans_time'].dt.day
    data['weekday'] = data['trans_date_trans_time'].dt.weekday
    data['month'] = data['trans_date_trans_time'].dt.month

    return data

def add_is_weekend_feature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)

    return data

def add_log_features(data: pd.DataFrame, log_features: List[str] = ['amt', 'city_pop']) -> Tuple[pd.DataFrame, List[str]]:
    new_features = []
    
    for col in log_features:
        new_col = f'log_{col}'
        data[new_col] = np.log1p(data[col])
        new_features.append(new_col)
    
    return data, new_features

def add_amt_per_km(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data['amt_per_km'] = data['amt'] / (data['distance'] + 0.01)  # to avoid division by zero

    return data

def rare_label_encoder(data: pd.DataFrame, col: str, threshold: float = 0.01) -> pd.DataFrame:
    data = data.copy()
    
    freq = data[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    
    new_col = f"{col}_grouped"
    data[new_col] = data[col].apply(lambda x: 'Other' if x in rare else x)
    
    return data


def feature_engineering(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    data = data.copy()

    new_features = {
        "numerical": [],
        "categorical": [],
        "temporal": [],
        "categorical_features_high_cardinality": [], 
        "categorical_features_low_cardinality": [], 
    }

    # Distance
    data = add_distance_feature(data)
    new_features["numerical"].append("distance")

    # Temporal features
    data = add_temporal_features(data)
    new_features["temporal"] += ['hour', 'day', 'weekday', 'month']

    # Is weekend
    data = add_is_weekend_feature(data)
    new_features["categorical"].append("is_weekend")
    new_features["categorical_features_low_cardinality"].append("is_weekend")

    # Log features
    data, log_feats = add_log_features(data)
    new_features["numerical"] += log_feats

    # Amount per km
    data = add_amt_per_km(data)
    new_features["numerical"].append("amt_per_km")

    # Rare label encoding
    data = rare_label_encoder(data, 'state')
    new_features['categorical'].append('state_grouped')
    new_features['categorical_features_high_cardinality'].append('state_grouped')

    return data, new_features



# --------------- outlier treatment ---------------------

def outlier_treatment_amt(data: pd.DataFrame, quantile: float = 0.99) -> Tuple[pd.DataFrame, float]:
    data = data.copy()
    
    cap_val = data['amt'].quantile(quantile)
    data['amt'] = np.minimum(data['amt'], cap_val)
    
    data['log_amt'] = np.log1p(data['amt'])

    return data, cap_val



# --------------- encoding ---------------------

def encode_low_cardinality(data: pd.DataFrame, features: dict, encoder: OneHotEncoder = None) -> Tuple[pd.DataFrame, OneHotEncoder, dict]:
    low_card_cols = features["categorical_features_low_cardinality"]

    if encoder is None:
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
        encoded_array = encoder.fit_transform(data[low_card_cols])
    else:
        encoded_array = encoder.transform(data[low_card_cols])

    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(low_card_cols), index=data.index)
    
    data = pd.concat([data.drop(columns=low_card_cols), encoded_df], axis=1)

    # Update features dictionary
    features["categorical_features_low_cardinality"] = list(encoded_df.columns)
    features["categorical"] = [f for f in features["categorical"] if f not in low_card_cols] + list(encoded_df.columns)

    return data, encoder, features


def encode_high_cardinality(data: pd.DataFrame, features: dict, mappings: dict = None) -> Tuple[pd.DataFrame, dict, dict]:
    high_card_cols = features["categorical_features_high_cardinality"]
    features["categorical_features_high_cardinality"] = []
    new_mappings = {}

    for col in high_card_cols:
        if mappings is None:
            # Compute frequency mapping during training
            freq_map = data[col].value_counts(normalize=True).to_dict()
            new_mappings[col] = freq_map
        else:
            # Use precomputed mapping for inference
            freq_map = mappings.get(col, {})

        data[col + "_freq"] = data[col].map(freq_map).fillna(0)
        features["categorical_features_high_cardinality"].append(col + "_freq")

    # Drop original high cardinality columns
    data.drop(columns=high_card_cols, inplace=True)
    features["categorical"] = [f for f in features["categorical"] if f not in high_card_cols] + features["categorical_features_high_cardinality"]

    return data, new_mappings if mappings is None else mappings, features


def encode_temporal(data: pd.DataFrame, features: dict) -> Tuple[pd.DataFrame, dict]:
    data = data.copy()
    encoded_cols = []

    cyclical_mappings = {
        'hour': 24,
        'weekday': 7,
        'month': 12,
        'day': 31}

    for col, period in cyclical_mappings.items():
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"

        data[sin_col] = np.sin(2 * np.pi * data[col] / period)
        data[cos_col] = np.cos(2 * np.pi * data[col] / period)

        encoded_cols.extend([sin_col, cos_col])

    features["temporal"] = encoded_cols

    # Drop original columns from the dataframe
    data.drop(columns=list(cyclical_mappings.keys()), inplace=True)

    return data, features


# --------------- scaling ---------------------

def scale_numerical(data: pd.DataFrame, numerical_features: list, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    data = data.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    else:
        data[numerical_features] = scaler.transform(data[numerical_features])
    
    return data, scaler


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.fillna(-9999, inplace=True)
    
    return data


def preprocessing_train(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    data = convert_datetime(data)
    data = convert_strings(data)
    data = create_time_features(data)
    data = cap_min_age(data)
    data = impute_merch_zipcode(data)

    features = {
        "numerical": list(params_from_yml["numerical_features"]),
        "categorical": list(params_from_yml["categorical_features"]),
        "categorical_features_low_cardinality": list(params_from_yml["categorical_features_low_cardinality"]),
        "categorical_features_high_cardinality": list(params_from_yml["categorical_features_high_cardinality"]),
        "temporal": list(params_from_yml["temporal_features"]),
    }

    data, features = drop_columns(data, features)
    data, new_features = feature_engineering(data)

    for key in features:
        features[key] = list(set(features[key] + new_features[key]))

    data, cap_value = outlier_treatment_amt(data)

    data, low_card_encoder, features = encode_low_cardinality(data, features)
    data, high_card_mappings, features = encode_high_cardinality(data, features)
    data, features = encode_temporal(data, features)

    data, scaler = scale_numerical(data, features["numerical"])
    data = clean_data(data)

    params = {
        "amt_cap_val": cap_value,
        "low_card_encoder": low_card_encoder,
        "high_card_mappings": high_card_mappings,
        "scaler": scaler
    }

    logger.info(
        f"Preprocessing complete. Shape: {data.shape} | "
        f"Numerical: {len(features['numerical'])} | "
        f"Low-card categorical: {len(features['categorical_features_low_cardinality'])} | "
        f"High-card categorical: {len(features['categorical_features_high_cardinality'])} | "
        f"Temporal: {len(features['temporal'])}"
    )

    for key, cols in features.items():
        logger.info(f"{key}: {cols}")

    return data, params
