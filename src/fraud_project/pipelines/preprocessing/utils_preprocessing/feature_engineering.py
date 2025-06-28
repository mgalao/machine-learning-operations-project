from fraud_project.utils import *

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

    data['hour'] = data['datetime'].dt.hour
    data['day'] = data['datetime'].dt.day
    data['weekday'] = data['datetime'].dt.weekday
    data['month'] = data['datetime'].dt.month

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

def add_flag_invalid_zip(data: pd.DataFrame) -> pd.DataFrame:
    ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")
    data = data.copy()
    data['invalid_zip'] = (~data['zip'].astype(str).apply(lambda x: bool(ZIP_REGEX.match(x)))).astype(float)
    return data

# def add_flag_invalid_cc_num(data: pd.DataFrame) -> pd.DataFrame:
#     CC_NUM_REGEX = re.compile(r"^\d{13,19}$")
#     data = data.copy()
#     data['invalid_cc_num'] = (~data['cc_num'].astype(str).apply(lambda x: bool(CC_NUM_REGEX.match(x)))).astype(float)
#     return data

def feature_engineering(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    data = data.copy()

    new_features = {
        "numerical": [],
        "categorical": [],
        "temporal": [],
        "categorical_features_high_cardinality": [], 
        "categorical_features_low_cardinality": [],
        "binary": [],
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

    # Invalid zip code flag
    data = add_flag_invalid_zip(data)
    new_features['categorical'].append('invalid_zip')
    new_features['categorical_features_low_cardinality'].append('invalid_zip')

    return data, new_features