
from fraud_project.utils import *

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