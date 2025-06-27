"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
from fraud_project.utils import *

from fraud_project.pipelines.preprocessing.utils_preprocessing.encoding import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.cleaning import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.feature_engineering import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.missing_values import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.outliers import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.scaling import *


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings


conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
params_from_yml = conf_loader.get("parameters", "parameters*")
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)

def preprocessing_batch(data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:

    data = convert_datetime(data)
    data = convert_strings(data)
    data = cap_min_age(data)
    data = clean_rest_data(data)
    data = impute_merch_zipcode(data, mappings=params["zipcode_mappings"])

    features = {
        "numerical": list(params_from_yml["numerical_features"]),
        "categorical": list(params_from_yml["categorical_features"]),
        "categorical_features_low_cardinality": list(params_from_yml["categorical_features_low_cardinality"]),
        "categorical_features_high_cardinality": list(params_from_yml["categorical_features_high_cardinality"]),
        "temporal": list(params_from_yml["temporal_features"]),
    }

    data, features = drop_columns(data, features)

    new_features = {
        "numerical": [],
        "categorical": [],
        "temporal": [],
        "categorical_features_high_cardinality": [], 
        "categorical_features_low_cardinality": [], 
    }

    # feature engineering
    data = add_distance_feature(data)
    new_features["numerical"].append("distance")
    
    data = add_temporal_features(data)
    new_features["temporal"] += ['hour', 'day', 'weekday', 'month']

    data = add_is_weekend_feature(data)
    new_features["categorical"].append("is_weekend")
    new_features["categorical_features_low_cardinality"].append("is_weekend")
    
    data, log_feats = add_log_features(data)
    new_features["numerical"] += log_feats

    data = add_amt_per_km(data)
    new_features["numerical"].append("amt_per_km")

    data = rare_label_encoder(data, 'state') 
    new_features['categorical'].append('state_grouped')
    new_features['categorical_features_high_cardinality'].append('state_grouped')

    # Invalid zip code flag
    data = add_flag_invalid_zip(data)
    new_features['categorical'].append('invalid_zip')
    new_features['categorical_features_low_cardinality'].append('invalid_zip')

    # Invalid credit card number flag
    data = add_flag_invalid_cc_num(data)
    new_features['categorical'].append('invalid_cc_num')
    new_features['categorical_features_low_cardinality'].append('invalid_cc_num')

    # Cap amount using training value
    data['amt'] = np.minimum(data['amt'], params["amt_cap_val"])
    data['log_amt'] = np.log1p(data['amt']).astype(float)

    for key in features:
        features[key] = list(set(features[key] + new_features[key]))
    
    data, features = drop_columns(data, features, features_to_drop=["datetime"])

    # Low-cardinality encoding using fitted encoder
    data, _, features = encode_low_cardinality(data, features, encoder=params["low_card_encoder"])

    # High-cardinality encoding using saved mappings
    data, _, features = encode_high_cardinality(data, features, mappings=params["high_card_mappings"])

    # Temporal encoding
    data, features = encode_temporal(data, features)

    # Scaling
    data, _ = scale_numerical(data, features["numerical"], scaler=params["scaler"])

    logger.info(f"Batch preprocessing complete. Final shape: {data.shape}")
    
    return data



