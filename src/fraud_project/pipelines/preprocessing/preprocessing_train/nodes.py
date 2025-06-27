"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# from fraud_project.utils import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.encoding import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.cleaning import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.feature_engineering import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.missing_values import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.outliers import *
from fraud_project.pipelines.preprocessing.utils_preprocessing.scaling import *



conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
params_from_yml = conf_loader.get("parameters", "parameters*")
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)



# --------------- log summary ---------------------

def log_feature_summary(features: dict, max_display: int = 5):
    for key, cols in features.items():
        n = len(cols)
        preview = ', '.join(cols[:max_display]) + ('...' if n > max_display else '')
        logger.info(f"{key.capitalize()} features ({n}): {preview}")

def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    data = convert_datetime(data)
    data = convert_strings(data)
    data = cap_min_age(data)
    data = clean_rest_data(data)
    data, zipcode_mappings = impute_merch_zipcode(data)
    params = {"zipcode_mappings": zipcode_mappings}
    return data, params


def feature_engineering_pipeline(data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    features = {
        "numerical": list(params_from_yml["numerical_features"]),
        "categorical": list(params_from_yml["categorical_features"]),
        "categorical_features_low_cardinality": list(params_from_yml["categorical_features_low_cardinality"]),
        "categorical_features_high_cardinality": list(params_from_yml["categorical_features_high_cardinality"]),
        "temporal": list(params_from_yml["temporal_features"]),
    }

    data, features = drop_columns(data, features)
    data, new_features = feature_engineering(data)
    data, features = drop_columns(data, features, features_to_drop=["datetime"])

    for key in features:
        features[key] = list(set(features[key] + new_features[key]))

    data, cap_value = outlier_treatment_amt(data)

    data, low_card_encoder, features = encode_low_cardinality(data, features)
    data, high_card_mappings, features = encode_high_cardinality(data, features)
    data, features = encode_temporal(data, features)

    binary_cols = new_features.get("binary", [])
    numerical_to_scale = [col for col in features["numerical"] if col not in binary_cols]
    data, scaler = scale_numerical(data, numerical_to_scale)

    params = {
        **params,
        "amt_cap_val": cap_value,
        "low_card_encoder": low_card_encoder,
        "high_card_mappings": high_card_mappings,
        "scaler": scaler,
        "features": features
    }

    logger.info(
        f"Preprocessing complete. Shape: {data.shape} | "
        f"Numerical: {len(features['numerical'])} | "
        f"Low-card categorical: {len(features['categorical_features_low_cardinality'])} | "
        f"High-card categorical: {len(features['categorical_features_high_cardinality'])} | "
        f"Temporal: {len(features['temporal'])}"
    )

    log_feature_summary(features)

    return data, params
