from fraud_project.utils import *

# --------------- initial cleaning and incoherences treatment ---------------------

def convert_datetime(data: pd.DataFrame) -> pd.DataFrame:
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    return data

def convert_strings(data: pd.DataFrame) -> pd.DataFrame:
    data['zip'] = data['zip'].astype(str)
    return data

def cap_min_age(data: pd.DataFrame, min_age: int = 16) -> pd.DataFrame:
    data = data.copy()

    data.loc[data['age'] < min_age, 'age'] = min_age

    return data

def drop_columns(data: pd.DataFrame, features: Dict[str, List[str]], features_to_drop: List=['merchant', 'first']) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    data = data.copy()

    data = data.drop(columns=features_to_drop, errors='ignore')
    for group in features:
        features[group] = [col for col in features[group] if col not in features_to_drop]

    return data, features


# --------------- cleaning_data ---------------------
def clean_rest_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.fillna(-9999, inplace=True)
    
    return data