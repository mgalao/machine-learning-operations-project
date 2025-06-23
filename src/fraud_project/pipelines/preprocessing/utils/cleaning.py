from fraud_project.utils import *

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


# --------------- cleaning_data ---------------------
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.fillna(-9999, inplace=True)
    
    return data