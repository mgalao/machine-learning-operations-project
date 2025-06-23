from fraud_project.utils import *

# --------------- scaling ---------------------
def scale_numerical(data: pd.DataFrame, numerical_features: list, scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    data = data.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    else:
        data[numerical_features] = scaler.transform(data[numerical_features])
    
    return data, scaler