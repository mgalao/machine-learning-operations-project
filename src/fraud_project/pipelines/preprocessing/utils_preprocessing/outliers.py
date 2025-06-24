from fraud_project.utils import *

# --------------- outlier treatment ---------------------

def outlier_treatment_amt(data: pd.DataFrame, quantile: float = 0.99) -> Tuple[pd.DataFrame, float]:
    data = data.copy()
    
    cap_val = data['amt'].quantile(quantile)
    data['amt'] = np.minimum(data['amt'], cap_val)
    
    data['log_amt'] = np.log1p(data['amt']).astype(float)

    return data, cap_val