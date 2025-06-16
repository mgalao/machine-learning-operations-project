import pandas as pd

def split_data_by_timestamp(df: pd.DataFrame, timestamp_col: str = "Timestamp"):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    reference = df[df[timestamp_col].dt.year == 2019].copy()
    validation = df[df[timestamp_col].dt.year > 2019].copy()
    return reference, validation
