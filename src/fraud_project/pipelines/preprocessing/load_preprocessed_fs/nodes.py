"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from pathlib import Path
import pandas as pd
from fraud_project.pipelines.preprocessing.utils_preprocessing.feature_store import load_from_feature_store

def load_and_save_preprocessed_data(feature_store_groups: dict, output_path: str):
    dfs = []
    for key, fg in feature_store_groups.items():
        # Skip non-feature config entries
        if key == "upload_features" or not isinstance(fg, dict):
            continue

        df = load_from_feature_store(fg["name"], fg["version"])
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["trans_num", "datetime"], how="left", suffixes=("", "_dup"))

    # Drop duplicate columns
    duplicate_cols = [col for col in merged_df.columns if col.endswith("_dup")]
    merged_df.drop(columns=duplicate_cols, inplace=True)

    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)