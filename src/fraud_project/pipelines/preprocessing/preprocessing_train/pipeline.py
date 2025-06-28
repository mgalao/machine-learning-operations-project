"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_and_merge_feature_groups,
    clean_data,
    feature_engineering_pipeline,
    upload_preprocessed_features,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_merge_feature_groups,
                inputs=["params:feature_store_groups"],
                outputs="raw_data",
                name="load_and_merge_feature_groups",
            ),
            node(
                func=clean_data,
                inputs="raw_data",
                outputs=["ref_data_cleaned", "cleaning_params"],
                name="clean_data",
            ),
            node(
                func=feature_engineering_pipeline,
                inputs=["ref_data_cleaned", "cleaning_params"],
                outputs=["preprocessed_training_data", "feature_engineering_params"],
                name="feature_engineering",
            ),
            node(
                func=upload_preprocessed_features,
                inputs=["preprocessed_training_data", "params:feature_store_preprocessed_groups"],
                outputs=None,
                name="upload_preprocessed_features"
            ),
        ]
    )
