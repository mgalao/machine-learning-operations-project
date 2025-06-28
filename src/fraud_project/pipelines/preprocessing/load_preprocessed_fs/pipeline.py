
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_random


from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_and_save_preprocessed_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_save_preprocessed_data,
                inputs={
                    "feature_store_groups": "params:feature_store_preprocessed_groups",
                    "output_path": "params:preprocessed_output_path"
                },
                outputs=None,
                name="load_and_save_preprocessed_data_node",
            )
        ]
    )
