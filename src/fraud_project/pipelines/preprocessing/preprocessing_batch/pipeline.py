
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocessing_batch

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= preprocessing_batch,
                inputs=["ana_data", "feature_engineering_params"],
                outputs= "preprocessed_batch_data",
                name="preprocessed_batch",
            ),
        ]
    )
