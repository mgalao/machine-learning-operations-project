"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineering_pipeline, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= clean_data,
                inputs="ref_data",
                outputs="ref_data_cleaned",
                name="clean_data",
            ),
            node(
                func= feature_engineering_pipeline,
                inputs="ref_data_cleaned",
                outputs=["preprocessed_training_data","params"],
                name="preprocessed_training",
            ),
        ]
    )
