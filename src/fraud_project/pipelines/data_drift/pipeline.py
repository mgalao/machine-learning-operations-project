
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_drift

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= data_drift,
                inputs=["ref_data","ana_data"],
                outputs= "drift_result",
                name="data_drift",
            ),

        ]
    )
