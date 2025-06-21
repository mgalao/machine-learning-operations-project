
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import profile_and_validate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= profile_and_validate_data,
                inputs="ingested_data",
                outputs= "profiling_report",
                name="data_profiling",
            ),

        ]
    )
