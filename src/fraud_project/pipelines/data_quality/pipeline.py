
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import manual_tests, industry_profiling

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=manual_tests,
                inputs=["ana_data", "parameters"],
                outputs=None,
                name="manual_tests_node",
            ),
            node(
                func=industry_profiling,
                inputs="ana_data",
                outputs=None,
                name="industry_profiling_node",
            ),
        ]
    )