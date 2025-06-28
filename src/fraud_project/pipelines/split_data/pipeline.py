
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  stratified_split_by_target


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= stratified_split_by_target,
                inputs= ["ingested_data", "params:target_column"],
                outputs=["ref_data","ana_data"],
                name="split_out_of_sample",
            ),
        ]
    )
