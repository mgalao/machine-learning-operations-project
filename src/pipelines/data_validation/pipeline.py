from kedro.pipeline import Pipeline, node
from .nodes import split_data_by_timestamp, generate_expectations_and_validate

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=split_data_by_timestamp,
            inputs="raw_data",
            outputs=["reference_data", "validation_data"],
            name="split_data_node"
        ),
        node(
            func=generate_expectations_and_validate,
            inputs=["reference_data", "validation_data", "params:gx_root_dir"],
            outputs=["expectation_suite", "validation_result"],
            name="validate_data_node"
        )
    ])
