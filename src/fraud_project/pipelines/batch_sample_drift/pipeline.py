from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_test_data_by_state, evaluate_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=filter_test_data_by_state,
            inputs="ana_data",
            outputs="ana_data_NY",
            name="filter_test_data",
        ),
        node(
            func=evaluate_predictions,
            inputs="df_with_predict",
            outputs="evaluation_metrics_df",
            name="evaluate_model_on_drifted_data",
        ),
    ])
