# from kedro.pipeline import Pipeline, node, pipeline
# from .nodes import filter_test_data_by_state, evaluate_predictions

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#             func=filter_test_data_by_state,
#             inputs="ana_data",
#             outputs="ana_data_NY",
#             name="filter_test_data",
#         ),
#         node(
#             func=evaluate_predictions,
#             inputs="df_with_predict",
#             outputs="evaluation_metrics_df",
#             name="evaluate_model_on_drifted_data",
#         ),
#     ])

from kedro.pipeline import Pipeline, node, pipeline

from fraud_project.pipelines.preprocessing.preprocessing_batch.nodes import preprocessing_batch
from fraud_project.pipelines.model_predict.nodes import model_predict
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
            func=preprocessing_batch,
            inputs=["ana_data_NY", "feature_engineering_params"],
            outputs="preprocessed_batch_data",
            name="preprocess_filtered_data",
        ),
        node(
            func=model_predict,
            inputs=["preprocessed_batch_data", "production_model", "best_columns"],
            outputs=["df_with_predict", "predict_describe"],
            name="predict_filtered_data",
        ),
        node(
            func=evaluate_predictions,
            inputs="df_with_predict",
            outputs="evaluation_metrics_df",
            name="evaluate_model_on_filtered_data",
        ),
    ])