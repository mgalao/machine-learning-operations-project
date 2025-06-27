"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=[
                    "X_train_data",
                    "X_test_data",
                    "y_train_data",
                    "y_test_data",
                    "parameters",
                    "parameters_optuna"
                ],
                outputs=[
                    "best_model_class",
                    "best_model_params",
                    "best_model_score"
                ],
                name="model_selection",
            ),
        ]
    )