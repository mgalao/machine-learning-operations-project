"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=[
                    "X_train_data",
                    "X_test_data",
                    "y_train_data",
                    "y_test_data",
                    "parameters",
                    "best_columns"
                ],
                outputs=[
                    "model_champion",      
                    "production_columns",
                    "production_model_metrics",
                    "output_plot",
                    "is_new_champion"
                ],
                name="model_training",
            ),
        ]
    )
