"""
Feature selection pipeline using multiple selection techniques
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    feature_selection,
    calculate_feature_correlations,
    calculate_feature_importance,
    plot_feature_importance,
    select_statistical_features,
    recursive_feature_elimination,
    filter_drifted_features
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_drifted_features,
                inputs=["X_train_data", "drifted_features"],
                outputs="X_train_stable",
                name="filter_drifted_features",
            ),
            node(
                func=calculate_feature_correlations,
                inputs="X_train_stable",
                outputs="feature_correlations",
                name="calculate_feature_correlations",
            ),
            node(
                func=calculate_feature_importance,
                inputs=["X_train_stable", "y_train_data"],
                outputs="feature_importance_scores",
                name="calculate_feature_importance",
            ),
            node(
                func=plot_feature_importance,
                inputs=["feature_importance_scores", "parameters"],
                outputs="feature_importance_plot",
                name="plot_feature_importance",
            ),
            node(
                func=select_statistical_features,
                inputs=["X_train_stable", "y_train_data", "parameters"],
                outputs="statistical_features",
                name="select_statistical_features",
            ),
            node(
                func=recursive_feature_elimination,
                inputs=["X_train_stable", "y_train_data", "parameters"],
                outputs="rfe_features",
                name="recursive_feature_elimination",
            ),
            node(
                func=feature_selection,
                inputs=[
                    "feature_correlations",
                    "feature_importance_scores",
                    "statistical_features",
                    "rfe_features",
                    "parameters"
                ],
                outputs="feature_selection_results",
                name="final_feature_selection",
            ),
        ]
    )
