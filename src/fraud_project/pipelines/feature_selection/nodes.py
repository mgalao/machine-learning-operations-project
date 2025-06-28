import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List

def calculate_feature_correlations(X_train_data: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = X_train_data.corr().abs()
    return corr_matrix

def calculate_feature_importance(X_train_data: pd.DataFrame, y_train_data: pd.Series) -> pd.DataFrame:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_data, y_train_data)
    feature_importance = pd.DataFrame({
        'feature': X_train_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return feature_importance

def plot_feature_importance(importance_df: pd.DataFrame, parameters: Dict) -> plt.Figure:
    top_n = parameters.get("feature_selection_params", {}).get("top_n_features", 10)
    top_features = importance_df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(top_features["feature"], top_features["importance"], color="skyblue")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", padding=5)
    plt.tight_layout()
    return fig

def select_statistical_features(X_train_data: pd.DataFrame, y_train_data: pd.Series, 
                                params: Dict) -> List[str]:
    k = params.get("feature_selection_params", {}).get("k_best_features", 10)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train_data, y_train_data)
    cols = X_train_data.columns[selector.get_support()]
    return list(cols)

def recursive_feature_elimination(X_train_data: pd.DataFrame, y_train_data: pd.Series,
                                 params: Dict) -> List[str]:
    n_features = params.get("feature_selection_params", {}).get("n_features_to_select", 10)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train_data, y_train_data)
    selected_features = X_train_data.columns[rfe.support_]
    return list(selected_features)

def feature_selection(
    X_train_data: pd.DataFrame,
    y_train_data: pd.Series,
    drifted_features: List[str],
    parameters: Dict
) -> Dict:
    """
    Perform feature selection excluding drifted features.

    Args:
        X_train_data: Training features dataframe
        y_train_data: Target variable series
        drifted_features: List of features identified as drifted (to exclude)
        parameters: Configuration dict with feature selection params

    Returns:
        Dict with selected features and intermediate results
    """

    # Filter out drifted features from X_train_data
    stable_features = [feat for feat in X_train_data.columns if feat not in drifted_features]
    X_filtered = X_train_data[stable_features]

    # Calculate correlations and feature importance on stable features only
    feature_correlations = calculate_feature_correlations(X_filtered)
    feature_importance_scores = calculate_feature_importance(X_filtered, y_train_data)
    statistical_features = select_statistical_features(X_filtered, y_train_data, parameters)
    rfe_features = recursive_feature_elimination(X_filtered, y_train_data, parameters)

    # Combine features based on the combination method
    method = parameters.get("feature_selection_params", {}).get("combination_method", "union")

    if method == "intersection":
        final_features = list(set(statistical_features).intersection(set(rfe_features)))
    elif method == "weighted":
        top_n = parameters.get("feature_selection_params", {}).get("top_n_features", 10)
        importance_features = list(feature_importance_scores['feature'].head(top_n))
        final_features = list(set(importance_features + statistical_features + rfe_features))
    else:  # default union
        final_features = list(set(statistical_features).union(set(rfe_features)))

    return {
        "best_columns": final_features,
        "feature_importance": feature_importance_scores,
        "correlation_matrix": feature_correlations,
        "statistical_features": statistical_features,
        "rfe_features": rfe_features
    }
