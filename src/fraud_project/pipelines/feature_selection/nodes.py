import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List

def calculate_feature_correlations(X_train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for features
    
    Args:
        X_train_data: Training features
        
    Returns:
        DataFrame containing feature correlations
    """
    corr_matrix = X_train_data.corr().abs()
    return corr_matrix

def calculate_feature_importance(X_train_data: pd.DataFrame, y_train_data: pd.Series) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest
    
    Args:
        X_train_data: Training features
        y_train_data: Target variable
        
    Returns:
        DataFrame with feature importance rankings
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_data, y_train_data)
    
    # Create importance dataframe
    feature_importance = pd.DataFrame({
        'feature': X_train_data.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def plot_feature_importance(importance_df: pd.DataFrame, parameters: Dict) -> plt.Figure:
    """
    Create a horizontal bar plot of top N feature importances with value labels.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        parameters: Dictionary containing plot configuration

    Returns:
        A matplotlib Figure
    """
    # Get top N features based on parameters
    top_n = parameters.get("feature_selection_params", {}).get("top_n_features", 10)
    top_features = importance_df.sort_values("importance", ascending=False).head(top_n)

    # Create horizontal bar plot
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
    """
    Select features based on statistical tests (ANOVA F-test)
    
    Args:
        X_train_data: Training features
        y_train_data: Target variable
        params: Parameters including k_best_features
        
    Returns:
        List of selected feature names
    """
    k = params.get("feature_selection_params", {}).get("k_best_features", 10)
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train_data, y_train_data)
    
    # Get column names of selected features
    cols = X_train_data.columns[selector.get_support()]
    return list(cols)

def recursive_feature_elimination(X_train_data: pd.DataFrame, y_train_data: pd.Series,
                                 params: Dict) -> List[str]:
    """
    Perform Recursive Feature Elimination
    
    Args:
        X_train_data: Training features
        y_train_data: Target variable
        params: Parameters including n_features_to_select
        
    Returns:
        List of selected feature names
    """
    n_features = params.get("feature_selection_params", {}).get("n_features_to_select", 10)
    
    # Create the RFE object
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train_data, y_train_data)
    
    # Get selected features
    selected_features = X_train_data.columns[rfe.support_]
    return list(selected_features)

def feature_selection(
    feature_correlations: pd.DataFrame,
    feature_importance_scores: pd.DataFrame,
    statistical_features: List[str],
    rfe_features: List[str],
    parameters: Dict
) -> Dict:
    """
    Combine feature selection results using specified method.

    Args:
        feature_correlations: Correlation matrix of features
        feature_importance_scores: DataFrame with feature importances
        statistical_features: List of features selected using statistical tests
        rfe_features: List of features selected using RFE
        parameters: Configuration dictionary for feature selection

    Returns:
        Dictionary containing selected features and intermediate results
    """

    method = parameters.get("feature_selection_params", {}).get("combination_method", "union")

    if method == "intersection":
        final_features = list(set(statistical_features).intersection(set(rfe_features)))
    elif method == "weighted":
        # Get top N features from importance
        top_n = parameters.get("feature_selection_params", {}).get("top_n_features", 10)
        importance_features = list(feature_importance_scores['feature'].head(top_n))
        final_features = list(set(importance_features + statistical_features + rfe_features))
    else:  # Default to union
        final_features = list(set(statistical_features).union(set(rfe_features)))

    return {
        "best_columns": final_features,
        "feature_importance": feature_importance_scores,
        "correlation_matrix": feature_correlations,
        "statistical_features": statistical_features,
        "rfe_features": rfe_features
    }
