import pandas as pd
import numpy as np
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

def feature_selection(X_train_data: pd.DataFrame, y_train_data: pd.Series, 
                     parameters: Dict) -> Dict:
    """
    Main feature selection function that combines multiple methods
    
    Args:
        X_train_data: Training features
        y_train_data: Target variable
        parameters: Pipeline parameters
        
    Returns:
        Dictionary containing selected features and metadata
    """
    # Calculate feature correlations
    correlations = calculate_feature_correlations(X_train_data)
    
    # Calculate feature importance
    importance = calculate_feature_importance(X_train_data, y_train_data)
    
    # Statistical feature selection
    statistical_features = select_statistical_features(X_train_data, y_train_data, parameters)
    
    # Recursive feature elimination
    rfe_features = recursive_feature_elimination(X_train_data, y_train_data, parameters)
    
    # Combine results - use the method specified in parameters or default to "union"
    method = parameters.get("feature_selection_params", {}).get("combination_method", "union")

    
    if method == "intersection":
        final_features = list(set(statistical_features).intersection(set(rfe_features)))
    elif method == "weighted":
        # Get top N features from importance
        top_n = parameters.get("feature_selection_params", {}).get("top_n_features", 15)
        importance_features = list(importance['feature'].head(top_n))
        final_features = list(set(importance_features + statistical_features + rfe_features))
    else:  # Default to union
        final_features = list(set(statistical_features).union(set(rfe_features)))
    
    # Return the results including metadata for analysis
    return {
        "best_columns": final_features,
        "feature_importance": importance,
        "correlation_matrix": correlations,
        "statistical_features": statistical_features,
        "rfe_features": rfe_features
    }