import pytest
import pandas as pd # type: ignore
import numpy as np # type: ignore
from src.fraud_project.pipelines.feature_selection.nodes import (
    calculate_feature_correlations,
    calculate_feature_importance,
    select_statistical_features,
    recursive_feature_elimination,
    feature_selection
)


@pytest.fixture # pytest fixture is a function that runs before each test function to set up any state you want to share across tests
def sample_data():
    """Create sample data for testing."""
    # create a simple dataset with 5 features and 100 samples
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'feature4': np.random.normal(0, 1, 100),
        'feature5': np.random.normal(0, 1, 100),
    })
    
    # create a linearly dependent target variable
    y = (0.5 * X['feature1'] + 0.8 * X['feature3'] > 0).astype(int)
    
    # make a feature that is correlated with feature1
    X['feature2'] = X['feature1'] * 0.9 + np.random.normal(0, 0.1, 100)
    
    return X, y


@pytest.fixture
def parameters():
    """Create parameters for testing."""
    return {
        "feature_selection_params": {
            "k_best_features": 3,
            "n_features_to_select": 2,
            "top_n_features": 3,
            "combination_method": "union"
        }
    }


def test_calculate_feature_correlations(sample_data):
    """Test the correlation matrix calculation."""
    X, _ = sample_data
    corr_matrix = calculate_feature_correlations(X)
    
    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (5, 5)
    
    # the diagonal should be 1 (self-correlation)
    assert np.all(np.diag(corr_matrix) == 1)
    
    # feature 1 and feature 2 should be highly correlated
    assert corr_matrix.loc['feature1', 'feature2'] > 0.7
    assert corr_matrix.loc['feature2', 'feature1'] > 0.7


def test_calculate_feature_importance(sample_data):
    """Test the feature importance calculation."""
    X, y = sample_data
    importance = calculate_feature_importance(X, y)
    
    # check output type and shape
    assert isinstance(importance, pd.DataFrame)
    assert importance.shape == (5, 2)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert importance['importance'].is_monotonic_decreasing    
    top_features = importance['feature'].head(3).values
    assert 'feature1' in top_features or 'feature3' in top_features


def test_select_statistical_features(sample_data, parameters):
    """Test the statistical feature selection."""
    X, y = sample_data
    selected = select_statistical_features(X, y, parameters)
    
    # type and length check for the selected features
    assert isinstance(selected, list)
    assert len(selected) == parameters["feature_selection_params"]["k_best_features"]
    
    # the selected features should be in the original dataset 
    for feature in selected:
        assert feature in X.columns


def test_recursive_feature_elimination(sample_data, parameters):
    """Test the RFE feature selection."""
    X, y = sample_data
    selected = recursive_feature_elimination(X, y, parameters)
    
    # type and length check for the selected features
    assert isinstance(selected, list)
    assert len(selected) == parameters["feature_selection_params"]["n_features_to_select"]
    
    # the selected features should be in the original dataset
    for feature in selected:
        assert feature in X.columns


def test_feature_selection_main(sample_data, parameters):
    """Test the main feature selection function."""
    X, y = sample_data
    results = feature_selection()
    
    # check output type and structure
    assert isinstance(results, dict)
    assert "best_columns" in results
    assert "feature_importance" in results
    assert "correlation_matrix" in results
    assert "statistical_features" in results
    assert "rfe_features" in results
    assert isinstance(results["best_columns"], list)
    
    # should include all features from statistical and RFE
    stat_features = results["statistical_features"]
    rfe_features = results["rfe_features"]
    best_columns = results["best_columns"]
    
    # all features from both methods should be included
    for feature in stat_features:
        assert feature in best_columns
    for feature in rfe_features:
        assert feature in best_columns


def test_feature_selection_intersection(sample_data):
    """Test the feature selection with intersection method."""
    X, y = sample_data
    parameters = {
        "feature_selection_params": {
            "k_best_features": 3,
            "n_features_to_select": 2,
            "combination_method": "intersection"
        }
    }
    
    # calculate all required inputs for feature_selection
    feature_correlations = calculate_feature_correlations(X)
    feature_importance_scores = calculate_feature_importance(X, y)
    statistical_features = select_statistical_features(X, y, parameters)
    rfe_features = recursive_feature_elimination(X, y, parameters)
    
    # call feature_selection with all required parameters
    results = feature_selection(
        feature_correlations,
        feature_importance_scores,
        statistical_features,
        rfe_features,
        parameters
    )
    
    # best columns should the intersection of statistical and RFE features
    stat_features = set(results["statistical_features"])
    rfe_features = set(results["rfe_features"])
    expected_intersection = stat_features.intersection(rfe_features)
    
    assert set(results["best_columns"]) == expected_intersection


def test_feature_selection_weighted(sample_data):
    """Test the feature selection with weighted method."""
    X, y = sample_data
    parameters = {
        "feature_selection_params": {
            "k_best_features": 3,
            "n_features_to_select": 2,
            "top_n_features": 3,
            "combination_method": "weighted"
        }
    }
    
    # calculate all required inputs for feature_selection
    feature_correlations = calculate_feature_correlations(X)
    feature_importance_scores = calculate_feature_importance(X, y)
    statistical_features = select_statistical_features(X, y, parameters)
    rfe_features = recursive_feature_elimination(X, y, parameters)
    
    # call feature_selection with all required parameters
    results = feature_selection(
        feature_correlations,
        feature_importance_scores,
        statistical_features,
        rfe_features,
        parameters
    )
    
    # all best columns must include top features from importance
    importance_df = results["feature_importance"]
    top_importance = set(importance_df['feature'].head(3))
    
    # top features should be in the best columns
    for feature in top_importance:
        assert feature in results["best_columns"]