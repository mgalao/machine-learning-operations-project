import pytest
import pandas as pd
import numpy as np
import pickle
import logging
from unittest.mock import patch, MagicMock

# Import the function to test
from src.fraud_project.pipelines.model_predict.nodes import model_predict

# Setup logging
logger = logging.getLogger(__name__)

@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    return X

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = np.random.randint(0, 2, size=100)
    return model

def test_model_predict_basic_functionality(dummy_data, mock_model):
    """Test basic functionality of model_predict function."""
    X = dummy_data
    columns = ['feat1', 'feat2', 'feat3']
    
    # Call the function
    with patch('logging.Logger.info') as mock_logger:
        result_df, describe_dict = model_predict(X, mock_model, columns)
    
    # Assertions
    assert 'y_pred' in result_df.columns
    assert len(result_df) == len(dummy_data)
    assert isinstance(describe_dict, dict)
    assert mock_model.predict.called
    assert mock_model.predict.call_args[0][0].equals(X[columns])
    mock_logger.assert_any_call('Service predictions created.')

def test_model_predict_with_empty_dataframe():
    """Test model_predict with empty DataFrame."""
    X = pd.DataFrame(columns=['feat1', 'feat2', 'feat3'])
    model = MagicMock()
    model.predict.return_value = np.array([])
    columns = ['feat1', 'feat2']
    
    # Call the function
    with patch('logging.Logger.info') as mock_logger:
        result_df, describe_dict = model_predict(X, model, columns)
    
    # Assertions
    assert 'y_pred' in result_df.columns
    assert len(result_df) == 0
    assert isinstance(describe_dict, dict)
    assert len(describe_dict) == 0  # Empty DataFrame describe() should return empty dict

def test_model_predict_all_columns(dummy_data, mock_model):
    """Test model_predict using all columns."""
    X = dummy_data
    columns = list(X.columns)  # Use all columns
    
    # Call the function
    result_df, describe_dict = model_predict(X, mock_model, columns)
    
    # Assertions
    assert 'y_pred' in result_df.columns
    assert mock_model.predict.called
    assert mock_model.predict.call_args[0][0].equals(X[columns])
    assert set(describe_dict.keys()) == set(X.columns)

def test_model_predict_with_real_model(dummy_data):
    """Test model_predict with an actual model instance."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a simple model
    X = dummy_data
    y = np.random.randint(0, 2, size=len(X))
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    columns = list(X.columns)
    
    # Call the function
    result_df, describe_dict = model_predict(X, model, columns)
    
    # Assertions
    assert 'y_pred' in result_df.columns
    assert len(result_df['y_pred'].unique()) <= 2  # Binary classification
    assert isinstance(describe_dict, dict)