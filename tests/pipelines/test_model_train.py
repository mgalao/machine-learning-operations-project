import pytest
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import patch, mock_open, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.fraud_project.pipelines.model_train.nodes import model_train
import matplotlib.pyplot as plt


@pytest.fixture # Create dummy data for testing that can be reused across tests
def dummy_data():
    """Create dummy data for testing."""
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    X_test = pd.DataFrame(np.random.rand(50, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    y_train = pd.DataFrame(np.random.randint(0, 2, size=100), columns=['target'])
    y_test = pd.DataFrame(np.random.randint(0, 2, size=50), columns=['target'])
    return X_train, X_test, y_train, y_test

@pytest.fixture
def mock_mlflow():
    """Mock MLflow functionality for testing.
    
    MagicMock is a subclass of Mock with default implementations
    of most of the magic methods. You can use MagicMock without having to
    configure the magic methods yourself. <- this is from the documentation
    
    
    """
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.sklearn.autolog') as mock_autolog, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.set_tag') as mock_set_tag, \
         patch('builtins.open', mock_open(read_data='tracking:\n  experiment:\n    name: test_exp')):
        
        
        
        
        # configure the mock experiment
        mock_exp = MagicMock()
        mock_exp.experiment_id = 'test_id'
        mock_get_exp.return_value = mock_exp
        
        # create a context manager mock
        context_manager_mock = MagicMock()
        mock_start_run.return_value.__enter__.return_value = context_manager_mock
        
        yield {
            'start_run': mock_start_run,
            'get_exp': mock_get_exp,
            'autolog': mock_autolog,
            'log_metric': mock_log_metric,
            'set_tag': mock_set_tag
        }

def test_model_train_basic_functionality(dummy_data, mock_mlflow, tmp_path):
    """Test basic functionality of model_train function."""
    X_train, X_test, y_train, y_test = dummy_data
    
    # mock the champion model path
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    
    # set up parameters
    model_cls = RandomForestClassifier
    model_params = {'n_estimators': 10, 'max_depth': 3}
    parameters = {
        'use_feature_selection': False,
        'champion_model_path': champion_path
    }
    
    # call the function
    with patch('os.path.exists', return_value=False):  # No champion model exists
        model, features_used, results_dict, plt_obj, is_new_champion = model_train(
            X_train, X_test, y_train, y_test, model_cls, model_params, parameters
        )
    
    # assertions
    assert isinstance(model, RandomForestClassifier)
    assert len(features_used) == 5
    assert list(features_used) == ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    assert results_dict['classifier'] == 'RandomForestClassifier'
    assert 'train_score' in results_dict
    assert 'test_score' in results_dict
    assert is_new_champion is True

def test_model_train_with_feature_selection(dummy_data, mock_mlflow, tmp_path):
    """Test model_train function with feature selection enabled."""
    X_train, X_test, y_train, y_test = dummy_data
    
    # mock the champion model path
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    
    # set up parameters
    model_cls = RandomForestClassifier
    model_params = {'n_estimators': 10, 'max_depth': 3}
    parameters = {
        'use_feature_selection': True,
        'champion_model_path': champion_path
    }
    best_columns = ['feat1', 'feat3', 'feat5']  # selected features
    
    with patch('os.path.exists', return_value=False):
        model, features_used, results_dict, plt_obj, is_new_champion = model_train(
            X_train, X_test, y_train, y_test, model_cls, model_params, parameters, best_columns
        )
    
    assert len(features_used) == 3
    assert list(features_used) == best_columns

def test_model_train_with_champion_comparison(dummy_data, mock_mlflow, tmp_path):
    """Test model_train function with champion model comparison."""
    X_train, X_test, y_train, y_test = dummy_data
    
    # create a mock champion model
    champion_model = RandomForestClassifier()
    champion_model.fit(X_train, np.ravel(y_train))
    
    # save the champion model
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    with open(champion_path, 'wb') as f:
        pickle.dump(champion_model, f)
    
    # setup parameters
    model_cls = RandomForestClassifier
    model_params = {'n_estimators': 10, 'max_depth': 3}
    parameters = {
        'use_feature_selection': False,
        'champion_model_path': champion_path
    }
    
    # test when new model is better
    with patch('sklearn.metrics.accuracy_score') as mock_accuracy:
        # mock accuracy scores: train_acc, test_acc, champion_acc, test_acc again
        mock_accuracy.side_effect = [0.7, 0.8, 0.6, 0.8]
        
        with patch('os.path.exists', return_value=True):
            _, _, _, _, is_new_champion = model_train(
                X_train, X_test, y_train, y_test, model_cls, model_params, parameters
            )
    
    assert is_new_champion is True

def test_model_train_champion_is_better(dummy_data, mock_mlflow, tmp_path):
    """Test when champion model performs better."""
    X_train, X_test, y_train, y_test = dummy_data
    
    # create and save a champion model
    champion_model = RandomForestClassifier()
    champion_model.fit(X_train, np.ravel(y_train))
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    with open(champion_path, 'wb') as f:
        pickle.dump(champion_model, f)
    
    # setup parameters
    model_cls = RandomForestClassifier
    model_params = {'n_estimators': 5, 'max_depth': 2}
    parameters = {
        'use_feature_selection': False,
        'champion_model_path': champion_path
    }
    
    with patch('sklearn.metrics.accuracy_score') as mock_accuracy:
        # champion model performs better
        mock_accuracy.side_effect = [0.6, 0.6, 0.8, 0.6]
        
        with patch('os.path.exists', return_value=True):
            _, _, _, _, is_new_champion = model_train(
                X_train, X_test, y_train, y_test, model_cls, model_params, parameters
            )
    
    assert is_new_champion is False

def test_model_train_with_shap_explainability(dummy_data, mock_mlflow, tmp_path):
    """Test SHAP explainability for tree-based models."""
    X_train, X_test, y_train, y_test = dummy_data
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    model_cls = RandomForestClassifier
    model_params = {'n_estimators': 10}
    parameters = {
        'use_feature_selection': False,
        'champion_model_path': champion_path
    }
    
    with patch('shap.TreeExplainer') as mock_explainer, \
         patch('shap.summary_plot') as mock_summary_plot, \
         patch('shap.initjs') as mock_initjs, \
         patch('os.path.exists', return_value=False):
        
        mock_explainer_instance = MagicMock()
        mock_explainer.return_value = mock_explainer_instance
        mock_explainer_instance.shap_values.return_value = [
            np.random.random((100, 5)), 
            np.random.random((100, 5))
        ]
        
        _, _, _, plt_obj, _ = model_train(
            X_train, X_test, y_train, y_test, model_cls, model_params, parameters
        )
    
    assert plt_obj is not None
    assert mock_explainer.called
    assert mock_summary_plot.called

def test_non_tree_based_model(dummy_data, mock_mlflow, tmp_path):
    """Test with a non-tree-based model."""
    X_train, X_test, y_train, y_test = dummy_data
    champion_path = str(tmp_path / "champion_model_000_.pkl")
    
    # use a non-tree based model
    model_cls = LogisticRegression
    model_params = {'C': 1.0}
    parameters = {
        'use_feature_selection': False,
        'champion_model_path': champion_path
    }
    
    with patch('os.path.exists', return_value=False):
        model, features_used, results_dict, plt_obj, is_new_champion = model_train(
            X_train, X_test, y_train, y_test, model_cls, model_params, parameters
        )
    
    assert isinstance(model, LogisticRegression)
    assert plt_obj is None  # SHAP plot should not be generated