import pandas as pd
import logging
from typing import Dict, Any
import numpy as np
import pickle
import warnings
import yaml

import optuna
from optuna.samplers import TPESampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import mlflow

warnings.filterwarnings("ignore", category=Warning)
logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    """Get or create an MLflow experiment by name."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def model_selection(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame,
                    parameters: Dict[str, Any],
                    parameters_optuna: Dict[str, Any]):
    """
    Compare baseline models and tune hyperparameters with Optuna.

    Returns:
        best_model_cls: Class of best model
        best_params: Dict of best hyperparameters
        best_score: Best validation accuracy
    """

    # Define candidate models for baseline comparison
    models_dict = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'XGBoostClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    }

    initial_results = {}

    # Load experiment name and ID from config
    with open('conf/local/mlflow.yml') as f:
        experiment_name = parameters["mlflow_model_selection_experiment"]
        experiment_id = _get_or_create_experiment_id(experiment_name)

    logger.info('Initial model comparison...')

    # Baseline comparison
    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=f"Baseline_{model_name}"):
            mlflow.set_tag("stage", "baseline_comparison")   # tag for tracking purpose
            mlflow.set_tag("model_type", model_name)

            # Log model parameters and metadata
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train_flat = np.ravel(y_train)
            model.fit(X_train, y_train_flat)
            score = model.score(X_test, y_test)
            initial_results[model_name] = score
            logger.info(f"Logged model {model_name} with score {score:.4f}")

    # Select best model from baseline
    best_model_name = max(initial_results, key=initial_results.get)
    best_model_cls = type(models_dict[best_model_name])
    logger.info(f"Best baseline model: {best_model_name} with score {initial_results[best_model_name]:.4f}")

    logger.info('Optuna hyperparameter tuning...')

    # Optuna hyperparameter tuning
    def objective(trial):
        """Objective function for Optuna hyperparameter tuning."""
        search_space = parameters_optuna['optuna_search']['search_spaces'][best_model_name]
        trial_params = {}

        # Sample each hyperparameter from its defined space
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
            elif param_config['type'] == 'categorical':
                trial_params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])

        # Train model using sampled hyperparameters
        model = best_model_cls(**trial_params)
        model.fit(X_train, y_train.values.ravel())
        pred = model.predict(X_test)

        # Return accuracy as objective metric
        return accuracy_score(y_test, pred)

    # Read tuning config
    direction = parameters_optuna['optuna_search'].get('direction', 'maximize')
    n_trials = parameters_optuna['optuna_search'].get('n_trials', 30)

    # Run Optuna study
    study = optuna.create_study(direction=direction, sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials)

    # Get best model and hyperparameters
    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best trial score: {best_score:.4f}")

    # Log tuning info to MLflow
    with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=f"Optuna_{best_model_name}"):
        mlflow.set_tag("stage", "optuna_tuning")
        mlflow.set_tag("model_type", best_model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_score)

    return best_model_cls, best_params, best_score