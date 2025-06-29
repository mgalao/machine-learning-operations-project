import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import shap
import matplotlib.pyplot as plt
from mlflow.sklearn import load_model

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                parameters: Dict[str, Any],
                best_columns: list = None):
    """
    Train the final model with best params, log results, and save artifacts.

    Returns:
        trained_model, features_used, results_dict, shap_plot_figure
    """

    experiment_name = "fraud_project_model_selection"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    logger.info(f"Using MLflow experiment: {experiment_name} (ID: {experiment_id})")

    mlflow_run_name_prefix = parameters.get("mlflow_run_tag", "Optuna_Best")

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Find runs where run name starts with the given prefix
    runs = client.search_runs(
        experiment.experiment_id,
        f"attributes.run_name LIKE '{mlflow_run_name_prefix}%'"
    )

    if not runs:
        raise FileNotFoundError(
            f"No MLflow run found with name starting with '{mlflow_run_name_prefix}' in experiment '{experiment_name}'"
        )

    challenger_run_id = runs[0].info.run_id
    logger.info(f"Loading challenger model from MLflow run ID: {challenger_run_id}")

    # Load model
    model_uri = f"runs:/{challenger_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    model_cls = type(model)
    model_params = model.get_params()

    # Start MLflow run for final training
    with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=f"FinalTrain_{model_cls.__name__}"):
        logger.info(f"Started MLflow run for final training of model: {model_cls.__name__}")
        mlflow.set_tag("stage", "final_training")
        mlflow.set_tag("model_type", model_cls.__name__)

        # Optional feature selection
        if parameters.get("use_feature_selection") and best_columns is not None and len(best_columns) > 0:
            logger.info("Applying feature selection based on selected columns...")
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]
        else:
            logger.info("No feature selection applied.")

        # Train the model with best parameters
        logger.info(f"Training {model_cls.__name__} with params: {model_params}")
        model = model_cls(**model_params)
        model.fit(X_train, np.ravel(y_train))
        logger.info("Model training completed.")

        # Predict on training and test sets
        logger.info("Generating predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        acc_train = accuracy_score(y_train, y_train_pred)
        prec_train = precision_score(y_train, y_train_pred, average='macro')
        rec_train = recall_score(y_train, y_train_pred, average='macro')
        f1_train = f1_score(y_train, y_train_pred, average='macro')

        acc_test = accuracy_score(y_test, y_test_pred)
        prec_test = precision_score(y_test, y_test_pred, average='macro')
        rec_test = recall_score(y_test, y_test_pred, average='macro')
        f1_test = f1_score(y_test, y_test_pred, average='macro')

        logger.info(f"Train macro F1: {f1_train:.4f}, Test macro F1: {f1_test:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("train_macro_precision", prec_train)
        mlflow.log_metric("train_macro_recall", rec_train)
        mlflow.log_metric("train_macro_f1", f1_train)

        mlflow.log_metric("test_accuracy", acc_test)
        mlflow.log_metric("test_macro_precision", prec_test)
        mlflow.log_metric("test_macro_recall", rec_test)
        mlflow.log_metric("test_macro_f1", f1_test)

        # Collect results
        results_dict = {
            'classifier': model_cls.__name__,
            'train_accuracy': acc_train,
            'train_macro_precision': prec_train,
            'train_macro_recall': rec_train,
            'train_macro_f1': f1_train,
            'test_accuracy': acc_test,
            'test_macro_precision': prec_test,
            'test_macro_recall': rec_test,
            'test_macro_f1': f1_test
        }

        # SHAP explainability
        logger.info("Generating SHAP summary plot for model explainability...")

        plt_obj = None
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)

            shap.initjs()
            shap.summary_plot(shap_values, X_train, show=False)
            plt.tight_layout()
            plt_obj = plt
            logger.info("SHAP summary plot generated.")

        # If SHAP fails, log a warning and return an empty figure
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            plt_obj = plt.figure()

        # Champion model comparison using macro F1
        is_new_champion = True
        champion_path = parameters.get("champion_model_path", os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'))

        if os.path.exists(champion_path):
            logger.info("Loading existing champion model for comparison...")
            try:
                with open(champion_path, 'rb') as f:
                    champion_model = pickle.load(f)
                champion_pred = champion_model.predict(X_test)
                champion_f1 = f1_score(y_test, champion_pred, average='macro')
                logger.info(f"Champion macro F1 score: {champion_f1:.4f}")
                logger.info(f"Candidate macro F1 score: {f1_test:.4f}")

                if f1_test <= champion_f1:
                    logger.info("New model did NOT outperform the current champion. Existing model retained.")
                    is_new_champion = False
                else:
                    logger.info("New model outperformed the current champion. Replacing champion model.")
            except Exception as e:
                logger.warning(f"Error loading or scoring champion model. Proceeding with new model. Error: {e}")
        else:
            logger.info("No existing champion model found. Proceeding to save current model as new champion.")

        # Save new model only if it's better
        if is_new_champion:
            with open(champion_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"New champion model saved to {champion_path}")

    return model, X_train.columns, results_dict, plt_obj, is_new_champion
