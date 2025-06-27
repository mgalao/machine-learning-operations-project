import re
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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                model_cls: type,
                model_params: Dict[str, Any],
                parameters: Dict[str, Any],
                best_columns: list = None):
    """
    Train the final model with best params, log results, and save artifacts.

    Returns:
        trained_model, features_used, results_dict, shap_plot_figure
    """

    # Load MLflow experiment name and experiment ID
    logger.info("Loading MLflow experiment configuration...")
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.safe_load(f)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info(f"Using MLflow experiment: {experiment_name} (ID: {experiment_id})")

    # Enable automatic logging for scikit-learn models
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    # Start MLflow run for final training
    with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=f"FinalTrain_{model_cls.__name__}"):
        logger.info(f"Started MLflow run for final training of model: {model_cls.__name__}")
        mlflow.set_tag("stage", "final_training")
        mlflow.set_tag("model_type", model_cls.__name__)

        # Optional feature selection
        if parameters.get("use_feature_selection") and best_columns:
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

        # Calculate accuracy metrics
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        logger.info(f"Train accuracy: {acc_train:.4f}, Test accuracy: {acc_test:.4f}")

        # Collect results
        results_dict = {
            'classifier': model_cls.__name__,
            'train_score': acc_train,
            'test_score': acc_test
        }

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("test_accuracy", acc_test)

        # SHAP explainability for tree-based models
        plt_obj = None
        if hasattr(model, 'predict_proba') and 'Tree' in model_cls.__name__:
            logger.info("Generating SHAP summary plot for model explainability...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.initjs()

            # SHAP summary plot for class 1
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap.summary_plot(shap_values[1], X_train, feature_names=X_train.columns, show=False)
            else:
                shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)

            plt.tight_layout()
            plt_obj = plt
            logger.info("SHAP summary plot generated.")
        else:
            logger.info("SHAP explanation skipped: model is not tree-based or doesn't support predict_proba.")

        # Champion model comparison
        is_new_champion = True
        champion_path = parameters.get("champion_model_path", os.path.join(os.getcwd(), 'data', '06_models', 'champion_model_000_.pkl'))
        # Use regex to extract the number reliably
        model_name = os.path.basename(champion_path)
        match = re.search(r'champion_model_(\d+)_\.pkl', model_name)
        if match:
            champion_model_number = match.group(1)
        else:
            champion_model_number = "000"  # Default if pattern doesn't match
        champion_plus_one = str(int(champion_model_number) + 1).zfill(3)
        challenger_path = champion_path.replace(f"_{champion_model_number}_", f"_{champion_plus_one}_")
        

        if os.path.exists(champion_path):
            logger.info("Loading existing champion model for comparison...")
            try:
                with open(champion_path, 'rb') as f:
                    champion_model = pickle.load(f)
                champion_pred = champion_model.predict(X_test)
                champion_score = accuracy_score(y_test, champion_pred)
                logger.info(f"Champion test accuracy: {champion_score:.4f}")
                logger.info(f"Candidate test accuracy: {acc_test:.4f}")

                if acc_test <= champion_score:
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
            with open(challenger_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"New champion model saved to {challenger_path}")

    return model, X_train.columns, results_dict, plt_obj, is_new_champion