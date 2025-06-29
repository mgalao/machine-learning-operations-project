import joblib
import numpy as np
from pathlib import Path
import re

# Path to the models directory
MODEL_DIR = Path("/app/models")


def load_model():
    """Load the latest champion model and the corresponding best columns."""
    model_path = MODEL_DIR / "model_champion.pkl"
    columns_path = MODEL_DIR / "best_cols.pkl"

    if not columns_path.exists():
        raise FileNotFoundError("best_cols.pkl not found in models directory.")

    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns

def predict(model, input_data: dict, columns: list):
    """
    Predict using the model with input features reordered as per production columns.

    Args:
        model: Trained sklearn-compatible model.
        input_data: Dict of feature values, e.g., {"age": 35, "amt": 200.0}
        columns: List of expected column names in order.

    Returns:
        Model prediction result.
    """
    try:
        features = [input_data[col] for col in columns]
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Missing required feature: {missing}")

    data = np.array(features).reshape(1, -1)
    return model.predict(data).tolist()[0]