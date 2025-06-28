import joblib
import numpy as np
from pathlib import Path
import re

# Path to the models directory
MODEL_DIR = Path("/app/models")

def get_latest_model_path():
    """Find the most recent champion_model_XXX_.pkl file by version number."""
    pattern = re.compile(r"champion_model_(\d+)_\.pkl")
    candidates = [f for f in MODEL_DIR.glob("champion_model_*_*.pkl") if pattern.match(f.name)]

    if not candidates:
        raise FileNotFoundError("No champion model found in the models directory.")

    latest = max(candidates, key=lambda f: int(pattern.search(f.name).group(1)))
    return latest

def load_model():
    """Load the latest champion model and the corresponding best columns."""
    model_path = get_latest_model_path()
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

# import pickle

# def load_model(model_path="models/v1/model_and_meta.pkl"):
#     """
#     Loads:
#       - algo: the Surprise SVD model
#       - meta: dict mapping track_id -> {"track_name": ..., "artists": ...}
#     """
#     with open(model_path, "rb") as f:
#         algo, meta = pickle.load(f)
#     return algo, meta

# def recommend_for_user(algo, meta, user_id, k=5):
#     """
#     Returns top-k recommendations for user_id.
#     Uses meta dict instead of re-reading CSV.
#     """
#     track_ids = list(meta.keys())
#     scores = [(tid, algo.predict(user_id, tid).est) for tid in track_ids]
#     top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

#     results = []
#     for tid, score in top_k:
#         info = meta[tid]
#         results.append({
#             "track_id": tid,
#             "track_name": info["track_name"],
#             "artists": info["artists"],
#             "score": score
#         })
#     return results