from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.model import predictor

app = FastAPI(title="Fraud Detection API")

model = None
production_columns = None

class InputData(BaseModel):
    data: Dict[str, Any]

@app.on_event("startup")
async def load_model_on_startup():
    global model, production_columns
    model, production_columns = predictor.load_model()

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running."}

@app.post("/predict")
def make_prediction(input_data: InputData):
    try:
        prediction = predictor.predict(model, input_data.data, production_columns)
        return {"prediction": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# from fastapi import FastAPI
# from pydantic import BaseModel
# from app.model.recommender import load_model, recommend_for_user

# app = FastAPI()
# algo, meta = load_model("models/v1/model_and_meta.pkl")

# class RecommendRequest(BaseModel):
#     user_id: int
#     num_recommendations: int = 5

# @app.post("/recommend")
# def recommend(req: RecommendRequest):
#     recs = recommend_for_user(algo, meta, req.user_id, req.num_recommendations)
#     return {"user_id": req.user_id, "recommendations": recs}

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Spotify Recommender API"}