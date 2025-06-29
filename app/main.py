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