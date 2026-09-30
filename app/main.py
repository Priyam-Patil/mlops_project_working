from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictionInput(BaseModel):
    feature1: float
    feature2: float


@app.get("/")
def home():
    return {"message": "MLOps API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    return {
        "feature1": data.feature1,
        "feature2": data.feature2,
        "prediction": 1
    }