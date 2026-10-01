from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model
# Load the already-trained ML model.
model = joblib.load("model/model.joblib")


class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get("/")
def home():
    return {"message": "MLOps API is running"}


@app.post("/predict")
def predict(data: PredictionInput):

    input_data = [[
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4
    ]]
# Give the user's input to the ML model and get its prediction.
    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }
    