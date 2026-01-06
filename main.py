from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="California Housing Price Predictor")

# Load model & metadata
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

@app.get("/")
def health_check():
    return {"status": "Model is running"}

@app.post("/predict")
def predict_price(data: HouseFeatures):
    input_data = np.array([[
        data.longitude,
        data.latitude,
        data.housing_median_age,
        data.total_rooms,
        data.total_bedrooms,
        data.population,
        data.households,
        data.median_income
    ]])

    prediction = model.predict(input_data)

    return {
        "predicted_house_value": float(prediction[0])
    }
