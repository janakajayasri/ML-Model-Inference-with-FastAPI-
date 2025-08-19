from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import sklearn as st
# Load the trained model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")

app = FastAPI(title="Iris Classification API", description="API for Iris flower species classification")

# Define input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to model format
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
            input_data.sepal_length * input_data.sepal_width,  # sepal_area
            input_data.petal_length * input_data.petal_width   # petal_area
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))
        
        return PredictionOutput(prediction=prediction, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "sepal_area",
            "petal_area"
        ]
    }