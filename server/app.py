# server/app.py

from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "champion_model.joblib"
)


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_load_error = str(e)


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality using the champion ML model.",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# ---------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Wine Quality Prediction API is running"
    }


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get("/health")
def health():

    if not model_loaded:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": model_load_error,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API and model are running",
    }


# ---------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------

@app.post("/predict")
def predict(features: WineFeatures):

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    # Convert request to DataFrame
    input_data = pd.DataFrame(
        [
            {
                "fixed acidity": features.fixed_acidity,
                "volatile acidity": features.volatile_acidity,
                "citric acid": features.citric_acid,
                "residual sugar": features.residual_sugar,
                "chlorides": features.chlorides,
                "free sulfur dioxide": features.free_sulfur_dioxide,
                "total sulfur dioxide": features.total_sulfur_dioxide,
                "density": features.density,
                "pH": features.pH,
                "sulphates": features.sulphates,
                "alcohol": features.alcohol,
            }
        ]
    )

    try:

        prediction = model.predict(
            input_data
        )

        return {
            "prediction": float(prediction[0])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ---------------------------------------------------------------------
# Run with Uvicorn
# ---------------------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

