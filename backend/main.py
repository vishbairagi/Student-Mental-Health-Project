"""
FastAPI Backend Application
Student Social Media & Mental Health Prediction System
======================================================
Provides RESTful API endpoints for predicting student Mental Health Scores
using the trained Scikit-learn Pipeline saved at models/mental_health_model.pkl.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Add current directory to sys.path so schemas can be imported regardless of execution working dir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from schemas import (  # noqa: E402
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    WelcomeResponse,
    ErrorResponse
)

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mental_health_api")

# ── Model Path Resolution ────────────────────────────────────────────────────
# Support multiple working directory contexts (running from backend/, root/, or Render)
CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(CURRENT_DIR, "..", "models", "mental_health_model.pkl")),
    os.path.abspath(os.path.join(CURRENT_DIR, "models", "mental_health_model.pkl")),
    os.path.abspath(os.path.join(os.getcwd(), "models", "mental_health_model.pkl")),
    os.path.abspath(os.path.join(os.getcwd(), "student-mental-health-project", "models", "mental_health_model.pkl")),
]

MODEL_PATH: Optional[str] = None
for candidate in CANDIDATE_PATHS:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

# Global reference to the loaded model pipeline
model_pipeline = None

EXPECTED_FEATURES: List[str] = [
    "Age",
    "Gender",
    "Country",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level"
]


def load_model():
    """Load the serialized Scikit-learn pipeline into memory."""
    global model_pipeline, MODEL_PATH
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found! Checked locations: {CANDIDATE_PATHS}")
        raise FileNotFoundError(f"Model file 'mental_health_model.pkl' could not be located in any known path: {CANDIDATE_PATHS}")

    logger.info(f"Loading ML pipeline from: {MODEL_PATH}")
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model successfully loaded. Steps: {list(model_pipeline.named_steps.keys())}")


# Attempt initial model load on module import
try:
    load_model()
except Exception as _load_err:
    logger.warning(f"Initial model load postponed to lifespan startup: {_load_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure model is loaded on server startup."""
    global model_pipeline
    if model_pipeline is None:
        try:
            load_model()
        except Exception as exc:
            logger.error(f"Failed to load model during startup: {exc}")
    yield
    logger.info("FastAPI backend shutting down...")


# ── FastAPI App Initialization ───────────────────────────────────────────────
app = FastAPI(
    title="Student Social Media & Mental Health Prediction API",
    description=(
        "Production-ready FastAPI backend serving an end-to-end Machine Learning pipeline "
        "to predict student Mental Health Scores (1-10 scale) based on lifestyle, social media usage, "
        "and academic indicators."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
# Allow frontend web apps (HTML/CSS/JS) to make cross-origin fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production on Render, can restrict to specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Custom Exception Handlers ────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic 422 validation errors with clear, beginner-friendly messages."""
    errors = []
    for err in exc.errors():
        field_name = " -> ".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field_name or "body", "message": msg})

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation Error: One or more input fields failed validation.",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch unhandled errors and return a standardized 500 JSON response."""
    logger.error(f"Unhandled error processing request: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "details": str(exc)
        }
    )


# ── Interpretation Helper ────────────────────────────────────────────────────
def interpret_score(score: float):
    """
    Map continuous predicted score (1-10) to clinically sensible, human-friendly categories.
    """
    if score >= 7.5:
        return (
            "Thriving / High Mental Well-being: Student exhibits strong positive indicators across sleep, balance, and controlled media use.",
            "Low Risk"
        )
    elif score >= 6.0:
        return (
            "Good / Moderate Well-being: Student reports stable mental balance with manageable stress levels.",
            "Moderate Risk"
        )
    elif score >= 4.5:
        return (
            "Mild-to-Moderate Distress: Elevated screen time, high unlocks, or high stress appear to be impacting mental wellness. Healthy habit interventions recommended.",
            "Elevated Risk"
        )
    else:
        return (
            "High Distress / Action Recommended: Strong signs of mental health strain associated with heavy social media dependence, sleep deprivation, or severe stress.",
            "High Risk"
        )


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get(
    "/",
    response_model=WelcomeResponse,
    tags=["General"],
    summary="Root Welcome Endpoint"
)
async def root():
    """Root endpoint welcoming users and directing them to interactive documentation."""
    return WelcomeResponse(
        message="Welcome to the Student Social Media & Mental Health Prediction API!",
        project="Student Social Media & Mental Health Prediction System",
        version="1.0.0",
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Health Check & Model Diagnostics"
)
async def health_check():
    """
    Check API operational health, model loading status, and input feature configuration.
    """
    global model_pipeline
    if model_pipeline is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline unavailable: {str(exc)}"
            )

    regressor = model_pipeline.named_steps.get("regressor")
    regressor_name = type(regressor).__name__ if regressor else "Unknown"

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=regressor_name,
        model_file_path=MODEL_PATH or "Unknown",
        features_count=len(EXPECTED_FEATURES),
        expected_features=EXPECTED_FEATURES,
        api_version="1.0.0"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Machine Learning"],
    summary="Predict Student Mental Health Score"
)
async def predict(request_data: PredictionRequest):
    """
    Predict student Mental Health Score from 12 validated lifestyle and usage features.
    
    - **Age**: Integer (10 to 100)
    - **Gender**: Categorical ('Male', 'Female')
    - **Country**: Country name string
    - **Academic_Level**: 'Undergraduate', 'Graduate', 'High School'
    - **Most_Used_Platform**: Platform name (e.g. 'Instagram', 'TikTok', 'YouTube')
    - **Purpose_Of_Use**: 'Entertainment', 'Education', 'Networking', 'News'
    - **Avg_Daily_Usage_Hours**: Float (0.0 to 24.0)
    - **Daily_Unlocks**: Integer (0 to 1000)
    - **Study_Hours**: Float (0.0 to 24.0)
    - **Physical_Activity_Hours**: Float (>= 0.0)
    - **Sleep_Hours_Per_Night**: Float (0.0 to 24.0)
    - **Stress_Level**: 'Low', 'Medium', 'High', 'Very High'
    """
    global model_pipeline
    if model_pipeline is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline unavailable: {str(exc)}"
            )

    try:
        # Convert Pydantic model to dictionary
        data_dict = request_data.model_dump()

        # Build single-row DataFrame with exact columns matching model expectation
        input_df = pd.DataFrame([data_dict])[EXPECTED_FEATURES]

        # Execute prediction through full preprocessing + regressor pipeline
        raw_pred = float(model_pipeline.predict(input_df)[0])

        # Clamp prediction to sensible bounds [1.0, 10.0]
        clamped_score = max(1.0, min(10.0, raw_pred))
        rounded_score = round(clamped_score, 2)

        # Generate interpretation
        interpretation_text, risk_cat = interpret_score(rounded_score)

        regressor = model_pipeline.named_steps.get("regressor")
        regressor_name = type(regressor).__name__ if regressor else "Trained Pipeline"

        return PredictionResponse(
            success=True,
            predicted_mental_health_score=rounded_score,
            raw_prediction=round(raw_pred, 4),
            interpretation=interpretation_text,
            risk_level=risk_cat,
            model_name=regressor_name
        )

    except Exception as exc:
        logger.error(f"Prediction execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Allow running directly: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
