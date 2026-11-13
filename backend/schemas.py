"""
Pydantic schemas for request validation, prediction output, and health status
for the Student Social Media & Mental Health Prediction System.
"""

from typing import Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Allowed categorical values for strict validation
VALID_STRESS_LEVELS = {"Low", "Medium", "High", "Very High"}
VALID_ACADEMIC_LEVELS = {"High School", "Undergraduate", "Graduate"}
VALID_PURPOSES = {"Education", "Entertainment", "Networking", "News"}


class PredictionRequest(BaseModel):
    """
    Input schema containing exactly the 12 features expected by the trained ML pipeline.
    Target column (Mental_Health_Score) is intentionally excluded to prevent data leakage.
    """
    Age: int = Field(
        ...,
        ge=10,
        le=100,
        description="Student age in years (dataset range: 18 - 24)",
        examples=[21]
    )
    Gender: str = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Gender of the student (e.g., 'Male', 'Female')",
        examples=["Female"]
    )
    Country: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Country of residence (e.g., 'USA', 'India', 'UK', 'Other')",
        examples=["USA"]
    )
    Academic_Level: str = Field(
        ...,
        description="Current academic level: 'Undergraduate', 'Graduate', or 'High School'",
        examples=["Undergraduate"]
    )
    Most_Used_Platform: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Primary social media platform used (e.g., 'Instagram', 'TikTok', 'YouTube')",
        examples=["Instagram"]
    )
    Purpose_Of_Use: str = Field(
        ...,
        description="Primary purpose of social media use: 'Entertainment', 'Education', 'Networking', 'News'",
        examples=["Entertainment"]
    )
    Avg_Daily_Usage_Hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Average daily social media usage in hours",
        examples=[4.5]
    )
    Daily_Unlocks: int = Field(
        ...,
        ge=0,
        le=1000,
        description="Estimated daily smartphone screen unlocks",
        examples=[160]
    )
    Study_Hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Average daily study hours",
        examples=[3.5]
    )
    Physical_Activity_Hours: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Daily physical activity in hours (non-negative)",
        examples=[1.5]
    )
    Sleep_Hours_Per_Night: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Average sleep duration per night in hours",
        examples=[7.0]
    )
    Stress_Level: str = Field(
        ...,
        description="Perceived stress level: 'Low', 'Medium', 'High', or 'Very High'",
        examples=["Medium"]
    )

    @field_validator("Stress_Level")
    @classmethod
    def validate_stress_level(cls, value: str) -> str:
        """Validate and standardize stress level."""
        cleaned = value.strip()
        # Case-insensitive mapping
        for valid in VALID_STRESS_LEVELS:
            if cleaned.lower() == valid.lower():
                return valid
        raise ValueError(
            f"Invalid Stress_Level '{value}'. Allowed values are: {sorted(VALID_STRESS_LEVELS)}"
        )

    @field_validator("Academic_Level")
    @classmethod
    def validate_academic_level(cls, value: str) -> str:
        """Validate and standardize academic level."""
        cleaned = value.strip()
        for valid in VALID_ACADEMIC_LEVELS:
            if cleaned.lower() == valid.lower():
                return valid
        # Return cleaned as string to allow flexible categories if passed
        return cleaned.title()

    @field_validator("Purpose_Of_Use")
    @classmethod
    def validate_purpose(cls, value: str) -> str:
        """Validate and standardize purpose of use."""
        cleaned = value.strip()
        for valid in VALID_PURPOSES:
            if cleaned.lower() == valid.lower():
                return valid
        return cleaned.title()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 21,
                "Gender": "Female",
                "Country": "USA",
                "Academic_Level": "Undergraduate",
                "Most_Used_Platform": "Instagram",
                "Purpose_Of_Use": "Entertainment",
                "Avg_Daily_Usage_Hours": 4.5,
                "Daily_Unlocks": 150,
                "Study_Hours": 3.5,
                "Physical_Activity_Hours": 1.5,
                "Sleep_Hours_Per_Night": 7.0,
                "Stress_Level": "Medium"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output schema returned by the /predict endpoint."""
    success: bool = Field(True, description="Request success indicator")
    predicted_mental_health_score: float = Field(
        ...,
        description="Predicted Mental Health Score on a 1 - 10 scale (rounded to 2 decimals)"
    )
    raw_prediction: float = Field(
        ...,
        description="Unrounded raw regression score from the model pipeline"
    )
    interpretation: str = Field(
        ...,
        description="Qualitative mental health assessment category"
    )
    risk_level: str = Field(
        ...,
        description="Risk category: 'Low Risk', 'Moderate Risk', 'Elevated Risk', or 'High Risk'"
    )
    model_name: str = Field(
        ...,
        description="Name of the underlying trained model pipeline"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of prediction"
    )


class HealthResponse(BaseModel):
    """Output schema returned by the /health endpoint."""
    status: str = Field("healthy", description="API health status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded in memory")
    model_type: str = Field(..., description="Class name of the trained regressor")
    model_file_path: str = Field(..., description="Path to the loaded .pkl file")
    features_count: int = Field(..., description="Number of expected input features")
    expected_features: List[str] = Field(..., description="List of expected feature names")
    api_version: str = Field("1.0.0", description="Backend API version")


class WelcomeResponse(BaseModel):
    """Output schema returned by the root GET / endpoint."""
    message: str
    project: str
    version: str
    docs_url: str
    health_url: str
    predict_url: str


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    success: bool = False
    error: str
    details: Optional[Any] = None

