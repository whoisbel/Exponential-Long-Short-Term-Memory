"""
API route handlers for stock prediction endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
)
from src.api.services import PredictionService, ModelHealthService
from src.config.settings import API_VERSION

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status"""
    health_data = ModelHealthService.check_model_health()

    return HealthResponse(
        status=health_data["status"],
        version=API_VERSION,
        models_loaded=health_data["models_loaded"],
    )


@router.get("/predict/next-month", response_model=PredictionResponse)
async def predict_next_month(
    seq_length: Optional[int] = Query(
        None, description="Custom sequence length for prediction"
    )
):
    """Predict stock prices for the next few days/months"""
    result = PredictionService.predict_future_months(seq_length=seq_length)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return PredictionResponse(**result)


@router.get("/predict/dataset", response_model=PredictionResponse)
async def predict_with_dataset():
    """Predict using historical dataset for validation"""
    result = PredictionService.predict_with_historical_dataset()

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return PredictionResponse(**result)


@router.get("/predict/last-week", response_model=PredictionResponse)
async def predict_last_week(
    seq_length: Optional[int] = Query(
        None, description="Custom sequence length for prediction"
    )
):
    """Predict last week's data for validation against actual values"""
    result = PredictionService.predict_last_week_data(seq_length=seq_length)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return PredictionResponse(**result)


@router.post("/predict/custom", response_model=PredictionResponse)
async def predict_custom(request: PredictionRequest):
    """Make predictions with custom input data (placeholder for future implementation)"""
    # This endpoint is reserved for future custom prediction functionality
    # where users can provide their own historical data
    raise HTTPException(
        status_code=501, detail="Custom prediction endpoint not yet implemented"
    )
