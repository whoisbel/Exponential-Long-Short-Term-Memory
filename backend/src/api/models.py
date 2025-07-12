"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for custom prediction"""

    past_values: List[float] = Field(..., description="Historical stock prices")
    seq_length: Optional[int] = Field(
        None, description="Custom sequence length override"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predicted_values: List[Dict[str, Any]]
    base_data: Optional[List[List[Any]]] = None
    last_week_data: Optional[List[Dict[str, Any]]] = None
    prediction_period: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    models_loaded: bool
