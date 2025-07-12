"""
Service layer for stock prediction operations
"""

import logging
from typing import Dict, Any, Optional
from src.scripts.predict import (
    predict_next_months,
    predict_with_dataset,
    predict_last_week,
)
from src.config.settings import DEFAULT_PRED_DAYS
from src.api.models import PredictionResponse, ErrorResponse

logger = logging.getLogger(__name__)


class PredictionService:
    """Service class for handling prediction operations"""

    @staticmethod
    def predict_future_months(seq_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict future stock prices for the next few days/months

        Args:
            seq_length: Optional custom sequence length

        Returns:
            Dictionary containing prediction results
        """
        try:
            result = predict_next_months(seq_length=seq_length)

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Prediction error: {result['error']}")
                return {"error": result["error"]}

            # Add metadata
            result["metadata"] = {
                "prediction_type": "future_months",
                "sequence_length": seq_length,
                "prediction_days": DEFAULT_PRED_DAYS,
            }

            return result

        except Exception as e:
            logger.error(f"Error in predict_future_months: {str(e)}")
            return {"error": f"Prediction service error: {str(e)}"}

    @staticmethod
    def predict_with_historical_dataset() -> Dict[str, Any]:
        """
        Predict using historical dataset for validation

        Returns:
            Dictionary containing prediction results
        """
        try:
            result = predict_with_dataset()

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Dataset prediction error: {result['error']}")
                return {"error": result["error"]}

            # Add metadata
            result["metadata"] = {
                "prediction_type": "historical_dataset",
                "data_source": "local_csv",
            }

            return result

        except Exception as e:
            logger.error(f"Error in predict_with_historical_dataset: {str(e)}")
            return {"error": f"Dataset prediction service error: {str(e)}"}

    @staticmethod
    def predict_last_week_data(seq_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict last week's data for validation

        Args:
            seq_length: Optional custom sequence length

        Returns:
            Dictionary containing prediction results
        """
        try:
            result = predict_last_week(seq_length=seq_length)

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Last week prediction error: {result['error']}")
                return {"error": result["error"]}

            # Add metadata
            result["metadata"] = {
                "prediction_type": "last_week_validation",
                "sequence_length": seq_length,
            }

            return result

        except Exception as e:
            logger.error(f"Error in predict_last_week_data: {str(e)}")
            return {"error": f"Last week prediction service error: {str(e)}"}


class ModelHealthService:
    """Service for checking model health and status"""

    @staticmethod
    def check_model_health() -> Dict[str, Any]:
        """
        Check if models are loaded and ready

        Returns:
            Dictionary containing health status
        """
        try:
            # This would check if models are properly loaded
            # For now, we'll do a basic check
            from src.scripts.predict import elu_lstm, tanh_lstm

            models_loaded = elu_lstm is not None and tanh_lstm is not None

            return {
                "status": "healthy" if models_loaded else "unhealthy",
                "models_loaded": models_loaded,
                "details": (
                    "All models loaded successfully"
                    if models_loaded
                    else "Models not loaded"
                ),
            }

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "status": "unhealthy",
                "models_loaded": False,
                "details": f"Health check failed: {str(e)}",
            }
