"""
FastAPI application factory and configuration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.api.routes import router
from src.config.settings import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    CORS_CREDENTIALS,
    CORS_METHODS,
    CORS_HEADERS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Application factory function to create and configure FastAPI app

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=CORS_CREDENTIALS,
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
    )

    # Include routers
    app.include_router(router)

    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {API_TITLE} v{API_VERSION}")
        logger.info("Loading prediction models...")
        # Models are loaded when the predict module is imported

    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down application...")

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        return {
            "message": f"Welcome to {API_TITLE}",
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app
