"""
Main entry point for the FastAPI application
"""

from src.api.app import create_app

# Create the FastAPI app using the application factory
app = create_app()

# For development with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
