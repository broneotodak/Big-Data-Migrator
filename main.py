"""
Big Data Migrator - Main Application Entry Point

This script initializes and runs the FastAPI backend service
"""
import os
import logging
import uvicorn
from dotenv import load_dotenv
from app.api.routes import app
from app.utils.logging_config import setup_logging

# Load environment variables from the correct location
load_dotenv(".env")  # Load from root directory, not config/.env

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Big Data Migrator API server")
    
    # Use localhost for better Windows compatibility
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Server configured for {host}:{port}")
    
    # Disable reload and workers for stability
    uvicorn.run(
        "app.api.routes:app",
        host=host,
        port=port,
        reload=False,  # Disabled for stability
        log_level="info"
    )