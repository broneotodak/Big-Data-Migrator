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

# Load environment variables
load_dotenv(os.path.join("config", ".env"))

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Big Data Migrator API server")
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 4))
    
    logger.info(f"Server running at http://{host}:{port}")
    
    uvicorn.run(
        "app.api.routes:app",
        host=host,
        port=port,
        workers=workers,
        reload=True,
    )