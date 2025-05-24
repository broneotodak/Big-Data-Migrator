"""
Logging configuration for Big Data Migrator
"""
import os
import logging
import logging.config
from pathlib import Path

def setup_logging(log_level=None):
    """
    Set up logging configuration
    
    Args:
        log_level: Optional override for the log level from environment
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "default",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 10,
                "encoding": "utf8",
            },
            "error_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "default",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 10,
                "encoding": "utf8",
            },
            "memory_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "default",
                "filename": "logs/memory.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {
                "level": log_level,
                "handlers": ["console", "file_handler", "error_file_handler"],
                "propagate": True,
            },
            "app.memory": {
                "level": log_level,
                "handlers": ["memory_file_handler", "console"],
                "propagate": False,
            },
        },
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log setup completed
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging setup completed with level {log_level}")
    
    return logger

def get_logger(name=None):
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name, defaults to caller's module name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)