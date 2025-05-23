"""
Database configuration settings.
"""
import os
from typing import Dict, Any

# Database connection settings
DB_CONFIG = {
    "url": os.getenv("SUPABASE_URL", ""),
    "key": os.getenv("SUPABASE_KEY", ""),
    "max_connections": int(os.getenv("DB_MAX_CONNECTIONS", "10")),
    "connection_timeout": int(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
    "retry_attempts": int(os.getenv("DB_RETRY_ATTEMPTS", "3")),
    "retry_delay": int(os.getenv("DB_RETRY_DELAY", "1")),
    "max_retry_delay": int(os.getenv("DB_MAX_RETRY_DELAY", "10")),
    "circuit_breaker_max_failures": int(os.getenv("DB_CIRCUIT_BREAKER_MAX_FAILURES", "3")),
    "circuit_breaker_reset_timeout": int(os.getenv("DB_CIRCUIT_BREAKER_RESET_TIMEOUT", "60")),
}

# Table settings
TABLE_CONFIG = {
    "user_sessions": {
        "max_session_data_size": 10000,  # bytes
        "session_timeout": 3600,  # seconds
    },
    "file_metadata": {
        "max_metadata_size": 50000,  # bytes
        "valid_statuses": ["pending", "processing", "completed", "failed"],
    },
    "processing_states": {
        "max_error_log_size": 1000,  # entries
        "max_state_data_size": 50000,  # bytes
    },
    "conversation_history": {
        "max_content_size": 100000,  # bytes
        "max_context_data_size": 50000,  # bytes
        "valid_message_types": ["user", "assistant", "system", "error"],
    },
    "data_insights": {
        "max_column_analysis_size": 100000,  # bytes
        "max_relationships_size": 50000,  # bytes
        "max_recommendations": 100,  # entries
    },
    "resource_usage": {
        "max_history_days": 30,
        "cleanup_batch_size": 1000,
    },
}

# Index settings
INDEX_CONFIG = {
    "user_sessions": ["user_id", "last_active"],
    "file_metadata": ["session_id", "processing_status"],
    "processing_states": ["file_id"],
    "conversation_history": ["session_id", "timestamp"],
    "data_insights": ["file_id"],
    "resource_usage": ["session_id", "timestamp"],
}

# Cleanup settings
CLEANUP_CONFIG = {
    "sessions": {
        "days_to_keep": int(os.getenv("DB_SESSIONS_DAYS_TO_KEEP", "30")),
        "batch_size": int(os.getenv("DB_SESSIONS_CLEANUP_BATCH_SIZE", "1000")),
    },
    "conversations": {
        "days_to_keep": int(os.getenv("DB_CONVERSATIONS_DAYS_TO_KEEP", "30")),
        "batch_size": int(os.getenv("DB_CONVERSATIONS_CLEANUP_BATCH_SIZE", "1000")),
    },
    "resource_usage": {
        "days_to_keep": int(os.getenv("DB_RESOURCE_USAGE_DAYS_TO_KEEP", "30")),
        "batch_size": int(os.getenv("DB_RESOURCE_USAGE_CLEANUP_BATCH_SIZE", "1000")),
    },
}

def get_db_config() -> Dict[str, Any]:
    """Get database configuration."""
    return DB_CONFIG

def get_table_config() -> Dict[str, Any]:
    """Get table configuration."""
    return TABLE_CONFIG

def get_index_config() -> Dict[str, Any]:
    """Get index configuration."""
    return INDEX_CONFIG

def get_cleanup_config() -> Dict[str, Any]:
    """Get cleanup configuration."""
    return CLEANUP_CONFIG 