"""
Database operations for managing sessions, files, and data.
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from app.database.supabase_manager import SupabaseManager
from app.database.models import (
    UserSession,
    FileMetadata,
    ProcessingState,
    ConversationHistory,
    DataInsights,
    ResourceUsage
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class DatabaseOperations:
    """
    Database operations for managing sessions, files, and data.
    """
    
    def __init__(self, supabase_manager: Optional[SupabaseManager] = None):
        """Initialize database operations."""
        self.db = supabase_manager or SupabaseManager()
        
    # Session Management
    def create_session(self, user_id: str, session_data: Optional[Dict[str, Any]] = None) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user_id,
            session_data=session_data or {}
        )
        
        try:
            result = self.db.execute_query(
                "create_session",
                params=session.dict()
            )
            return UserSession(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
            
    def get_session(self, session_id: UUID) -> Optional[UserSession]:
        """Get a user session by ID."""
        try:
            result = self.db.execute_query(
                "get_session",
                params={"session_id": str(session_id)}
            )
            return UserSession(**result.data[0]) if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to get session: {str(e)}")
            raise
            
    def update_session(self, session_id: UUID, session_data: Dict[str, Any]) -> UserSession:
        """Update a user session."""
        try:
            result = self.db.execute_query(
                "update_session",
                params={
                    "session_id": str(session_id),
                    "session_data": session_data,
                    "last_active": datetime.utcnow().isoformat()
                }
            )
            return UserSession(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to update session: {str(e)}")
            raise
            
    def delete_session(self, session_id: UUID) -> bool:
        """Delete a user session."""
        try:
            self.db.execute_query(
                "delete_session",
                params={"session_id": str(session_id)}
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            raise
            
    # File Operations
    def save_file_metadata(self, file_metadata: FileMetadata) -> FileMetadata:
        """Save file metadata."""
        try:
            result = self.db.execute_query(
                "save_file_metadata",
                params=file_metadata.dict()
            )
            return FileMetadata(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to save file metadata: {str(e)}")
            raise
            
    def update_processing_status(self, 
                               file_id: UUID, 
                               status: str,
                               error_log: Optional[List[str]] = None) -> FileMetadata:
        """Update file processing status."""
        try:
            result = self.db.execute_query(
                "update_processing_status",
                params={
                    "file_id": str(file_id),
                    "status": status,
                    "error_log": error_log or [],
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            return FileMetadata(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to update processing status: {str(e)}")
            raise
            
    def get_file_history(self, file_id: UUID) -> List[ProcessingState]:
        """Get file processing history."""
        try:
            result = self.db.execute_query(
                "get_file_history",
                params={"file_id": str(file_id)}
            )
            return [ProcessingState(**state) for state in result.data]
            
        except Exception as e:
            logger.error(f"Failed to get file history: {str(e)}")
            raise
            
    # Conversation Management
    def save_conversation(self, conversation: ConversationHistory) -> ConversationHistory:
        """Save conversation history."""
        try:
            result = self.db.execute_query(
                "save_conversation",
                params=conversation.dict()
            )
            return ConversationHistory(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            raise
            
    def get_conversation_history(self, 
                               session_id: UUID,
                               limit: int = 100,
                               before_timestamp: Optional[datetime] = None) -> List[ConversationHistory]:
        """Get conversation history."""
        try:
            params = {
                "session_id": str(session_id),
                "limit": limit
            }
            if before_timestamp:
                params["before_timestamp"] = before_timestamp.isoformat()
                
            result = self.db.execute_query(
                "get_conversation_history",
                params=params
            )
            return [ConversationHistory(**conv) for conv in result.data]
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {str(e)}")
            raise
            
    # Data Insights
    def save_data_insights(self, insights: DataInsights) -> DataInsights:
        """Save data insights."""
        try:
            result = self.db.execute_query(
                "save_data_insights",
                params=insights.dict()
            )
            return DataInsights(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to save data insights: {str(e)}")
            raise
            
    def get_data_insights(self, file_id: UUID) -> Optional[DataInsights]:
        """Get data insights for a file."""
        try:
            result = self.db.execute_query(
                "get_data_insights",
                params={"file_id": str(file_id)}
            )
            return DataInsights(**result.data[0]) if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to get data insights: {str(e)}")
            raise
            
    # Resource Usage Tracking
    def save_resource_usage(self, usage: ResourceUsage) -> ResourceUsage:
        """Save resource usage data."""
        try:
            result = self.db.execute_query(
                "save_resource_usage",
                params=usage.dict()
            )
            return ResourceUsage(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to save resource usage: {str(e)}")
            raise
            
    def get_resource_usage_history(self,
                                 session_id: UUID,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> List[ResourceUsage]:
        """Get resource usage history."""
        try:
            params = {"session_id": str(session_id)}
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()
                
            result = self.db.execute_query(
                "get_resource_usage_history",
                params=params
            )
            return [ResourceUsage(**usage) for usage in result.data]
            
        except Exception as e:
            logger.error(f"Failed to get resource usage history: {str(e)}")
            raise
            
    # Batch Operations
    def batch_save_conversations(self, conversations: List[ConversationHistory]) -> List[ConversationHistory]:
        """Save multiple conversations in a batch."""
        try:
            operations = [
                {
                    "query": "save_conversation",
                    "params": conv.dict()
                }
                for conv in conversations
            ]
            
            results = self.db.execute_transaction(operations)
            return [ConversationHistory(**result.data[0]) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to batch save conversations: {str(e)}")
            raise
            
    def batch_save_resource_usage(self, usage_data: List[ResourceUsage]) -> List[ResourceUsage]:
        """Save multiple resource usage records in a batch."""
        try:
            operations = [
                {
                    "query": "save_resource_usage",
                    "params": usage.dict()
                }
                for usage in usage_data
            ]
            
            results = self.db.execute_transaction(operations)
            return [ResourceUsage(**result.data[0]) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to batch save resource usage: {str(e)}")
            raise
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            operations = [
                {
                    "query": "cleanup_old_sessions",
                    "params": {"cutoff_date": cutoff_date.isoformat()}
                },
                {
                    "query": "cleanup_old_conversations",
                    "params": {"cutoff_date": cutoff_date.isoformat()}
                },
                {
                    "query": "cleanup_old_resource_usage",
                    "params": {"cutoff_date": cutoff_date.isoformat()}
                }
            ]
            
            results = self.db.execute_transaction(operations)
            
            return {
                "sessions_deleted": results[0].data[0]["count"],
                "conversations_deleted": results[1].data[0]["count"],
                "resource_usage_deleted": results[2].data[0]["count"]
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            raise
            
    def close(self) -> None:
        """Close database connections."""
        self.db.close()
        
    def __del__(self):
        """Cleanup when the object is deleted."""
        self.close() 