"""
Pydantic models for database entities.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

class UserSession(BaseModel):
    """User session model."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    session_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('session_data')
    def validate_session_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session data size."""
        if len(str(v)) > 10000:  # 10KB limit
            raise ValueError("Session data too large")
        return v
        
class FileMetadata(BaseModel):
    """File metadata model."""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    filename: str
    file_type: str
    size: int  # in bytes
    processing_status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('processing_status')
    def validate_status(cls, v: str) -> str:
        """Validate processing status."""
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of {valid_statuses}")
        return v
        
class ProcessingState(BaseModel):
    """File processing state model."""
    id: UUID = Field(default_factory=uuid4)
    file_id: UUID
    current_step: str
    progress_percentage: float = 0.0
    error_log: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    state_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('progress_percentage')
    def validate_progress(cls, v: float) -> float:
        """Validate progress percentage."""
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v
        
class ConversationHistory(BaseModel):
    """Conversation history model."""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    message_type: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    llm_response: Optional[str] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('message_type')
    def validate_message_type(cls, v: str) -> str:
        """Validate message type."""
        valid_types = ["user", "assistant", "system", "error"]
        if v not in valid_types:
            raise ValueError(f"Invalid message type. Must be one of {valid_types}")
        return v
        
class DataInsights(BaseModel):
    """Data insights model."""
    id: UUID = Field(default_factory=uuid4)
    file_id: UUID
    column_analysis: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: float = 0.0
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('quality_score')
    def validate_quality_score(cls, v: float) -> float:
        """Validate quality score."""
        if not 0 <= v <= 100:
            raise ValueError("Quality score must be between 0 and 100")
        return v
        
class ResourceUsage(BaseModel):
    """Resource usage tracking model."""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    operation_type: str
    operation_id: Optional[UUID] = None
    
    @validator('cpu_percent', 'memory_percent', 'disk_usage_percent')
    def validate_percentages(cls, v: float) -> float:
        """Validate percentage values."""
        if not 0 <= v <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v 