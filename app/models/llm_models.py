"""
Pydantic models for LLM conversation API
"""
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union

class ConversationRequest(BaseModel):
    """Request to create a new conversation"""
    title: str = "Data Conversation"
    data_files: Optional[List[str]] = None

class MessageRequest(BaseModel):
    """Request to add a message to a conversation"""
    message: str
    conversation_id: Optional[str] = None

class GuidanceRequest(BaseModel):
    """Request to generate guidance for a conversation"""
    conversation_id: Optional[str] = None

class SchemaOptimizationRequest(BaseModel):
    """Request to optimize schema using online fallback"""
    conversation_id: Optional[str] = None

class ConversationResponse(BaseModel):
    """Response containing conversation information"""
    conversation_id: str
    title: str
    messages: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class MessageResponse(BaseModel):
    """Response containing message and related information"""
    response: str
    conversation_id: str
    guidance: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    smart_processing: Optional[bool] = None
    processed_results: Optional[Dict[str, Any]] = None
    llm_used: Optional[str] = None

class GuidanceResponse(BaseModel):
    """Response containing guidance information"""
    suggestions: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class SchemaOptimizationResponse(BaseModel):
    """Response containing schema optimization results"""
    optimized_schema: Dict[str, Any] = {}
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}
