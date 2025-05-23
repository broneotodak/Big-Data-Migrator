"""
FastAPI routes definition for Big Data Migrator
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import List, Optional, Dict, Any
from app.models.requests import ProcessingRequest
from app.models.responses import ProcessingResponse, MemoryStatus
from app.models.llm_models import (
    ConversationRequest, MessageRequest, GuidanceRequest, 
    SchemaOptimizationRequest, ConversationResponse, 
    MessageResponse, GuidanceResponse, SchemaOptimizationResponse
)
from app.processors.processing_orchestrator import ProcessingOrchestrator
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer
from app.llm.conversation_system import LLMConversationSystem
from app.llm.online_llm_fallback import OnlineLLMConfig

# Initialize FastAPI app
app = FastAPI(
    title="Big Data Migrator API",
    description="API for processing and migrating large data files",
    version="0.1.0"
)

# Initialize components
memory_monitor = MemoryMonitor()
resource_optimizer = ResourceOptimizer(memory_monitor)
orchestrator = ProcessingOrchestrator(memory_monitor=memory_monitor)
logger = logging.getLogger(__name__)

# Initialize LLM conversation system
ENABLE_ONLINE_FALLBACK = os.getenv("ENABLE_ONLINE_FALLBACK", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Configure online LLM fallback if enabled
online_llm_config = None
if ENABLE_ONLINE_FALLBACK and OPENAI_API_KEY:
    online_llm_config = OnlineLLMConfig(
        api_key=OPENAI_API_KEY,
        model=os.getenv("ONLINE_LLM_MODEL", "gpt-4o")
    )

# Initialize LLM conversation system
llm_conversation_system = LLMConversationSystem(
    local_llm_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
    local_llm_model=os.getenv("LOCAL_LLM_MODEL", "CodeLlama-34B-Instruct"),
    memory_monitor=memory_monitor,
    resource_optimizer=resource_optimizer,
    online_llm_config=online_llm_config,
    enable_online_fallback=ENABLE_ONLINE_FALLBACK
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy"}

@app.get("/memory-status", response_model=MemoryStatus)
async def get_memory_status():
    """Get current memory usage status"""
    memory_stats = memory_monitor.get_current_usage()
    return memory_stats

@app.post("/upload-file", response_model=ProcessingResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for processing
    """
    try:
        # Check file size against available memory
        content = await file.read()
        file_size = len(content)
        
        # Check if we can safely process this file
        max_safe_size = memory_monitor.get_max_safe_file_size()
        if file_size > max_safe_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large to process safely. Maximum size: {max_safe_size/1024/1024:.2f} MB"
            )
            
        # Process the file
        result = await orchestrator.process_file(file.filename, content)
        
        return result
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessingResponse)
async def process_data(request: ProcessingRequest):
    """
    Process data with specified options
    """
    try:
        result = await orchestrator.process_request(request)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# LLM conversation endpoints
@app.post("/llm/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest):
    """
    Create a new LLM conversation
    """
    try:
        conversation_id = llm_conversation_system.create_conversation(
            title=request.title,
            data_files=request.data_files
        )
        
        # Get conversation details
        conversation = llm_conversation_system.conversation_manager.get_conversation(conversation_id)
        
        return ConversationResponse(
            conversation_id=conversation_id,
            title=conversation.title,
            messages=[m.__dict__ for m in conversation.messages],
            metadata=conversation.metadata
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """
    Get conversation details
    """
    try:
        conversation = llm_conversation_system.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        return ConversationResponse(
            conversation_id=conversation_id,
            title=conversation.title,
            messages=[m.__dict__ for m in conversation.messages],
            metadata=conversation.metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(conversation_id: str, request: MessageRequest, background_tasks: BackgroundTasks):
    """
    Add a message to a conversation and get a response
    """
    try:
        # Validate conversation exists
        if not llm_conversation_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Add message and get response
        response = llm_conversation_system.add_message(
            message=request.message,
            conversation_id=conversation_id
        )
        
        return MessageResponse(
            response=response["response"],
            conversation_id=conversation_id,
            guidance=response.get("guidance"),
            processing_time_ms=response.get("processing_time_ms")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/conversations/{conversation_id}/guidance", response_model=GuidanceResponse)
async def generate_guidance(conversation_id: str):
    """
    Generate guidance for a conversation
    """
    try:
        # Validate conversation exists
        if not llm_conversation_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Generate guidance
        guidance = llm_conversation_system.generate_guidance(conversation_id)
        
        return GuidanceResponse(
            suggestions=guidance.get("suggestions", []),
            questions=guidance.get("questions", []),
            improvements=guidance.get("improvements", []),
            metadata=guidance.get("metadata", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating guidance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/conversations/{conversation_id}/optimize-schema", response_model=SchemaOptimizationResponse)
async def optimize_schema(conversation_id: str, background_tasks: BackgroundTasks):
    """
    Optimize schema using online LLM fallback
    """
    try:
        # Validate conversation exists
        if not llm_conversation_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if online fallback is enabled
        if not llm_conversation_system.enable_online_fallback:
            raise HTTPException(
                status_code=400, 
                detail="Online LLM fallback is not enabled. Configure ENABLE_ONLINE_FALLBACK=true and set OPENAI_API_KEY."
            )
        
        # Optimize schema in background
        background_tasks.add_task(
            llm_conversation_system.optimize_schema_with_fallback,
            conversation_id
        )
        
        return SchemaOptimizationResponse(
            optimized_schema={},
            recommendations=["Schema optimization started in background. Check back later for results."],
            metadata={"status": "processing"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))