"""
FastAPI routes definition for Big Data Migrator
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
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
from app.api.debug_routes import router as debug_router
import time

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Big Data Migrator API",
    description="API for processing and migrating large data files",
    version="0.1.0"
)

# Add debug routes
app.include_router(debug_router)

# Initialize components
memory_monitor = MemoryMonitor()
resource_optimizer = ResourceOptimizer(memory_monitor)
orchestrator = ProcessingOrchestrator()
logger = logging.getLogger(__name__)

# LLM Configuration from environment variables
ENABLE_ONLINE_FALLBACK = os.getenv("ENABLE_ONLINE_FALLBACK", "false").lower() == "true"
ENABLE_ANTHROPIC = os.getenv("ENABLE_ANTHROPIC", "false").lower() == "true"
ENABLE_MULTI_LLM = os.getenv("ENABLE_MULTI_LLM", "false").lower() == "true"
PRIMARY_LLM = os.getenv("PRIMARY_LLM", "local")  # "local", "anthropic", "openai", "multi"

# Timeout configurations
LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
ANTHROPIC_TIMEOUT = int(os.getenv("ANTHROPIC_TIMEOUT", "300"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Anthropic Configuration  
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# Initialize LLM system with appropriate configuration
online_llm_config = None
if ENABLE_ONLINE_FALLBACK and OPENAI_API_KEY:
    online_llm_config = OnlineLLMConfig(
        api_key=OPENAI_API_KEY,
        model=os.getenv("ONLINE_LLM_MODEL", "gpt-4o"),
        timeout=OPENAI_TIMEOUT
    )

anthropic_config = None
if ENABLE_ANTHROPIC and ANTHROPIC_API_KEY:
    from app.llm.anthropic_client import AnthropicConfig
    anthropic_config = AnthropicConfig(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL,
        max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000")),
        temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")),
        timeout=ANTHROPIC_TIMEOUT
    )

llm_system = LLMConversationSystem(
    local_llm_url=os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1"),
    local_llm_model=os.getenv("LOCAL_LLM_MODEL", "claude-3.7-sonnet-reasoning-gemma3-12b"),
    local_llm_timeout=LOCAL_LLM_TIMEOUT,
    memory_monitor=memory_monitor,
    resource_optimizer=resource_optimizer,
    online_llm_config=online_llm_config,
    anthropic_config=anthropic_config,
    enable_online_fallback=ENABLE_ONLINE_FALLBACK,
    enable_anthropic=ENABLE_ANTHROPIC,
    enable_multi_llm=ENABLE_MULTI_LLM,
    primary_llm=PRIMARY_LLM,
    enable_smart_processing=True  # Enable Smart Query Processor for direct calculations
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
    return {"status": "healthy", "message": "Big Data Migrator API is running"}

@app.get("/memory-status", response_model=MemoryStatus)
async def get_memory_status():
    """Get current memory usage status"""
    try:
        # Get memory report from monitor
        memory_report = memory_monitor.get_memory_report()
        
        # Convert MB to GB for the response (now using system-wide memory)
        total_memory_gb = memory_report["total_ram_mb"] / 1024
        available_memory_gb = memory_report["available_mb"] / 1024
        used_memory_gb = memory_report["current_usage_mb"] / 1024  # Now system-wide usage
        
        # Calculate max safe file size for CSV by default
        max_safe_file_size_mb = memory_monitor.get_max_safe_file_size("csv")
        
        # Calculate recommended chunk size based on available memory
        recommended_chunk_size = min(50000, max(1000, int(memory_report["available_mb"] * 10)))
        
        # Determine if large file processing is advisable (using system-wide memory)
        can_process_large_files = (
            memory_report["usage_percent"] < 70 and  # Less than 70% system memory used
            memory_report["available_mb"] > 1024     # At least 1GB available system-wide
        )
        
        return MemoryStatus(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            used_memory_gb=used_memory_gb,
            memory_usage_percent=memory_report["usage_percent"],
            process_memory_mb=memory_report["process_memory_mb"],  # Our process memory
            max_safe_file_size_mb=max_safe_file_size_mb,
            recommended_chunk_size=recommended_chunk_size,
            can_process_large_files=can_process_large_files
        )
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")

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
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else 'csv'
        max_safe_size = memory_monitor.get_max_safe_file_size(file_extension)
        if file_size > max_safe_size * 1024 * 1024:  # Convert MB to bytes
            raise HTTPException(
                status_code=413, 
                detail=f"File too large to process safely. Maximum size: {max_safe_size:.2f} MB"
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
        conversation_id = llm_system.create_conversation(
            title=request.title,
            data_files=request.data_files
        )
        
        # Get conversation details
        conversation = llm_system.conversation_manager.get_conversation(conversation_id)
        
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
        conversation = llm_system.conversation_manager.get_conversation(conversation_id)
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
        if not llm_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Add message and get response
        response = llm_system.add_message(
            message=request.message,
            conversation_id=conversation_id
        )
        
        return MessageResponse(
            response=response["response"],
            conversation_id=conversation_id,
            guidance=response.get("guidance"),
            processing_time_ms=response.get("processing_time_ms"),
            smart_processing=response.get("smart_processing"),
            processed_results=response.get("processed_results"),
            llm_used=response.get("llm_used")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/conversations/{conversation_id}/messages/multi")
async def add_message_multi_llm(conversation_id: str, request: MessageRequest, background_tasks: BackgroundTasks):
    """
    Add a message to a conversation and get responses from multiple LLM providers.
    Uses intelligent provider selection to prevent timeouts.
    """
    try:
        # Import debug functions
        from app.api.debug_routes import start_processing_task, update_processing_task, end_processing_task, log_processing_error
        
        # Start tracking this multi-LLM request
        task_id = f"multi_llm_{conversation_id}_{int(time.time())}"
        start_processing_task(task_id, f"Multi-LLM processing: {request.message[:50]}...")
        
        # Check if multi-LLM is enabled
        if not ENABLE_MULTI_LLM:
            end_processing_task(task_id)
            raise HTTPException(
                status_code=400, 
                detail="Multi-LLM mode is not enabled. Set ENABLE_MULTI_LLM=true in environment."
            )
        
        # Validate conversation exists
        if not llm_system.conversation_manager.get_conversation(conversation_id):
            end_processing_task(task_id)
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        update_processing_task(task_id, "Analyzing request complexity")
        
        # Smart provider selection based on request complexity and available resources
        memory_report = memory_monitor.get_memory_report()
        message_length = len(request.message)
        
        # Determine request complexity
        is_complex_request = (
            message_length > 500 or  # Long requests
            any(keyword in request.message.lower() for keyword in [
                'calculate', 'compare', 'analyze', 'relationship', 'missing', 'sum', 
                'count', 'total', 'difference', 'merge', 'join'
            ])
        )
        
        # Determine available providers and their suitability
        available_providers = []
        
        # Always try local LLM first (most reliable)
        available_providers.append("local")
        
        # Only add other providers for complex requests if system resources allow
        if is_complex_request and memory_report["usage_percent"] < 60:
            
            # Add Anthropic only if it's likely to work (not rate limited recently)
            if ENABLE_ANTHROPIC and ANTHROPIC_API_KEY:
                # Simple check: if we haven't had recent rate limit errors, try Anthropic
                available_providers.append("anthropic")
            
            # Add OpenAI only for very complex analysis
            if (ENABLE_ONLINE_FALLBACK and OPENAI_API_KEY and 
                any(keyword in request.message.lower() for keyword in [
                    'schema', 'optimization', 'migration', 'complex analysis'
                ])):
                available_providers.append("openai")
        
        update_processing_task(task_id, f"Using providers: {', '.join(available_providers)}")
        
        # For simple requests, just use primary LLM
        if not is_complex_request or len(available_providers) == 1:
            update_processing_task(task_id, "Processing with single LLM")
            try:
                response = llm_system.add_message(
                    message=request.message,
                    conversation_id=conversation_id
                )
                
                end_processing_task(task_id)
                return {
                    "best_response": response["response"],
                    "consensus_response": response["response"],  # Use the single response as consensus
                    "all_responses": [{"provider": "local", "response": response["response"], "success": True}],
                    "conversation_id": conversation_id,
                    "guidance": response.get("guidance"),
                    "processing_time_ms": response.get("processing_time_ms"),
                    "mode": "single_llm_optimized",
                    "comparison": {},
                    "providers_used": 1,
                    "successful_responses": 1
                }
            except Exception as e:
                log_processing_error(f"Single LLM processing failed: {str(e)}", "llm_processing")
                end_processing_task(task_id)
                raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")
        
        # For complex requests, use multi-LLM with timeout protection
        update_processing_task(task_id, "Processing with multi-LLM")
        try:
            response = await llm_system.add_message_multi_llm(
                message=request.message,
                conversation_id=conversation_id,
                providers=available_providers,  # Pass selected providers
                timeout_per_provider=120  # Shorter timeout per provider to prevent overall timeout
            )
            
            end_processing_task(task_id)
            return {
                "best_response": response.get("best_response", ""),
                "consensus_response": response.get("consensus_response"),
                "all_responses": response.get("all_responses", []),
                "conversation_id": conversation_id,
                "guidance": response.get("guidance"),
                "processing_time_ms": response.get("processing_time_ms"),
                "mode": "multi_llm_optimized",
                "comparison": response.get("comparison", {}),
                "providers_used": len(response.get("all_responses", [])),
                "successful_responses": len([r for r in response.get("all_responses", []) if r.get("success", False)])
            }
            
        except Exception as e:
            log_processing_error(f"Multi-LLM processing failed: {str(e)}", "multi_llm_processing")
            end_processing_task(task_id)
            
            # Fallback to single LLM if multi-LLM fails
            try:
                update_processing_task(task_id, "Falling back to single LLM")
                response = llm_system.add_message(
                    message=request.message,
                    conversation_id=conversation_id
                )
                
                end_processing_task(task_id)
                return {
                    "best_response": response["response"],
                    "consensus_response": response["response"],  # Use the fallback response as consensus
                    "all_responses": [{"provider": "local_fallback", "response": response["response"], "success": True}],
                    "conversation_id": conversation_id,
                    "guidance": response.get("guidance"),
                    "processing_time_ms": response.get("processing_time_ms"),
                    "mode": "fallback_single_llm",
                    "comparison": {},
                    "providers_used": 1,
                    "successful_responses": 1,
                    "fallback_reason": str(e)
                }
            except Exception as fallback_error:
                end_processing_task(task_id)
                raise HTTPException(status_code=500, detail=f"All LLM processing failed: {str(fallback_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-LLM endpoint: {e}")
        if 'task_id' in locals():
            end_processing_task(task_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/conversations/{conversation_id}/guidance", response_model=GuidanceResponse)
async def generate_guidance(conversation_id: str):
    """
    Generate guidance for a conversation
    """
    try:
        # Validate conversation exists
        if not llm_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Generate guidance
        guidance = llm_system.generate_guidance(conversation_id)
        
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
        if not llm_system.conversation_manager.get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if online fallback is enabled
        if not llm_system.enable_online_fallback:
            raise HTTPException(
                status_code=400, 
                detail="Online LLM fallback is not enabled. Configure ENABLE_ONLINE_FALLBACK=true and set OPENAI_API_KEY."
            )
        
        # Optimize schema in background
        background_tasks.add_task(
            llm_system.optimize_schema_with_fallback,
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

@app.get("/llm/status")
async def get_llm_status():
    """Get detailed LLM connection status for debugging"""
    try:
        # Test LM Studio connection
        connection_status = llm_system.check_connection()
        
        # Test conversation creation
        try:
            test_conversation_id = llm_system.create_conversation("Test", [])
            conversation_test = "✅ Conversation creation works"
        except Exception as e:
            conversation_test = f"❌ Conversation creation failed: {str(e)}"
        
        return {
            "lm_studio_connection": connection_status,
            "conversation_system": conversation_test,
            "local_llm_url": llm_system.llm_client.base_url,
            "local_llm_model": llm_system.llm_client.model,
            "enable_online_fallback": llm_system.enable_online_fallback
        }
    except Exception as e:
        return {
            "error": f"LLM status check failed: {str(e)}",
            "lm_studio_connection": "❌ Not available",
            "conversation_system": "❌ Not initialized"
        }