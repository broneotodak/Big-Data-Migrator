"""
Integration module for LLM conversation system components.

This module provides the main interface for the LLM conversation system,
integrating all components for a seamless data discussion experience.
"""
import os
import json
from typing import Dict, List, Optional, Any, Union, Generator

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer
from app.llm.lm_studio_client import LMStudioClient
from app.llm.conversation_manager import ConversationManager
from app.llm.data_context_builder import DataContextBuilder
from app.llm.user_guidance import UserGuidanceSystem
from app.llm.online_llm_fallback import OnlineLLMFallback, OnlineLLMConfig

logger = get_logger(__name__)

class LLMConversationSystem:
    """
    Main interface for the LLM conversation system.
    
    Integrates all components:
    - LMStudioClient for local LLM integration
    - ConversationManager for conversation history
    - DataContextBuilder for data analysis
    - UserGuidanceSystem for intelligent guidance
    - OnlineLLMFallback for complex tasks
    """
    
    def __init__(self,
                local_llm_url: str = "http://localhost:1234/v1",
                local_llm_model: str = "CodeLlama-34B-Instruct",
                memory_monitor: Optional[MemoryMonitor] = None,
                resource_optimizer: Optional[ResourceOptimizer] = None,
                online_llm_config: Optional[OnlineLLMConfig] = None,
                conversation_dir: str = "conversations",
                enable_online_fallback: bool = False):
        """
        Initialize the LLM conversation system.
        
        Args:
            local_llm_url: URL for local LLM
            local_llm_model: Model name for local LLM
            memory_monitor: Optional memory monitor instance
            resource_optimizer: Optional resource optimizer instance
            online_llm_config: Configuration for online LLM
            conversation_dir: Directory for conversation storage
            enable_online_fallback: Whether to enable online LLM fallback
        """
        # Initialize memory management
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer(self.memory_monitor)
        
        # Initialize LLM clients
        self.llm_client = LMStudioClient(
            base_url=local_llm_url,
            model=local_llm_model,
            memory_monitor=self.memory_monitor
        )
        
        # Initialize conversation management
        self.conversation_manager = ConversationManager(
            llm_client=self.llm_client,
            storage_dir=conversation_dir,
            memory_monitor=self.memory_monitor
        )
        
        # Initialize data context builder
        self.data_context_builder = DataContextBuilder(
            memory_monitor=self.memory_monitor,
            resource_optimizer=self.resource_optimizer
        )
        
        # Initialize user guidance system
        self.user_guidance = UserGuidanceSystem(
            llm_client=self.llm_client
        )
        
        # Initialize online LLM fallback (if enabled)
        self.enable_online_fallback = enable_online_fallback
        self.online_llm = None
        
        if enable_online_fallback and online_llm_config:
            self.online_llm = OnlineLLMFallback(
                config=online_llm_config,
                memory_monitor=self.memory_monitor
            )
        
        # Data context cache
        self._data_context_cache = {}
        self._active_data_context = None
        
        logger.info("LLMConversationSystem initialized")
    
    def create_conversation(self, title: str = "Data Conversation", data_files: List[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            data_files: List of data files to associate with conversation
            
        Returns:
            Conversation ID
        """
        conversation_id = self.conversation_manager.create_conversation(title, data_files)
        
        # Build data context if files provided
        if data_files:
            self.build_data_context(data_files, conversation_id)
        
        return conversation_id
    
    def add_message(self, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a user message and generate a response.
        
        Args:
            message: User message
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = self.memory_monitor.current_time_ms()
        
        # Add user message
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        self.conversation_manager.add_user_message(message, conv_id)
        
        # Generate response
        response = self.conversation_manager.generate_response(conv_id)
        
        # Generate guidance
        guidance = None
        if self._active_data_context:
            # Extract conversation history
            history = self._get_conversation_history(conv_id)
            
            # Generate guidance
            guidance = self.user_guidance.generate_guidance(
                data_contexts=self._active_data_context,
                conversation_history=history
            )
        
        # Calculate processing time
        processing_time_ms = self.memory_monitor.current_time_ms() - start_time
        
        return {
            "response": response,
            "conversation_id": conv_id,
            "guidance": guidance,
            "processing_time_ms": processing_time_ms
        }
    
    def build_data_context(self, data_files: List[str], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build context for data files.
        
        Args:
            data_files: List of data files
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Generated data context
        """
        # Build context
        data_context = self.data_context_builder.build_context_for_files(
            file_paths=data_files,
            include_stats=True,
            include_samples=True,
            include_relationships=True,
            include_quality=True
        )
        
        # Cache the context
        context_key = self._create_context_key(data_files)
        self._data_context_cache[context_key] = data_context
        self._active_data_context = data_context
        
        # Add to conversation
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        if conv_id:
            # Add data files to conversation
            self.conversation_manager.add_data_context(data_files, conv_id)
            
            # Update system prompt with context summary
            summary = data_context.get("summary", "")
            system_prompt = (
                "You are a data assistant helping with data analysis and understanding. "
                "Provide clear, concise answers about data structure, relationships, and insights. "
                f"\n\nData context: {summary}"
            )
            self.conversation_manager.update_system_prompt(system_prompt, conv_id)
        
        return data_context
    
    def generate_guidance(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate guidance for the current conversation.
        
        Args:
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Dictionary with guidance elements
        """
        if not self._active_data_context:
            return {"error": "No active data context"}
        
        # Get conversation history
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        if not conv_id:
            return {"error": "No active conversation"}
            
        history = self._get_conversation_history(conv_id)
        
        # Generate guidance
        return self.user_guidance.generate_guidance(
            data_contexts=self._active_data_context,
            conversation_history=history
        )
    
    def optimize_schema_with_fallback(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize schema using online fallback if available.
        
        Args:
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Schema optimization results
        """
        if not self._active_data_context:
            return {"error": "No active data context"}
        
        if not self.enable_online_fallback or not self.online_llm:
            return {"error": "Online LLM fallback not enabled"}
        
        # Extract file contexts and relationships
        file_contexts = self._active_data_context.get("files", {})
        relationships = self._active_data_context.get("relationships", [])
        
        # Optimize schema using online LLM
        result = self.online_llm.optimize_schema(file_contexts, relationships)
        
        # Add results to conversation
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        if conv_id:
            schema_summary = "I've analyzed your data and optimized the database schema. "
            
            if "tables" in result:
                table_count = len(result.get("tables", []))
                schema_summary += f"The optimized schema contains {table_count} tables with proper relationships and data types."
            else:
                schema_summary += "The schema has been optimized for data integrity and query efficiency."
                
            # Add as assistant message
            self.conversation_manager.add_assistant_message(schema_summary, conv_id)
        
        return result
    
    def detect_advanced_relationships_with_fallback(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect advanced relationships using online fallback if available.
        
        Args:
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Advanced relationship detection results
        """
        if not self._active_data_context:
            return {"error": "No active data context"}
        
        if not self.enable_online_fallback or not self.online_llm:
            return {"error": "Online LLM fallback not enabled"}
        
        # Extract file contexts
        file_contexts = self._active_data_context.get("files", {})
        
        # Detect relationships using online LLM
        result = self.online_llm.detect_advanced_relationships(file_contexts)
        
        # Add results to conversation
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        if conv_id:
            relationship_summary = "I've analyzed your data files in depth and found additional relationships. "
            
            if "relationships" in result:
                rel_count = len(result.get("relationships", []))
                relationship_summary += f"I discovered {rel_count} potential relationships between your datasets."
            else:
                relationship_summary += "The relationships between your datasets have been mapped."
                
            # Add as assistant message
            self.conversation_manager.add_assistant_message(relationship_summary, conv_id)
        
        return result
    
    def streaming_response(self, message: str, conversation_id: Optional[str] = None, 
                        callback: Optional[callable] = None) -> Generator[str, None, None]:
        """
        Generate a streaming response to a message.
        
        Args:
            message: User message
            conversation_id: Conversation ID (uses active if None)
            callback: Optional callback for each chunk
            
        Returns:
            Generator yielding response chunks
        """
        # Add user message
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        self.conversation_manager.add_user_message(message, conv_id)
        
        # Create a buffer for the complete response
        full_response = []
        
        def buffer_callback(chunk):
            full_response.append(chunk)
            if callback:
                callback(chunk)
        
        # Generate streaming response
        response_gen = self.conversation_manager.generate_response(
            conv_id,
            stream=True,
            callback=buffer_callback
        )
        
        # Process the stream
        try:
            for chunk in response_gen:
                yield chunk
                
            # After streaming completes, add full response to conversation
            full_response_text = "".join(full_response)
            self.conversation_manager.add_assistant_message(full_response_text, conv_id)
            
        except Exception as e:
            logger.error(f"Error during streaming response: {str(e)}")
            error_message = f"Sorry, an error occurred while generating the response: {str(e)}"
            self.conversation_manager.add_assistant_message(error_message, conv_id)
            yield error_message
    
    def explain_data_pattern(self, pattern: str) -> str:
        """
        Explain a data pattern in simple terms.
        
        Args:
            pattern: Description of the pattern to explain
            
        Returns:
            Simple explanation
        """
        if not self._active_data_context:
            return "I need data context to explain patterns. Please upload some data files first."
        
        return self.user_guidance.explain_data_pattern(pattern, self._active_data_context)
    
    def load_conversation(self, conversation_id: str) -> bool:
        """
        Load a conversation and associated data context.
        
        Args:
            conversation_id: ID of conversation to load
            
        Returns:
            Success flag
        """
        # Load conversation
        result = self.conversation_manager.set_active_conversation(conversation_id)
        if not result:
            return False
        
        # Load associated data context
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation and conversation.data_files:
            data_files = conversation.data_files
            
            # Check if we have cached context
            context_key = self._create_context_key(data_files)
            if context_key in self._data_context_cache:
                self._active_data_context = self._data_context_cache[context_key]
            else:
                # Build context
                self._active_data_context = self.data_context_builder.build_context_for_files(
                    file_paths=data_files
                )
                self._data_context_cache[context_key] = self._active_data_context
        
        return True
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to local LLM.
        
        Returns:
            Connection status
        """
        return self.llm_client.check_connection()
    
    def _create_context_key(self, file_paths: List[str]) -> str:
        """Create a unique key for data context cache."""
        return "|".join(sorted(file_paths))
    
    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get processed conversation history for a conversation."""
        conv = self.conversation_manager.get_conversation(conversation_id)
        if not conv:
            return []
            
        history = []
        for msg in conv.messages:
            if msg.role != "system":  # Skip system messages
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return history
