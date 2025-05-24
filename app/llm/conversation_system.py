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
from app.llm.anthropic_client import AnthropicClient, AnthropicConfig
from app.llm.multi_llm_orchestrator import MultiLLMOrchestrator, MultiLLMResult
from app.processors.smart_query_processor import SmartQueryProcessor, ProcessedResults
import pandas as pd

logger = get_logger(__name__)

class LLMConversationSystem:
    """
    Main interface for the LLM conversation system.
    
    Integrates all components:
    - LMStudioClient for local LLM integration
    - ConversationManager for conversation history
    - DataContextBuilder for data analysis
    - UserGuidanceSystem for intelligent guidance
    - OnlineLLMFallback for complex tasks (OpenAI)
    - AnthropicClient for Claude integration
    - MultiLLMOrchestrator for consensus-based responses
    - SmartQueryProcessor for direct data processing
    """
    
    def __init__(self,
                local_llm_url: str = "http://127.0.0.1:1234/v1",
                local_llm_model: str = "claude-3.7-sonnet-reasoning-gemma3-12b",
                local_llm_timeout: int = 300,  # Add timeout configuration for local LLM
                memory_monitor: MemoryMonitor = None,
                resource_optimizer: ResourceOptimizer = None,
                online_llm_config: OnlineLLMConfig = None,
                anthropic_config: AnthropicConfig = None,
                conversation_dir: str = "conversations",
                enable_online_fallback: bool = False,
                enable_anthropic: bool = False,
                primary_llm: str = "local",  # "local", "anthropic", "openai", "multi"
                enable_multi_llm: bool = False,  # NEW: Enable multi-LLM mode
                enable_smart_processing: bool = True):  # NEW: Enable smart data processing
        """
        Initialize the LLM conversation system.
        
        Args:
            local_llm_url: URL for local LLM
            local_llm_model: Model name for local LLM
            local_llm_timeout: Timeout for local LLM requests in seconds
            memory_monitor: Optional memory monitor instance
            resource_optimizer: Optional resource optimizer instance
            online_llm_config: Configuration for OpenAI
            anthropic_config: Configuration for Anthropic Claude
            conversation_dir: Directory for conversation storage
            enable_online_fallback: Whether to enable OpenAI fallback
            enable_anthropic: Whether to enable Anthropic Claude
            primary_llm: Primary LLM to use ("local", "anthropic", "openai", "multi")
            enable_multi_llm: Whether to enable multi-LLM consensus mode
            enable_smart_processing: Whether to enable smart data processing
        """
        # Initialize memory management
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer(self.memory_monitor)
        
        # Store configuration
        self.primary_llm = primary_llm
        self.enable_online_fallback = enable_online_fallback
        self.enable_anthropic = enable_anthropic
        self.enable_multi_llm = enable_multi_llm
        self.enable_smart_processing = enable_smart_processing
        
        # Initialize Smart Query Processor
        self.smart_processor = SmartQueryProcessor() if enable_smart_processing else None
        
        # Initialize LLM clients
        self.llm_client = LMStudioClient(
            base_url=local_llm_url,
            model=local_llm_model,
            timeout=local_llm_timeout,  # Pass timeout parameter
            memory_monitor=self.memory_monitor
        )
        
        # Initialize Anthropic client if enabled
        self.anthropic_client = None
        if enable_anthropic and anthropic_config:
            self.anthropic_client = AnthropicClient(
                config=anthropic_config,
                memory_monitor=self.memory_monitor
            )
        
        # Initialize OpenAI client if enabled
        self.online_llm = None
        if enable_online_fallback and online_llm_config:
            self.online_llm = OnlineLLMFallback(
                config=online_llm_config,
                memory_monitor=self.memory_monitor
            )
        
        # Initialize Multi-LLM Orchestrator
        self.multi_llm_orchestrator = None
        if enable_multi_llm:
            self.multi_llm_orchestrator = MultiLLMOrchestrator(
                local_llm_client=self.llm_client,
                anthropic_client=self.anthropic_client,
                openai_client=self.online_llm,
                memory_monitor=self.memory_monitor
            )
        
        # Initialize conversation management (use appropriate client)
        primary_client = self._get_primary_client()
        self.conversation_manager = ConversationManager(
            llm_client=primary_client,
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
            llm_client=primary_client
        )
        
        # Data context cache
        self._data_context_cache = {}
        self._active_data_context = None
        self._active_data_files = {}  # Store actual DataFrames for smart processing
        
        logger.info(f"LLMConversationSystem initialized with primary LLM: {primary_llm}, multi-LLM: {enable_multi_llm}, smart processing: {enable_smart_processing}")
    
    def _get_primary_client(self):
        """Get the primary LLM client based on configuration."""
        if self.primary_llm == "anthropic" and self.anthropic_client:
            return self.anthropic_client
        elif self.primary_llm == "openai" and self.online_llm:
            return self.online_llm
        else:
            # Default to local LLM
            return self.llm_client
    
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
        Add a user message and generate a response with smart data processing.
        
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
        
        # SAFETY CHECK: Ensure data files are loaded for this conversation
        conversation = self.conversation_manager.get_conversation(conv_id)
        if (conversation and conversation.data_files and 
            (not self._active_data_files or len(self._active_data_files) == 0)):
            
            logger.warning(f"Data files not loaded for conversation {conv_id}, forcing reload...")
            logger.warning(f"Conversation has {len(conversation.data_files)} data files")
            logger.warning(f"Current active data files: {len(self._active_data_files) if self._active_data_files else 0}")
            
            try:
                # Force rebuild data context
                self.build_data_context(conversation.data_files, conv_id)
                logger.info(f"Forced data reload completed. Active files: {len(self._active_data_files) if self._active_data_files else 0}")
            except Exception as e:
                logger.error(f"Failed to force reload data files: {str(e)}")
        
        # Debug logging for smart processing
        logger.info(f"Smart processing check - enabled: {self.enable_smart_processing}")
        logger.info(f"Smart processor available: {self.smart_processor is not None}")
        logger.info(f"Active data files count: {len(self._active_data_files) if self._active_data_files else 0}")
        if self._active_data_files:
            logger.info(f"Active data files: {list(self._active_data_files.keys())}")
        
        # Try smart processing first if enabled and data is available
        if (self.enable_smart_processing and self.smart_processor and 
            self._active_data_files and len(self._active_data_files) > 0):
            
            try:
                logger.info("Attempting smart data processing...")
                logger.info(f"Message being processed: '{message}'")
                logger.info(f"Message type: {type(message)}")
                logger.info(f"Active data files count: {len(self._active_data_files)}")
                
                processed_results = self.smart_processor.process_query(message, self._active_data_files)
                
                # If smart processing succeeded, use those results
                if processed_results and processed_results.calculation_method != "error_handling":
                    logger.info(f"Smart processing successful: {processed_results.calculation_method}")
                    
                    # Create optimized context for LLM
                    smart_context = self.smart_processor.create_llm_context(message, processed_results)
                    
                    # Generate LLM response using processed results
                    if self.primary_llm == "anthropic" and self.anthropic_client:
                        response = self._generate_anthropic_smart_response(smart_context, conv_id)
                    else:
                        response = self._generate_local_smart_response(smart_context, conv_id)
                    
                    # Calculate processing time
                    processing_time_ms = self.memory_monitor.current_time_ms() - start_time
                    
                    return {
                        "response": response,
                        "conversation_id": conv_id,
                        "processing_time_ms": processing_time_ms,
                        "llm_used": self.primary_llm,
                        "smart_processing": True,
                        "processed_results": {
                            "primary_answer": processed_results.primary_answer,
                            "calculation_method": processed_results.calculation_method,
                            "detailed_results": processed_results.detailed_results
                        }
                    }
                else:
                    logger.warning(f"Smart processing failed with method: {processed_results.calculation_method if processed_results else 'None'}")
                    logger.warning(f"Failed result primary answer: {processed_results.primary_answer if processed_results else 'None'}")
                    
            except Exception as e:
                logger.warning(f"Smart processing failed, falling back to normal processing: {str(e)}")
                import traceback
                logger.warning(f"Smart processing traceback: {traceback.format_exc()}")
        else:
            logger.info("Smart processing conditions not met:")
            logger.info(f"  - Smart processing enabled: {self.enable_smart_processing}")
            logger.info(f"  - Smart processor exists: {self.smart_processor is not None}")
            logger.info(f"  - Active data files exist: {self._active_data_files is not None}")
            logger.info(f"  - Active data files count: {len(self._active_data_files) if self._active_data_files else 0}")
        
        # Fallback to normal processing
        logger.info("Using normal LLM processing...")
        
        # Generate response using primary LLM
        if self.primary_llm == "anthropic" and self.anthropic_client:
            response = self._generate_anthropic_response(message, conv_id)
        else:
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
            "processing_time_ms": processing_time_ms,
            "llm_used": self.primary_llm,
            "smart_processing": False
        }
    
    def _generate_local_smart_response(self, smart_context: str, conversation_id: str) -> str:
        """Generate response using local LLM with smart processed context."""
        try:
            # Create a temporary conversation with smart context
            temp_messages = [
                {"role": "system", "content": "You are a data analyst explaining calculated results. The calculations have been done correctly - just interpret and explain them clearly."},
                {"role": "user", "content": smart_context}
            ]
            
            # Generate response
            response = self.llm_client.chat_completion(temp_messages)
            
            # Add response to conversation
            self.conversation_manager.add_assistant_message(response, conversation_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating local smart response: {str(e)}")
            # Fallback to normal processing
            return self.conversation_manager.generate_response(conversation_id)
    
    def _generate_anthropic_smart_response(self, smart_context: str, conversation_id: str) -> str:
        """Generate response using Anthropic with smart processed context."""
        try:
            messages = [{"role": "user", "content": smart_context}]
            response = self.anthropic_client.chat_completion(messages)
            
            # Add response to conversation
            self.conversation_manager.add_assistant_message(response, conversation_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Anthropic smart response: {str(e)}")
            # Fallback to normal processing
            return self._generate_anthropic_response("Please explain the data analysis results.", conversation_id)
    
    def _generate_anthropic_response(self, message: str, conversation_id: str) -> str:
        """Generate response using Anthropic Claude."""
        try:
            # Get conversation history
            history = self._get_conversation_history(conversation_id)
            
            # Format messages for Anthropic
            messages = []
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Generate response using Anthropic
            if self._active_data_context:
                response = self.anthropic_client.analyze_data_structure(
                    self._active_data_context, message
                )
            else:
                response = self.anthropic_client.chat_completion(messages)
            
            # Add response to conversation
            self.conversation_manager.add_assistant_message(response, conversation_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {str(e)}")
            # Fallback to local LLM
            return self.conversation_manager.generate_response(conversation_id)
    
    def build_data_context(self, data_files: List[str], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build context for data files and prepare for smart processing.
        
        Args:
            data_files: List of data files
            conversation_id: Conversation ID (uses active if None)
            
        Returns:
            Generated data context
        """
        logger.info(f"🏗️ Starting build_data_context with {len(data_files)} files")
        logger.info(f"Files to process: {data_files}")
        logger.info(f"Smart processing enabled: {self.enable_smart_processing}")
        
        # Build context
        logger.info("📊 Building data context...")
        data_context = self.data_context_builder.build_context_for_files(
            file_paths=data_files,
            include_stats=True,
            include_samples=True,
            include_relationships=True,
            include_quality=True
        )
        logger.info("✅ Data context built successfully")
        
        # Load DataFrames for smart processing
        if self.enable_smart_processing:
            logger.info("🤖 Loading DataFrames for smart processing...")
            self._active_data_files = {}
            
            for i, file_path in enumerate(data_files, 1):
                logger.info(f"📁 Processing file {i}/{len(data_files)}: {file_path}")
                logger.info(f"   File exists: {os.path.exists(file_path)}")
                
                try:
                    if file_path.endswith('.csv'):
                        logger.info(f"   📊 Attempting to load CSV...")
                        df = pd.read_csv(file_path)
                        self._active_data_files[file_path] = df
                        logger.info(f"   ✅ Successfully loaded {len(df)} rows from {file_path} for smart processing")
                    else:
                        logger.info(f"   ⚠️ Skipping non-CSV file: {file_path}")
                        
                except Exception as e:
                    logger.error(f"   ❌ Could not load {file_path} for smart processing: {str(e)}")
                    logger.error(f"   Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
            
            logger.info(f"🎯 Smart processing summary: {len(self._active_data_files)} files loaded successfully")
            if self._active_data_files:
                for file_path, df in self._active_data_files.items():
                    logger.info(f"   ✅ {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} columns")
            else:
                logger.warning("⚠️ No files were loaded into _active_data_files!")
        else:
            logger.info("⚠️ Smart processing is disabled, skipping DataFrame loading")
        
        # Cache the context
        context_key = self._create_context_key(data_files)
        self._data_context_cache[context_key] = data_context
        self._active_data_context = data_context
        logger.info(f"💾 Cached data context with key: {context_key}")
        
        # Add to conversation
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        logger.info(f"🗨️ Using conversation ID: {conv_id}")
        logger.info(f"   Active conversation ID: {self.conversation_manager.active_conversation_id}")
        
        if conv_id:
            logger.info("📝 Adding data context to conversation...")
            # Add data files to conversation
            self.conversation_manager.add_data_context(data_files, conv_id)
            
            # Update system prompt with context summary
            summary = data_context.get("summary", "")
            
            # Create enhanced prompt with data samples for calculations
            data_samples_text = self._create_data_samples_text(data_context)
            
            # Determine if this is multi-file analysis
            is_multi_file = len(data_files) > 1
            logger.info(f"📊 Multi-file analysis: {is_multi_file}")
            
            if self.enable_smart_processing:
                # Use simplified prompt since smart processing handles calculations
                if is_multi_file:
                    system_prompt = (
                        "You are a data analyst explaining pre-calculated results from data comparisons.\n\n"
                        "IMPORTANT: Data calculations have already been performed correctly by our processing engine.\n"
                        "Your role is to interpret and explain the calculated results in a clear, conversational way.\n\n"
                        "When you receive calculated results:\n"
                        "1. Trust the calculations - they are mathematically correct\n"
                        "2. Focus on explaining what the numbers mean\n"
                        "3. Provide insights about the data relationships\n"
                        "4. Answer follow-up questions about the analysis\n\n"
                        "NEVER suggest Excel or external tools - the analysis is complete.\n\n"
                        f"Data context: {summary}"
                    )
                else:
                    system_prompt = (
                        "You are a data analyst explaining pre-calculated results from data analysis.\n\n"
                        "IMPORTANT: Data calculations have already been performed correctly.\n"
                        "Your role is to interpret and explain the calculated results clearly.\n\n"
                        f"Data context: {summary}"
                    )
            else:
                # Use the original forceful prompts for backwards compatibility
                if is_multi_file:
                    system_prompt = (
                        "YOU HAVE COMPLETE ACCESS TO THE DATA. ANALYZE IT DIRECTLY.\n\n"
                        "MANDATORY RULES - NO EXCEPTIONS:\n"
                        "1. NEVER mention Excel, spreadsheets, or external tools\n"
                        "2. PERFORM ALL CALCULATIONS DIRECTLY using the provided data\n"
                        "3. You have FULL ACCESS to both datasets with complete statistics\n"
                        "4. When asked to compare files, do the comparison immediately\n"
                        "5. When asked for missing amounts, calculate them directly\n\n"
                        "CALCULATION METHODOLOGY:\n"
                        "- Use the provided FULL DATASET statistics for all calculations\n"
                        "- Compare totals, counts, and identify differences directly\n"
                        "- Match transactions using available identifiers\n"
                        "- Provide specific numbers, not generic advice\n\n"
                        "REQUIRED RESPONSE FORMAT:\n"
                        "'Direct analysis of your datasets:\n"
                        "File 1: [name] - [count] transactions, total RM [amount]\n"
                        "File 2: [name] - [count] transactions, total RM [amount]\n"
                        "Difference: [calculation and explanation]'\n\n"
                        "YOU MUST ANALYZE THE DATA DIRECTLY. DO NOT SUGGEST TOOLS.\n\n"
                        f"Data context: {summary}\n\n"
                        f"Complete multi-dataset information:\n{data_samples_text}"
                    )
                else:
                    system_prompt = (
                        "YOU HAVE COMPLETE ACCESS TO THIS DATASET. ANALYZE IT DIRECTLY.\n\n"
                        "MANDATORY RULES:\n"
                        "1. Use FULL DATASET statistics for all calculations\n"
                        "2. Never mention Excel or external tools\n"
                        "3. Provide specific numerical answers\n"
                        "4. Base calculations on the provided total sums and counts\n\n"
                        f"Data context: {summary}\n\n"
                        f"Complete dataset information:\n{data_samples_text}"
                    )
            
            self.conversation_manager.update_system_prompt(system_prompt, conv_id)
            logger.info("✅ Updated system prompt for conversation")
        else:
            logger.warning("⚠️ No conversation ID available, skipping conversation updates")
        
        logger.info("🎉 build_data_context completed successfully")
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
        
        # Extract file contexts and relationships
        file_contexts = self._active_data_context.get("files", {})
        relationships = self._active_data_context.get("relationships", [])
        
        # Try Anthropic first if available
        if self.enable_anthropic and self.anthropic_client:
            try:
                result = self.anthropic_client.optimize_schema(file_contexts, relationships)
                
                # Add results to conversation
                conv_id = conversation_id or self.conversation_manager.active_conversation_id
                if conv_id:
                    schema_summary = "I've analyzed your data using Claude and optimized the database schema. "
                    
                    if "tables" in result:
                        table_count = len(result.get("tables", []))
                        schema_summary += f"The optimized schema contains {table_count} tables with proper relationships and data types."
                    else:
                        schema_summary += "The schema has been optimized for data integrity and query efficiency."
                        
                    # Add as assistant message
                    self.conversation_manager.add_assistant_message(schema_summary, conv_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Error with Anthropic schema optimization: {str(e)}")
        
        # Fallback to OpenAI if available
        if self.enable_online_fallback and self.online_llm:
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
        
        return {"error": "No online LLM services available for schema optimization"}
    
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
        
        # Extract file contexts
        file_contexts = self._active_data_context.get("files", {})
        
        # Try Anthropic first if available
        if self.enable_anthropic and self.anthropic_client:
            try:
                # Use Anthropic's general chat completion for relationship detection
                system_prompt = """You are an expert in database relationships and data modeling. 
                Analyze the provided data files to detect potential relationships beyond simple key matches. 
                Look for semantic relationships, hierarchical structures, and complex dependencies."""
                
                # Format the input data
                files_info = []
                for file_path, context in file_contexts.items():
                    file_name = os.path.basename(file_path)
                    columns = context.get("column_names", [])
                    
                    file_info = f"File: {file_name}\nColumns: {', '.join(columns)}"
                    if "sample_data" in context:
                        file_info += f"\nSample data available: {len(context['sample_data'])} rows"
                    files_info.append(file_info)
                
                user_message = f"""Please analyze these data files to detect potential relationships:

{chr(10).join(files_info)}

Provide insights about potential relationships between these datasets."""
                
                messages = [{"role": "user", "content": user_message}]
                response = self.anthropic_client.chat_completion(messages, system_prompt)
                
                result = {"explanation": response, "relationships": []}
                
                # Add results to conversation
                conv_id = conversation_id or self.conversation_manager.active_conversation_id
                if conv_id:
                    relationship_summary = "I've analyzed your data files using Claude and found potential relationships between your datasets."
                    self.conversation_manager.add_assistant_message(relationship_summary, conv_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Error with Anthropic relationship detection: {str(e)}")
        
        # Fallback to OpenAI if available
        if self.enable_online_fallback and self.online_llm:
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
        
        return {"error": "No online LLM services available for relationship detection"}
    
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
        
        # Generate streaming response using appropriate client
        if self.primary_llm == "anthropic" and self.anthropic_client:
            try:
                # Get conversation history
                history = self._get_conversation_history(conv_id)
                messages = []
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": message})
                
                # Generate streaming response
                response_gen = self.anthropic_client.chat_completion(
                    messages, stream=True
                )
                
                for chunk in response_gen:
                    full_response.append(chunk)
                    if callback:
                        callback(chunk)
                    yield chunk
                    
            except Exception as e:
                error_message = f"Error with Anthropic streaming: {str(e)}"
                logger.error(error_message)
                yield error_message
        else:
            # Use local LLM streaming
            response_gen = self.conversation_manager.generate_response(
                conv_id,
                stream=True,
                callback=buffer_callback
            )
            
            try:
                for chunk in response_gen:
                    yield chunk
            except Exception as e:
                logger.error(f"Error during streaming response: {str(e)}")
                error_message = f"Sorry, an error occurred while generating the response: {str(e)}"
                yield error_message
        
        # After streaming completes, add full response to conversation
        if full_response:
            full_response_text = "".join(full_response)
            self.conversation_manager.add_assistant_message(full_response_text, conv_id)
    
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
        Check connection to all available LLM services.
        
        Returns:
            Connection status for all services
        """
        status = {
            "local_llm": self.llm_client.check_connection(),
            "primary_llm": self.primary_llm,
            "multi_llm_enabled": self.enable_multi_llm
        }
        
        if self.enable_anthropic and self.anthropic_client:
            status["anthropic"] = self.anthropic_client.check_connection()
        
        if self.enable_online_fallback and self.online_llm:
            # Check OpenAI connection (simplified)
            status["openai_fallback"] = {"status": "configured"}
        
        # Multi-LLM orchestrator stats
        if self.enable_multi_llm and self.multi_llm_orchestrator:
            status["multi_llm_providers"] = self.multi_llm_orchestrator.get_provider_stats()
        
        return status
    
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

    async def add_message_multi_llm(self, message: str, conversation_id: Optional[str] = None, 
                                  providers: Optional[List[str]] = None,
                                  timeout_per_provider: Optional[int] = None) -> Dict[str, Any]:
        """
        Add a user message and generate responses from multiple LLMs.
        
        Args:
            message: User message
            conversation_id: Conversation ID (uses active if None)
            providers: Specific providers to use (None for all available)
            timeout_per_provider: Timeout per provider in seconds
            
        Returns:
            Dictionary with multi-LLM responses and metadata
        """
        if not self.enable_multi_llm or not self.multi_llm_orchestrator:
            # Fallback to single LLM mode
            return self.add_message(message, conversation_id)
        
        start_time = self.memory_monitor.current_time_ms()
        
        # Add user message
        conv_id = conversation_id or self.conversation_manager.active_conversation_id
        self.conversation_manager.add_user_message(message, conv_id)
        
        # Get multi-LLM response
        try:
            multi_result = await self.multi_llm_orchestrator.get_multi_llm_response(
                query=message,
                context=self._active_data_context,
                providers=providers,  # Pass through provider selection
                include_consensus=True
            )
            
            # Add the best response to conversation
            if multi_result.best_response:
                best_response_text = multi_result.best_response.response
                self.conversation_manager.add_assistant_message(best_response_text, conv_id)
            
            # Generate guidance
            guidance = None
            if self._active_data_context:
                history = self._get_conversation_history(conv_id)
                guidance = self.user_guidance.generate_guidance(
                    data_contexts=self._active_data_context,
                    conversation_history=history
                )
            
            # Calculate processing time
            processing_time_ms = self.memory_monitor.current_time_ms() - start_time
            
            return {
                "multi_llm_result": multi_result,
                "best_response": multi_result.best_response.response if multi_result.best_response else "No response generated",
                "consensus_response": multi_result.consensus_response,
                "all_responses": [
                    {
                        "provider": r.provider,
                        "response": r.response,
                        "confidence": r.confidence_score,
                        "response_time": r.response_time_ms,
                        "success": r.success,
                        "error": r.error
                    }
                    for r in multi_result.responses
                ],
                "conversation_id": conv_id,
                "guidance": guidance,
                "processing_time_ms": processing_time_ms,
                "mode": "multi_llm",
                "comparison": self.multi_llm_orchestrator.get_response_comparison(multi_result)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-LLM response: {str(e)}")
            # Fallback to single LLM
            return self.add_message(message, conv_id)

    def _create_data_samples_text(self, data_context: Dict[str, Any]) -> str:
        """Create a text representation of data samples for the system prompt."""
        data_samples_text = ""
        
        try:
            files_data = data_context.get("files", {})
            
            for file_path, context in files_data.items():
                file_name = os.path.basename(file_path)
                data_samples_text += f"\n=== DATA FROM FILE: {file_name} ===\n"
                
                # Add comprehensive file info
                if "row_count" in context and "column_count" in context:
                    data_samples_text += f"FULL DATASET: {context.get('row_count', 'unknown')} total rows, {context['column_count']} columns\n"
                
                # Add column information
                if "column_names" in context and context["column_names"]:
                    data_samples_text += f"Columns: {', '.join(context['column_names'])}\n"
                
                # Add full dataset statistics if available
                if "statistics" in context and "column_stats" in context["statistics"]:
                    stats = context["statistics"]["column_stats"]
                    data_samples_text += f"\nFULL DATASET STATISTICS:\n"
                    for col, col_stats in stats.items():
                        if isinstance(col_stats, dict):
                            if "sum" in col_stats and col_stats.get("full_dataset", False):
                                data_samples_text += f"{col} - Total Sum: {col_stats['sum']}, Count: {col_stats.get('count', 'N/A')}, Mean: {col_stats.get('mean', 'N/A')}\n"
                            elif "unique_count" in col_stats:
                                data_samples_text += f"{col} - Unique Values: {col_stats['unique_count']}\n"
                
                # Add sample data for format understanding
                if "sample_data" in context and context["sample_data"]:
                    sample_data = context["sample_data"]
                    data_samples_text += f"\nSAMPLE ROWS (for format reference - calculations should use FULL dataset):\n"
                    
                    try:
                        # Convert to DataFrame if needed
                        if isinstance(sample_data, list):
                            df = pd.DataFrame(sample_data)
                        else:
                            df = sample_data
                        
                        # Show first 5 rows as examples
                        df_sample = df.head(5)
                        
                        # Format each row clearly
                        data_samples_text += f"Showing first {len(df_sample)} rows as examples:\n\n"
                        
                        for idx, row in df_sample.iterrows():
                            data_samples_text += f"Row {idx + 1}:\n"
                            for col, value in row.items():
                                if pd.notna(value):
                                    data_samples_text += f"  {col}: {value}\n"
                                else:
                                    data_samples_text += f"  {col}: [empty]\n"
                            data_samples_text += "\n"
                    
                    except Exception as e:
                        data_samples_text += f"Error formatting sample data: {str(e)}\n"
                
                data_samples_text += "\n" + "="*50 + "\n"
                
        except Exception as e:
            data_samples_text += f"Error creating data samples text: {str(e)}\n"
        
        # Add final instruction
        data_samples_text += "\nIMPORTANT: Use the FULL DATASET statistics above for calculations. Sample rows are shown only for format reference.\n"
        
        return data_samples_text
