"""
Conversation management system for data-centric LLM interactions.

This module provides conversation history management with data context,
multi-turn discussion handling, and context compression for efficient
memory usage when discussing data files.
"""
import time
import json
import uuid
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict

from app.utils.logging_config import get_logger
from app.llm.lm_studio_client import LMStudioClient
from app.memory.memory_monitor import MemoryMonitor

logger = get_logger(__name__)

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    """A conversation with metadata and messages."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Data Conversation"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Data context information
    data_files: List[str] = field(default_factory=list)
    compressed_context: Optional[str] = None
    context_summary: Optional[str] = None

class ConversationManager:
    """
    Manages conversations about data with context tracking.
    
    This class:
    - Maintains conversation history with data context
    - Manages multi-turn discussions about uploaded data
    - Implements context compression for long conversations
    - Provides data-aware response generation
    """
    
    def __init__(self, 
                llm_client: LMStudioClient,
                storage_dir: str = "conversations",
                max_conversation_messages: int = 20,
                compression_threshold: int = 10,
                memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the conversation manager.
        
        Args:
            llm_client: LM Studio client for generating responses
            storage_dir: Directory for persisting conversations
            max_conversation_messages: Maximum number of messages before compression
            compression_threshold: Number of messages that triggers compression
            memory_monitor: Memory monitor for resource tracking
        """
        self.llm_client = llm_client
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
        # Configure storage
        self.storage_dir = os.path.join(os.getcwd(), storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Settings
        self.max_conversation_messages = max_conversation_messages
        self.compression_threshold = compression_threshold
        
        # Active conversations
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation_id: Optional[str] = None
        
        # System prompts
        self.default_system_prompt = (
            "You are a data assistant helping with data analysis and understanding. "
            "Provide clear, concise answers about data structure, relationships, and insights. "
            "When discussing data, focus on practical insights and actionable recommendations."
        )
        
        logger.info("ConversationManager initialized")
    
    def create_conversation(self, title: str = "Data Conversation", data_files: List[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            title: The conversation title
            data_files: List of data file paths associated with this conversation
            
        Returns:
            ID of the created conversation
        """
        conversation = Conversation(
            title=title,
            data_files=data_files or []
        )
        
        # Add system message
        system_message = Message(
            role="system",
            content=self.default_system_prompt
        )
        conversation.messages.append(system_message)
        
        # Store and set as active
        self.conversations[conversation.conversation_id] = conversation
        self.active_conversation_id = conversation.conversation_id
        
        logger.info(f"Created new conversation: {conversation.conversation_id} - {title}")
        
        return conversation.conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation if found, None otherwise
        """
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
            
        # Try to load from disk
        try:
            return self._load_conversation(conversation_id)
        except:
            logger.warning(f"Conversation not found: {conversation_id}")
            return None
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all available conversations.
        
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        # Add in-memory conversations
        for conv_id, conv in self.conversations.items():
            conversations.append({
                "conversation_id": conv_id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": len(conv.messages),
                "data_files": conv.data_files
            })
        
        # Look for conversations on disk that aren't loaded
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    conv_id = filename.replace('.json', '')
                    
                    # Skip already loaded conversations
                    if conv_id in self.conversations:
                        continue
                        
                    try:
                        conv = self._load_conversation(conv_id)
                        if conv:
                            conversations.append({
                                "conversation_id": conv_id,
                                "title": conv.title,
                                "created_at": conv.created_at,
                                "updated_at": conv.updated_at,
                                "message_count": len(conv.messages),
                                "data_files": conv.data_files
                            })
                    except Exception as e:
                        logger.error(f"Error loading conversation {conv_id}: {str(e)}")
        
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Remove from memory if loaded
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
            # Clear active conversation if needed
            if self.active_conversation_id == conversation_id:
                self.active_conversation_id = None
        
        # Remove from disk if exists
        file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        
        return False
    
    def add_user_message(self, content: str, conversation_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a user message to a conversation.
        
        Args:
            content: The message content
            conversation_id: ID of the conversation (uses active if None)
            metadata: Additional message metadata
            
        Returns:
            ID of the message
        """
        conv_id = conversation_id or self.active_conversation_id
        
        # Create conversation if none exists
        if not conv_id or conv_id not in self.conversations:
            conv_id = self.create_conversation()
        
        conversation = self.conversations[conv_id]
        
        # Create the message
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {}
        )
        
        # Add to conversation
        conversation.messages.append(message)
        conversation.updated_at = time.time()
        
        # Save the conversation
        self._save_conversation(conversation)
        
        logger.info(f"Added user message to conversation {conv_id}: {message.message_id}")
        
        return message.message_id
    
    def add_assistant_message(self, content: str, conversation_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add an assistant message to a conversation.
        
        Args:
            content: The message content
            conversation_id: ID of the conversation (uses active if None)
            metadata: Additional message metadata
            
        Returns:
            ID of the message
        """
        conv_id = conversation_id or self.active_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            logger.error("No active conversation for assistant message")
            raise ValueError("No active conversation for assistant message")
        
        conversation = self.conversations[conv_id]
        
        # Create the message
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {}
        )
        
        # Add to conversation
        conversation.messages.append(message)
        conversation.updated_at = time.time()
        
        # Check if we need to compress
        if len(conversation.messages) > self.compression_threshold:
            self._compress_conversation_if_needed(conversation)
        
        # Save the conversation
        self._save_conversation(conversation)
        
        logger.info(f"Added assistant message to conversation {conv_id}: {message.message_id}")
        
        return message.message_id
    
    def generate_response(self, 
                         conversation_id: Optional[str] = None, 
                         temperature: float = 0.7,
                         stream: bool = False,
                         callback: Optional[callable] = None) -> Union[str, Any]:
        """
        Generate a response for the current conversation state.
        
        Args:
            conversation_id: ID of the conversation (uses active if None)
            temperature: Temperature for generation
            stream: Whether to stream the response
            callback: Function to call with chunks when streaming
            
        Returns:
            Generated response text or stream generator
        """
        conv_id = conversation_id or self.active_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            logger.error("No active conversation for response generation")
            raise ValueError("No active conversation for response generation")
        
        conversation = self.conversations[conv_id]
        
        # Convert conversation to messages format
        messages = self._prepare_messages_for_llm(conversation)
        
        # Generate response using LLM client
        self.memory_monitor.start_tracking_step("conversation_response")
        
        try:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                stream=stream,
                callback=callback
            )
            
            # If not streaming, add to conversation
            if not stream:
                self.add_assistant_message(response, conv_id)
                
            return response
                
        finally:
            self.memory_monitor.end_tracking_step("conversation_response")
    
    def update_system_prompt(self, system_prompt: str, conversation_id: Optional[str] = None) -> bool:
        """
        Update the system prompt for a conversation.
        
        Args:
            system_prompt: New system prompt
            conversation_id: ID of the conversation (uses active if None)
            
        Returns:
            True if updated successfully, False otherwise
        """
        conv_id = conversation_id or self.active_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            logger.error("No active conversation for system prompt update")
            return False
        
        conversation = self.conversations[conv_id]
        
        # Find system message
        system_found = False
        for message in conversation.messages:
            if message.role == "system":
                message.content = system_prompt
                message.timestamp = time.time()
                system_found = True
                break
        
        # Add system message if not found
        if not system_found:
            system_message = Message(
                role="system",
                content=system_prompt
            )
            conversation.messages.insert(0, system_message)
        
        conversation.updated_at = time.time()
        
        # Save the conversation
        self._save_conversation(conversation)
        
        logger.info(f"Updated system prompt for conversation {conv_id}")
        
        return True
    
    def set_active_conversation(self, conversation_id: str) -> bool:
        """
        Set the active conversation.
        
        Args:
            conversation_id: ID of the conversation to set active
            
        Returns:
            True if set successfully, False otherwise
        """
        if conversation_id in self.conversations:
            self.active_conversation_id = conversation_id
            return True
            
        # Try to load from disk
        try:
            conversation = self._load_conversation(conversation_id)
            if conversation:
                self.conversations[conversation_id] = conversation
                self.active_conversation_id = conversation_id
                return True
        except:
            pass
            
        return False
    
    def add_data_context(self, data_files: List[str], conversation_id: Optional[str] = None) -> bool:
        """
        Add data files to the conversation context.
        
        Args:
            data_files: List of data file paths
            conversation_id: ID of the conversation (uses active if None)
            
        Returns:
            True if added successfully, False otherwise
        """
        conv_id = conversation_id or self.active_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            logger.error("No active conversation for adding data context")
            return False
        
        conversation = self.conversations[conv_id]
        
        # Add new files, avoiding duplicates
        for file_path in data_files:
            if file_path not in conversation.data_files:
                conversation.data_files.append(file_path)
        
        conversation.updated_at = time.time()
        
        # Save the conversation
        self._save_conversation(conversation)
        
        logger.info(f"Added data context to conversation {conv_id}: {data_files}")
        
        return True
    
    def update_context_summary(self, summary: str, conversation_id: Optional[str] = None) -> bool:
        """
        Update the context summary for a conversation.
        
        Args:
            summary: New context summary
            conversation_id: ID of the conversation (uses active if None)
            
        Returns:
            True if updated successfully, False otherwise
        """
        conv_id = conversation_id or self.active_conversation_id
        
        if not conv_id or conv_id not in self.conversations:
            logger.error("No active conversation for context summary update")
            return False
        
        conversation = self.conversations[conv_id]
        conversation.context_summary = summary
        conversation.updated_at = time.time()
        
        # Save the conversation
        self._save_conversation(conversation)
        
        logger.info(f"Updated context summary for conversation {conv_id}")
        
        return True
    
    def _prepare_messages_for_llm(self, conversation: Conversation) -> List[Dict[str, str]]:
        """
        Prepare messages for sending to the LLM.
        
        Args:
            conversation: The conversation to prepare messages from
            
        Returns:
            List of formatted message dictionaries
        """
        formatted_messages = []
        
        # Determine if we need to use compressed history
        if conversation.compressed_context and len(conversation.messages) > self.max_conversation_messages:
            # Add compressed context as a system message
            formatted_messages.append({
                "role": "system",
                "content": conversation.compressed_context
            })
            
            # Add recent messages only
            recent_messages = conversation.messages[-self.max_conversation_messages:]
            for message in recent_messages:
                if message.role != "system":  # Skip system messages as we've added compressed context
                    formatted_messages.append({
                        "role": message.role,
                        "content": message.content
                    })
        else:
            # Include all messages
            for message in conversation.messages:
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        # Add context summary if available
        if conversation.context_summary:
            # Find the system message to enhance
            system_idx = None
            for i, msg in enumerate(formatted_messages):
                if msg["role"] == "system":
                    system_idx = i
                    break
            
            if system_idx is not None:
                # Enhance existing system message
                formatted_messages[system_idx]["content"] += f"\n\nData context: {conversation.context_summary}"
            else:
                # Add as a new system message
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": f"Data context: {conversation.context_summary}"
                })
        
        return formatted_messages
    
    def _compress_conversation_if_needed(self, conversation: Conversation) -> None:
        """
        Compress conversation history if it exceeds the maximum length.
        
        Args:
            conversation: The conversation to compress
        """
        if len(conversation.messages) <= self.max_conversation_messages:
            return
        
        logger.info(f"Compressing conversation {conversation.conversation_id} with {len(conversation.messages)} messages")
        
        # Extract messages to compress (leave recent ones)
        messages_to_compress = conversation.messages[1:-self.max_conversation_messages]  # Skip system message
        
        if not messages_to_compress:
            return
            
        # Format messages for summarization
        messages_text = ""
        for msg in messages_to_compress:
            role = "User" if msg.role == "user" else "Assistant"
            messages_text += f"\n\n{role}: {msg.content}"
            
        # Create summarization prompt
        system_prompt = (
            "Your task is to compress the following conversation history into a concise summary "
            "that captures all important information, context, and key points discussed. "
            "Focus especially on insights about data structure, relationships, and conclusions reached. "
            "The summary will be used as context for continuing the conversation, so include essential details."
        )
        
        user_prompt = f"Please compress this conversation history:\n{messages_text}"
        
        # Generate compression
        try:
            summary_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            compressed = self.llm_client.chat_completion(
                messages=summary_messages,
                temperature=0.3  # Lower temperature for more consistent summarization
            )
            
            # Store the compressed context
            conversation.compressed_context = (
                f"This is a summary of the conversation history: {compressed}"
            )
            
            # Remove the compressed messages
            conversation.messages = [conversation.messages[0]] + conversation.messages[-self.max_conversation_messages:]
            
            logger.info(f"Successfully compressed conversation {conversation.conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to compress conversation: {str(e)}")
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """
        Save a conversation to disk.
        
        Args:
            conversation: The conversation to save
        """
        try:
            # Convert to dictionary
            conv_dict = asdict(conversation)
            
            # Save to file
            file_path = os.path.join(self.storage_dir, f"{conversation.conversation_id}.json")
            with open(file_path, 'w') as f:
                json.dump(conv_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.conversation_id}: {str(e)}")
    
    def _load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation from disk.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Loaded conversation or None if not found
        """
        file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                conv_dict = json.load(f)
                
            # Convert messages to proper objects
            messages = []
            for msg_dict in conv_dict.get("messages", []):
                messages.append(Message(
                    role=msg_dict["role"],
                    content=msg_dict["content"],
                    timestamp=msg_dict["timestamp"],
                    message_id=msg_dict["message_id"],
                    metadata=msg_dict.get("metadata", {})
                ))
                
            # Create conversation object
            conversation = Conversation(
                conversation_id=conv_dict["conversation_id"],
                title=conv_dict["title"],
                created_at=conv_dict["created_at"],
                updated_at=conv_dict["updated_at"],
                messages=messages,
                metadata=conv_dict.get("metadata", {}),
                data_files=conv_dict.get("data_files", []),
                compressed_context=conv_dict.get("compressed_context"),
                context_summary=conv_dict.get("context_summary")
            )
            
            # Add to in-memory store
            self.conversations[conversation_id] = conversation
            
            return conversation
                
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return None
