"""
Anthropic Claude client for online LLM capabilities.

This module provides integration with Anthropic's Claude API for advanced
data analysis and conversation capabilities.
"""
import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union, Generator
from dataclasses import dataclass

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor

logger = get_logger(__name__)

@dataclass
class AnthropicConfig:
    """Configuration for Anthropic Claude API."""
    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000
    timeout: int = 60
    temperature: float = 0.2

class AnthropicClient:
    """
    Client for Anthropic Claude API integration.
    
    This class provides:
    - Direct Claude API integration
    - Streaming response support
    - Data analysis optimized prompts
    - Error handling and fallback
    """
    
    def __init__(self, 
                config: AnthropicConfig,
                memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            config: Anthropic API configuration
            memory_monitor: Optional memory monitor instance
        """
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.session = requests.Session()
        
        # Set up headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        })
        
        logger.info(f"AnthropicClient initialized with model {config.model}")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       system_prompt: Optional[str] = None,
                       stream: bool = False,
                       **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a chat completion using Claude.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Response string or generator for streaming
        """
        self.memory_monitor.start_tracking_step("anthropic_chat_completion")
        
        try:
            # Prepare the request data
            data = {
                "model": self.config.model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": self._format_messages(messages),
                "stream": stream
            }
            
            # Add system prompt if provided
            if system_prompt:
                data["system"] = system_prompt
            
            # Make the API request
            response = self.session.post(
                f"{self.config.base_url}/messages",
                json=data,
                timeout=self.config.timeout,
                stream=stream
            )
            
            if response.status_code != 200:
                error_msg = f"Anthropic API error: {response.status_code} {response.text}"
                logger.error(error_msg)
                if stream:
                    return self._error_generator(error_msg)
                else:
                    return f"Error: {error_msg}"
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_standard_response(response)
                
        except requests.exceptions.Timeout:
            error_msg = "Anthropic API request timed out"
            logger.error(error_msg)
            if stream:
                return self._error_generator(error_msg)
            else:
                return f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error calling Anthropic API: {str(e)}"
            logger.error(error_msg)
            if stream:
                return self._error_generator(error_msg)
            else:
                return f"Error: {error_msg}"
                
        finally:
            self.memory_monitor.end_tracking_step("anthropic_chat_completion")
    
    def analyze_data_structure(self, 
                             data_context: Dict[str, Any],
                             user_question: str) -> str:
        """
        Analyze data structure and answer user questions.
        
        Args:
            data_context: Data context information
            user_question: User's question about the data
            
        Returns:
            Analysis response
        """
        # Create a specialized system prompt for data analysis
        system_prompt = """You are an expert data analyst helping users understand their data. 
        
        Your role:
        - Analyze data structures, relationships, and patterns
        - Provide clear, actionable insights
        - Suggest data quality improvements
        - Recommend analysis approaches
        - Explain complex concepts in simple terms
        
        Guidelines:
        - Be concise but thorough
        - Use bullet points for clarity
        - Provide specific examples when possible
        - Focus on practical recommendations
        - Avoid generating code unless specifically requested"""
        
        # Format the data context
        context_summary = self._format_data_context(data_context)
        
        # Create the user message
        user_message = f"""Data Context:
{context_summary}

User Question: {user_question}

Please analyze the data and provide a helpful response."""
        
        messages = [{"role": "user", "content": user_message}]
        
        return self.chat_completion(messages, system_prompt)
    
    def optimize_schema(self, 
                       file_contexts: Dict[str, Any],
                       relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize database schema using Claude's analysis.
        
        Args:
            file_contexts: File context information
            relationships: Detected relationships
            
        Returns:
            Schema optimization results
        """
        system_prompt = """You are a database schema optimization expert. Analyze the provided data files and relationships to recommend an optimized database schema.

        Provide your response as a JSON object with these sections:
        - "tables": Array of table definitions with columns and data types
        - "relationships": Array of foreign key relationships
        - "indexes": Recommended indexes for performance
        - "recommendations": General optimization recommendations
        - "data_quality_notes": Any data quality issues to address"""
        
        # Format the input data
        files_info = []
        for file_path, context in file_contexts.items():
            file_name = os.path.basename(file_path)
            columns = context.get("column_names", [])
            column_types = context.get("column_types", {})
            
            file_info = {
                "file_name": file_name,
                "columns": [{"name": col, "type": column_types.get(col, "unknown")} for col in columns],
                "row_count": context.get("row_count", 0),
                "data_quality": context.get("data_quality", {})
            }
            files_info.append(file_info)
        
        user_message = f"""Please optimize the database schema for these data files:

Files: {json.dumps(files_info, indent=2)}

Relationships: {json.dumps(relationships, indent=2)}

Provide an optimized schema as a JSON response."""
        
        messages = [{"role": "user", "content": user_message}]
        response = self.chat_completion(messages, system_prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response if it contains other text
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)
            else:
                return {"error": "Could not parse JSON from response", "raw_response": response}
                
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON from response", "raw_response": response}
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to Anthropic API.
        
        Returns:
            Connection status
        """
        try:
            # Simple test request
            test_data = {
                "model": self.config.model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            response = self.session.post(
                f"{self.config.base_url}/messages",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "connected",
                    "model": self.config.model,
                    "provider": "anthropic"
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "provider": "anthropic"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "provider": "anthropic"
            }
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Anthropic API."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # System messages are handled separately in Anthropic
                continue
            elif role in ["user", "assistant"]:
                formatted.append({"role": role, "content": content})
            else:
                # Default unknown roles to user
                formatted.append({"role": "user", "content": content})
        
        return formatted
    
    def _format_data_context(self, data_context: Dict[str, Any]) -> str:
        """Format data context for Claude analysis."""
        context_parts = []
        
        # Add summary
        if "summary" in data_context:
            context_parts.append(f"Summary: {data_context['summary']}")
        
        # Add file details
        if "files" in data_context:
            context_parts.append("\nFiles:")
            for file_path, file_info in data_context["files"].items():
                file_name = os.path.basename(file_path)
                context_parts.append(f"- {file_name}:")
                context_parts.append(f"  Rows: {file_info.get('row_count', 'unknown')}")
                context_parts.append(f"  Columns: {file_info.get('column_count', 'unknown')}")
                
                if "column_names" in file_info:
                    cols = file_info["column_names"][:10]  # Limit to first 10
                    if len(file_info["column_names"]) > 10:
                        cols.append(f"... and {len(file_info['column_names']) - 10} more")
                    context_parts.append(f"  Column names: {', '.join(cols)}")
        
        # Add relationships
        if "relationships" in data_context and data_context["relationships"]:
            context_parts.append(f"\nRelationships: {len(data_context['relationships'])} detected")
        
        # Add recommendations
        if "recommendations" in data_context and data_context["recommendations"]:
            context_parts.append(f"\nRecommendations:")
            for rec in data_context["recommendations"][:3]:  # Limit to first 3
                context_parts.append(f"- {rec}")
        
        return "\n".join(context_parts)
    
    def _handle_standard_response(self, response: requests.Response) -> str:
        """Handle standard (non-streaming) response."""
        try:
            data = response.json()
            content = data.get("content", [])
            
            if content and len(content) > 0:
                return content[0].get("text", "No response content")
            else:
                return "No response content"
                
        except json.JSONDecodeError:
            return f"Error parsing response: {response.text}"
    
    def _handle_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Handle streaming response."""
        try:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if "text" in delta:
                                    yield delta["text"]
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"Error in streaming: {str(e)}"
    
    def _error_generator(self, error_msg: str) -> Generator[str, None, None]:
        """Generate error message for streaming responses."""
        yield error_msg 