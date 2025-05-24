"""
LMStudio Client for connecting to and interacting with local LLMs.

This module provides a client for connecting to LLMs running in LM Studio,
with specific optimizations for handling large data analysis contexts.
"""
import os
import json
import time
import requests
import tiktoken
from typing import Dict, List, Optional, Union, Generator, Callable, Any

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor

logger = get_logger(__name__)

class LMStudioClient:
    """
    Client for interacting with LLMs running in LM Studio.
    
    Optimized for:
    - Connecting to locally hosted models like CodeLlama-34B
    - Large context windows for data analysis
    - Streaming responses for better UX
    - Memory-efficient prompt management
    - Context window optimization
    """
    
    def __init__(self, 
                base_url: str = "http://127.0.0.1:1234/v1", 
                model: str = "claude-3.7-sonnet-reasoning-gemma3-12b",
                timeout: int = 300,  # Increased from 30 to 300 seconds for complex analysis
                max_retries: int = 3,
                max_tokens: int = 4096,
                context_window: int = 8192,
                memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the LM Studio client.
        
        Args:
            base_url: Base URL for the LM Studio API
            model: Model identifier
            timeout: Timeout for requests (default 300s for complex analysis)
            max_retries: Maximum number of retries for requests
            max_tokens: Maximum response tokens
            context_window: Model's context window size
            memory_monitor: Memory monitor for tracking resource usage
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
        # Initialize tokenizer for token counting
        # Using cl100k_base as it's suitable for many recent models
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Could not load cl100k_base tokenizer. Using fallback token estimation.")
            self.tokenizer = None
        
        logger.info(f"Initialized LMStudioClient for {model} with {context_window} token context window and {timeout}s timeout")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation (rough approximation)
            return len(text) // 4
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to LM Studio server.
        
        Returns:
            Response from the server
        """
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully connected to LM Studio. Available models: {data}")
                return data
            else:
                logger.error(f"Failed to connect to LM Studio. Status code: {response.status_code}")
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to LM Studio: {str(e)}")
            return {"error": str(e)}
    
    def generate_completion(self, 
                          prompt: str,
                          temperature: float = 0.7,
                          max_tokens: Optional[int] = None,
                          stream: bool = False,
                          callback: Optional[Callable[[str], None]] = None,
                          timeout: Optional[int] = None) -> Union[str, Generator[str, None, None]]:
        """
        Generate a completion from the model.
        
        Args:
            prompt: The prompt to generate from
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Function to call with each chunk when streaming
            timeout: Timeout for the request in seconds
            
        Returns:
            Generated text or a generator yielding chunks
        """
        # Track memory usage
        self.memory_monitor.start_tracking_step("llm_generation")
        
        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.timeout
        
        # Count tokens to ensure we don't exceed context window
        prompt_tokens = self.count_tokens(prompt)
        logger.info(f"Prompt has {prompt_tokens} tokens")
        
        if prompt_tokens > self.context_window - 100:  # Leave some room for response
            logger.warning(f"Prompt too long ({prompt_tokens} tokens). Truncating...")
            # Truncate the prompt by approximating token count from characters
            # This is a simple approach - a more sophisticated one would truncate by tokens
            chars_per_token = len(prompt) / prompt_tokens
            target_chars = int((self.context_window - 200) * chars_per_token)
            prompt = prompt[-target_chars:]
            logger.info(f"Truncated prompt to approximately {self.context_window - 200} tokens")
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream
        }
        
        # Execute the request
        try:
            if stream:
                return self._stream_completion(payload, headers, callback, request_timeout)
            else:
                return self._standard_completion(payload, headers, request_timeout)
        finally:
            # End memory tracking
            self.memory_monitor.end_tracking_step("llm_generation")
    
    def _standard_completion(self, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 300) -> str:
        """
        Generate a completion without streaming.
        
        Args:
            payload: Request payload
            headers: Request headers
            timeout: Request timeout in seconds (default 300s for complex analysis)
            
        Returns:
            Generated text
        """
        # Check available memory and adjust timeout if needed
        memory_report = self.memory_monitor.get_memory_report()
        available_memory_mb = memory_report.get("available_mb", 0)
        
        # For complex analysis with large datasets, allow longer processing if memory permits
        if available_memory_mb > 8000:  # If more than 8GB available
            timeout = max(timeout, 600)  # Allow up to 10 minutes
            logger.info(f"High memory available ({available_memory_mb:.1f}MB), extending timeout to {timeout}s")
        elif available_memory_mb > 4000:  # If more than 4GB available
            timeout = max(timeout, 450)  # Allow up to 7.5 minutes
        
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("text", "")
                logger.debug(f"Generated {self.count_tokens(text)} tokens of response")
                return text
            else:
                error_msg = f"Error from LM Studio: Status {response.status_code}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
        except requests.exceptions.Timeout:
            logger.error(f"Request to LM Studio timed out after {timeout} seconds")
            return f"Error: Complex analysis took too long (>{timeout}s). Try breaking down the request or check if more memory is available."
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return f"Error: {str(e)}"
    
    def _stream_completion(self, 
                          payload: Dict[str, Any], 
                          headers: Dict[str, str],
                          callback: Optional[Callable[[str], None]] = None,
                          timeout: int = 300) -> Generator[str, None, None]:
        """
        Stream a completion from the model.
        
        Args:
            payload: Request payload
            headers: Request headers
            callback: Function to call with each chunk
            timeout: Request timeout in seconds (default 300s for complex analysis)
            
        Yields:
            Chunks of generated text
        """
        # Check available memory and adjust timeout if needed
        memory_report = self.memory_monitor.get_memory_report()
        available_memory_mb = memory_report.get("available_mb", 0)
        
        # For complex analysis with large datasets, allow longer processing if memory permits
        if available_memory_mb > 8000:  # If more than 8GB available
            timeout = max(timeout, 600)  # Allow up to 10 minutes
            logger.info(f"High memory available ({available_memory_mb:.1f}MB), extending streaming timeout to {timeout}s")
        elif available_memory_mb > 4000:  # If more than 4GB available
            timeout = max(timeout, 450)  # Allow up to 7.5 minutes
        
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout
            )
            
            if response.status_code != 200:
                error_msg = f"Error from LM Studio: Status {response.status_code}"
                logger.error(error_msg)
                yield f"Error: {error_msg}"
                return
                
            # Process the streamed response
            for line in response.iter_lines():
                if line:
                    # Parse the line
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data_str = line_text[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == "[DONE]":
                                break
                            
                            data = json.loads(data_str)
                            chunk = data.get("choices", [{}])[0].get("text", "")
                            
                            # Call the callback if provided
                            if callback and chunk:
                                callback(chunk)
                                
                            yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse streaming response line: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing streaming response: {str(e)}")
                        continue
                        
        except requests.exceptions.Timeout:
            logger.error(f"Streaming request to LM Studio timed out after {timeout} seconds")
            yield f"Error: Complex analysis took too long (>{timeout}s). Try breaking down the request or check if more memory is available."
            
        except Exception as e:
            logger.error(f"Error streaming completion: {str(e)}")
            yield f"Error: {str(e)}"
    
    def chat_completion(self,
                      messages: List[Dict[str, str]],
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      stream: bool = False,
                      callback: Optional[Callable[[str], None]] = None,
                      timeout: Optional[int] = None) -> Union[str, Generator[str, None, None]]:
        """
        Generate a chat completion using the provided messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            callback: Function to call with each chunk when streaming
            timeout: Timeout for the request in seconds
            
        Returns:
            Generated text or a generator yielding chunks
        """
        # For LM Studio local models, we convert chat format to text prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        # Generate using the text completion endpoint
        return self.generate_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            callback=callback,
            timeout=timeout
        )
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a single text prompt.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted text prompt
        """
        # Basic formatting for CodeLlama instruction format
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                if prompt_parts and prompt_parts[-1].endswith("[/INST]"):
                    # Start a new instruction
                    prompt_parts.append(f"<s>[INST] {content} [/INST]")
                else:
                    # First user message or after assistant
                    prompt_parts.append(f"{content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f"{content}</s>")
        
        # Ensure the prompt ends with an open instruction if the last message was from user
        if messages and messages[-1].get("role", "") == "user":
            if not prompt_parts[-1].endswith("[/INST]"):
                prompt_parts[-1] += " [/INST]"
        
        # Join all parts
        prompt = "".join(prompt_parts)
        
        # Ensure proper format start
        if not prompt.startswith("<s>"):
            prompt = "<s>" + prompt
            
        return prompt
