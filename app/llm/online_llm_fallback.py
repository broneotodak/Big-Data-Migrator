"""
Online LLM fallback for complex data tasks requiring advanced capabilities.

This module provides a fallback mechanism to more powerful online LLMs
when the local LLM needs assistance with complex data tasks.
"""
import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor

logger = get_logger(__name__)

@dataclass
class OnlineLLMConfig:
    """Configuration for online LLM services."""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    timeout: int = 60
    max_tokens: int = 1000

class OnlineLLMFallback:
    """
    Fallback to online LLMs for complex schema optimization and relationship detection.
    
    This class:
    - Handles complex schema optimization
    - Performs advanced relationship detection
    - Provides final validation and recommendations
    - Falls back to online LLMs when local models aren't sufficient
    """
    
    def __init__(self, 
                config: OnlineLLMConfig,
                memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the online LLM fallback.
        
        Args:
            config: Configuration for online LLM service
            memory_monitor: Optional memory monitor instance
        """
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        logger.info(f"OnlineLLMFallback initialized with model {config.model}")
    
    def optimize_schema(self, 
                      file_contexts: Dict[str, Any],
                      relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform advanced schema optimization using online LLM.
        
        Args:
            file_contexts: File context information
            relationships: Detected relationships
            
        Returns:
            Dictionary with optimized schema
        """
        self.memory_monitor.start_tracking_step("online_llm_schema_optimization")
        
        try:
            # Create a system prompt for schema optimization
            system_prompt = (
                "You are a database schema optimization expert. "
                "Analyze the provided data files and their relationships to recommend an optimized database schema. "
                "Include table definitions, column data types, primary and foreign keys, and any necessary indexes. "
                "Focus on normalization, efficient queries, and data integrity."
            )
            
            # Format data file information
            files_info = []
            for file_path, context in file_contexts.items():
                file_name = os.path.basename(file_path)
                columns = []
                
                for col_name in context.get("column_names", []):
                    col_type = context.get("column_types", {}).get(col_name, "unknown")
                    col_desc = f"{col_name} ({col_type})"
                    columns.append(col_desc)
                
                file_info = f"- {file_name}: {len(columns)} columns [{', '.join(columns[:10])}]"
                if len(columns) > 10:
                    file_info += f" and {len(columns) - 10} more columns"
                
                files_info.append(file_info)
            
            # Format relationships
            rels_info = []
            for rel in relationships:
                source_file = os.path.basename(rel.get("source_file", ""))
                target_file = os.path.basename(rel.get("target_file", ""))
                source_col = rel.get("source_column", "")
                target_col = rel.get("target_column", "")
                rel_type = rel.get("relationship_type", "")
                confidence = rel.get("confidence", 0)
                
                rel_info = (
                    f"- {rel_type} relationship between {source_file}.{source_col} and "
                    f"{target_file}.{target_col} (confidence: {confidence:.2f})"
                )
                rels_info.append(rel_info)
            
            # Create user prompt
            user_prompt = f"""
            Please optimize the database schema for the following files and detected relationships:
            
            Data Files:
            {os.linesep.join(files_info)}
            
            Detected Relationships:
            {os.linesep.join(rels_info) if rels_info else "No reliable relationships detected."}
            
            Provide the following in your response:
            1. Optimized table definitions with proper data types
            2. Primary and foreign key constraints
            3. Recommended indexes
            4. Normalization recommendations
            5. Any data quality improvements needed
            
            Format your response as a JSON object with these sections.
            """
            
            # Call online LLM
            response = self._call_online_llm(system_prompt, user_prompt)
            
            # Parse the JSON response
            try:
                # The response might contain explanatory text outside the JSON
                if response.strip().startswith('{') and response.strip().endswith('}'):
                    schema = json.loads(response)
                else:
                    # Try to extract JSON from the response
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx + 1]
                        schema = json.loads(json_str)
                    else:
                        logger.error("Could not extract JSON from online LLM response")
                        schema = {"error": "Could not parse schema from response", "raw_response": response}
            except json.JSONDecodeError:
                logger.error("Error parsing JSON from online LLM response")
                schema = {"error": "Could not parse schema from response", "raw_response": response}
            
            return schema
            
        finally:
            self.memory_monitor.end_tracking_step("online_llm_schema_optimization")
    
    def detect_advanced_relationships(self, file_contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect complex relationships between data files using online LLM.
        
        Args:
            file_contexts: File context information
            
        Returns:
            Dictionary with detected relationships
        """
        self.memory_monitor.start_tracking_step("online_llm_relationship_detection")
        
        try:
            # Create a system prompt for relationship detection
            system_prompt = (
                "You are an expert in database relationships and data modeling. "
                "Analyze the provided data files to detect potential relationships beyond simple key matches. "
                "Look for semantic relationships, hierarchical structures, and complex dependencies. "
                "Focus on identifying relationships that would be valuable for data modeling."
            )
            
            # Format data file information with sample data
            files_info = []
            for file_path, context in file_contexts.items():
                file_name = os.path.basename(file_path)
                sample_data = context.get("sample_data", [])
                
                file_info = f"== {file_name} ==\nColumns: {', '.join(context.get('column_names', []))}"
                
                # Add sample data
                if sample_data:
                    file_info += "\nSample data (first 3 rows):"
                    for i, row in enumerate(sample_data[:3]):
                        row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                        file_info += f"\nRow {i+1}: {row_str}"
                
                files_info.append(file_info)
            
            # Create user prompt
            user_prompt = f"""
            Please analyze these data files to detect potential relationships between them:
            
            {os.linesep.join(files_info)}
            
            Look for:
            1. Direct key relationships (primary/foreign keys)
            2. Semantic relationships (fields with similar meanings)
            3. Hierarchical relationships (parent-child)
            4. Temporal relationships (time-based dependencies)
            5. Logical groupings
            
            Provide the following in your response:
            1. A JSON array of detected relationships with:
               - source_file: The name of the source file
               - target_file: The name of the target file
               - source_column: The column in the source file
               - target_column: The column in the target file
               - relationship_type: The type of relationship (e.g., "one-to-one", "one-to-many", etc.)
               - confidence: Your confidence score (0.0-1.0)
               - description: A brief description of the relationship
            2. A brief explanation of your findings
            
            Format your response as a JSON object with "relationships" and "explanation" keys.
            """
            
            # Call online LLM
            response = self._call_online_llm(system_prompt, user_prompt)
            
            # Parse the JSON response
            try:
                # Try to extract JSON from the response
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    result = json.loads(json_str)
                else:
                    logger.error("Could not extract JSON from online LLM response")
                    result = {"error": "Could not parse relationships from response", "raw_response": response}
            except json.JSONDecodeError:
                logger.error("Error parsing JSON from online LLM response")
                result = {"error": "Could not parse relationships from response", "raw_response": response}
            
            return result
            
        finally:
            self.memory_monitor.end_tracking_step("online_llm_relationship_detection")
    
    def validate_and_recommend(self, 
                            data_contexts: Dict[str, Any],
                            analysis_goal: str) -> Dict[str, Any]:
        """
        Validate datasets and provide advanced recommendations.
        
        Args:
            data_contexts: Data context information
            analysis_goal: Description of analysis goal
            
        Returns:
            Dictionary with validation results and recommendations
        """
        self.memory_monitor.start_tracking_step("online_llm_validation")
        
        try:
            # Create a system prompt for validation and recommendations
            system_prompt = (
                "You are a data science expert providing validation and recommendations for data analysis. "
                "Given the data context and analysis goals, validate the suitability of the data "
                "and provide actionable recommendations for successful analysis."
            )
            
            # Extract context summary
            context_summary = data_contexts.get("summary", "")
            
            # Extract quality issues
            quality_issues = []
            for file_path, file_info in data_contexts.get("files", {}).items():
                file_name = os.path.basename(file_path)
                issues = file_info.get("data_quality", {}).get("issues", [])
                
                if issues:
                    quality_issues.append(f"- {file_name}: {', '.join(issues)}")
            
            # Create user prompt
            user_prompt = f"""
            Please validate the suitability of the following data for this analysis goal: "{analysis_goal}"
            
            Data Context Summary:
            {context_summary}
            
            Data Quality Issues:
            {os.linesep.join(quality_issues) if quality_issues else "No major quality issues detected."}
            
            Provide:
            1. An assessment of data suitability for the stated goal
            2. Advanced recommendations for data preparation
            3. Suggested analysis approaches
            4. Potential limitations or challenges
            5. Additional data that might be needed
            
            Format your response as a JSON object with these sections.
            """
            
            # Call online LLM
            response = self._call_online_llm(system_prompt, user_prompt)
            
            # Parse the JSON response
            try:
                # Try to extract JSON from the response
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    result = json.loads(json_str)
                else:
                    logger.error("Could not extract JSON from online LLM response")
                    result = {"error": "Could not parse validation from response", "raw_response": response}
            except json.JSONDecodeError:
                logger.error("Error parsing JSON from online LLM response")
                result = {"error": "Could not parse validation from response", "raw_response": response}
            
            return result
            
        finally:
            self.memory_monitor.end_tracking_step("online_llm_validation")
    
    def _call_online_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the online LLM service.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            LLM response as string
        """
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.2  # Lower temperature for more consistent responses
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"Error from online LLM API: {response.status_code} {response.text}")
                return f"Error: Could not get response from online LLM (Status {response.status_code})"
        
        except requests.exceptions.Timeout:
            logger.error("Online LLM API request timed out")
            return "Error: Online LLM request timed out"
            
        except Exception as e:
            logger.error(f"Error calling online LLM: {str(e)}")
            return f"Error: {str(e)}"
