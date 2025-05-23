"""
File size calculation utilities for determining optimal processing parameters.
"""
import os
import math
import logging
from typing import Dict, List, Optional, Tuple

import psutil
import pandas as pd

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor

logger = get_logger(__name__)

class FileSizeCalculator:
    """
    Calculates optimal file size limits based on system resources.
    
    This class provides recommendations for batch processing large files,
    calculates memory requirements for different file types, and helps
    determine chunk sizes for efficient processing.
    """
    
    # Default safety factor to avoid memory issues
    SAFETY_FACTOR = 0.7
    
    # File type memory factors (multiplier over raw file size)
    FILE_TYPE_FACTORS = {
        "csv": 5.0,
        "excel": 6.0,  # Excel files need more overhead
        "json": 4.5,
        "parquet": 2.5,  # Parquet is more efficient
        "feather": 3.0,
        "pickle": 2.0,
        "hdf": 2.5,
        "sql": 4.0,
        "xml": 8.0,  # XML is very verbose
        "html": 7.0,
        "txt": 4.0,
        "pdf": 10.0,  # PDF extraction is memory intensive
        "docx": 8.0,  # Word docs require heavy processing
        "default": 5.0
    }
    
    # LLM context overhead in MB per 1K tokens
    LLM_CONTEXT_OVERHEAD_MB = {
        "small": 0.5,    # ~7B parameter models
        "medium": 1.0,   # ~13B parameter models
        "large": 2.0,    # ~34B parameter models
        "xlarge": 4.0,   # ~70B parameter models
        "default": 1.0
    }
    
    def __init__(self, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the file size calculator.
        
        Args:
            memory_monitor: Optional MemoryMonitor instance to use for memory checks
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
    def estimate_dataframe_memory(self, 
                                 rows: int, 
                                 columns: int, 
                                 dtypes: Optional[Dict[str, str]] = None) -> float:
        """
        Estimate memory usage of a pandas DataFrame with given dimensions.
        
        Args:
            rows: Number of rows
            columns: Number of columns
            dtypes: Optional dict of column data types
            
        Returns:
            Estimated memory usage in MB
        """
        # Default type sizes in bytes
        type_sizes = {
            'int': 8,
            'float': 8,
            'bool': 1,
            'datetime': 8,
            'category': 4,  # Depends on cardinality
            'object': 40,   # String average - rough estimate
            'default': 8
        }
        
        # Calculate base size
        if dtypes:
            # Use provided dtypes to get more accurate estimate
            total_bytes = 0
            for col, dtype in dtypes.items():
                dtype_key = 'default'
                for key in type_sizes:
                    if key in dtype.lower():
                        dtype_key = key
                        break
                total_bytes += rows * type_sizes[dtype_key]
        else:
            # Use average estimate
            total_bytes = rows * columns * type_sizes['default']
            
        # Add pandas overhead (index, etc.)
        total_bytes *= 1.2
            
        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        
        logger.debug(f"Estimated DataFrame memory: {total_mb:.2f} MB for {rows} rows, {columns} cols")
        return total_mb
    
    def estimate_file_memory_usage(self, 
                                  file_path: str, 
                                  file_type: Optional[str] = None) -> float:
        """
        Estimate memory required to process a specific file.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (inferred from extension if not provided)
            
        Returns:
            Estimated memory required in MB
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return 0.0
            
        # Get file size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Determine file type if not provided
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip('.').lower()
            
        # Get memory factor for this file type
        memory_factor = self.FILE_TYPE_FACTORS.get(file_type, self.FILE_TYPE_FACTORS["default"])
        
        # Calculate estimated memory
        estimated_memory_mb = file_size_mb * memory_factor
        
        logger.info(f"File {file_path} ({file_size_mb:.2f} MB) estimated to require {estimated_memory_mb:.2f} MB RAM")
        return estimated_memory_mb
    
    def calculate_optimal_chunk_size(self, 
                                    file_path: str, 
                                    file_type: Optional[str] = None,
                                    target_memory_usage_mb: Optional[float] = None) -> int:
        """
        Calculate optimal chunk size for processing a large file.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (inferred from extension if not provided)
            target_memory_usage_mb: Target memory usage per chunk in MB
            
        Returns:
            Optimal chunk size (number of rows)
        """
        # If target memory not specified, calculate based on available memory
        if target_memory_usage_mb is None:
            _, usage_percent, total_ram_mb = self.memory_monitor.get_memory_usage()
            available_ram_mb = total_ram_mb * (1 - usage_percent) * self.SAFETY_FACTOR
            target_memory_usage_mb = available_ram_mb * 0.3  # Use 30% of available RAM per chunk
        
        # Determine file type if not provided
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip('.').lower()
            
        # Sample file to get row size estimate
        try:
            if file_type in ['csv', 'txt']:
                # Read first 1000 rows to estimate
                df_sample = pd.read_csv(file_path, nrows=1000)
            elif file_type in ['excel', 'xlsx', 'xls']:
                df_sample = pd.read_excel(file_path, nrows=1000)
            elif file_type == 'json':
                df_sample = pd.read_json(file_path, lines=True, nrows=1000)
            elif file_type == 'parquet':
                df_sample = pd.read_parquet(file_path)
                df_sample = df_sample.head(1000)
            else:
                logger.warning(f"Unsupported file type for sampling: {file_type}")
                # Default to a conservative estimate
                return 10000
            
            # Get memory usage per row
            mem_usage = df_sample.memory_usage(deep=True).sum() / len(df_sample)
            mem_usage_mb = mem_usage / (1024 * 1024)
            
            # Calculate optimal chunk size
            chunk_size = max(100, int(target_memory_usage_mb / mem_usage_mb))
            
            logger.info(f"Optimal chunk size for {file_path}: {chunk_size} rows")
            return chunk_size
            
        except Exception as e:
            logger.warning(f"Error sampling file {file_path}: {str(e)}")
            # Return a safe default
            return 10000
    
    def calculate_batch_recommendations(self, 
                                       total_rows: int,
                                       memory_per_row_kb: float) -> Dict:
        """
        Calculate batch processing recommendations.
        
        Args:
            total_rows: Total number of rows to process
            memory_per_row_kb: Memory usage per row in KB
            
        Returns:
            Dict with batch processing recommendations
        """
        # Get available memory
        _, usage_percent, total_ram_mb = self.memory_monitor.get_memory_usage()
        available_ram_mb = total_ram_mb * (1 - usage_percent) * self.SAFETY_FACTOR
        
        # Calculate max rows per batch
        max_rows_per_batch = int(available_ram_mb * 1024 / memory_per_row_kb)
        
        # Calculate number of batches needed
        num_batches = math.ceil(total_rows / max_rows_per_batch)
        
        # Calculate actual rows per batch
        rows_per_batch = math.ceil(total_rows / num_batches)
        
        # Factor in pandas overhead
        adjusted_rows_per_batch = int(rows_per_batch * 0.8)  # 20% buffer
        adjusted_num_batches = math.ceil(total_rows / adjusted_rows_per_batch)
        
        recommendations = {
            "total_rows": total_rows,
            "available_memory_mb": available_ram_mb,
            "memory_per_row_kb": memory_per_row_kb,
            "recommended_rows_per_batch": adjusted_rows_per_batch,
            "recommended_num_batches": adjusted_num_batches,
            "memory_usage_per_batch_mb": adjusted_rows_per_batch * memory_per_row_kb / 1024,
            "percentage_of_available_memory": (adjusted_rows_per_batch * memory_per_row_kb / 1024) / available_ram_mb * 100
        }
        
        logger.info(f"Batch recommendations: {adjusted_num_batches} batches with {adjusted_rows_per_batch} rows each")
        return recommendations
    
    def calculate_llm_memory_requirements(self, 
                                         text_length: int, 
                                         model_size: str = "medium", 
                                         include_response: bool = True) -> float:
        """
        Calculate memory requirements for LLM processing.
        
        Args:
            text_length: Length of text in characters
            model_size: Size of the LLM model ("small", "medium", "large", "xlarge")
            include_response: Whether to include memory for response generation
            
        Returns:
            Estimated memory requirement in MB
        """
        # Rough estimate of tokens per character (varies by language)
        chars_per_token = 4.0
        
        # Estimate token count
        token_count = text_length / chars_per_token
        
        # Get memory overhead per 1K tokens for this model size
        memory_per_1k = self.LLM_CONTEXT_OVERHEAD_MB.get(model_size.lower(), 
                                                        self.LLM_CONTEXT_OVERHEAD_MB["default"])
        
        # Calculate base memory requirement
        memory_mb = (token_count / 1000) * memory_per_1k
        
        # Add response overhead if needed
        if include_response:
            # Assume response might be 20% of input size
            memory_mb *= 1.2
        
        logger.info(f"LLM memory requirement: {memory_mb:.2f} MB for {token_count:.0f} tokens ({model_size} model)")
        return memory_mb