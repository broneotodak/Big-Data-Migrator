"""
Resource optimization utilities for efficiently processing large datasets.
"""
import os
import gc
import logging
import tempfile
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.memory.file_size_calculator import FileSizeCalculator

logger = get_logger(__name__)

class ResourceOptimizer:
    """
    Implements resource optimization strategies for processing large datasets.
    
    This class provides chunk-based processing for large files,
    memory-efficient data structures, streaming data processing capabilities,
    and smart caching with memory limits.
    """
    
    def __init__(self, 
                memory_monitor: Optional[MemoryMonitor] = None,
                file_calculator: Optional[FileSizeCalculator] = None,
                cache_dir: Optional[str] = None,
                max_cache_size_mb: float = 1000.0):
        """
        Initialize the resource optimizer.
        
        Args:
            memory_monitor: Optional MemoryMonitor instance
            file_calculator: Optional FileSizeCalculator instance
            cache_dir: Directory to use for caching (temp dir if None)
            max_cache_size_mb: Maximum size of cache in MB
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.file_calculator = file_calculator or FileSizeCalculator(self.memory_monitor)
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.max_cache_size_mb = max_cache_size_mb
        self._cache = {}  # In-memory cache
        self._cache_size = 0.0  # Current cache size in MB
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_optimal_chunk_size(self, file_path: str, target_memory_usage_mb: Optional[float] = None) -> int:
        """
        Get optimal chunk size for processing a file.
        
        Args:
            file_path: Path to the file to analyze
            target_memory_usage_mb: Target memory usage for chunks (uses available if None)
            
        Returns:
            Optimal chunk size in number of rows
        """
        return self.file_calculator.calculate_optimal_chunk_size(
            file_path=file_path,
            target_memory_usage_mb=target_memory_usage_mb
        )

    def process_large_file_in_chunks(self, 
                                    file_path: str, 
                                    process_chunk_fn: Callable[[pd.DataFrame], Any],
                                    chunk_size: Optional[int] = None,
                                    show_progress: bool = True) -> List[Any]:
        """
        Process a large file in chunks to minimize memory usage.
        
        Args:
            file_path: Path to the file
            process_chunk_fn: Function to process each chunk
            chunk_size: Number of rows per chunk (auto-calculated if None)
            show_progress: Whether to show a progress bar
            
        Returns:
            List of results from processing each chunk
        """
        # Start memory tracking
        self.memory_monitor.start_tracking_step(f"process_file_{os.path.basename(file_path)}")
        
        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            chunk_size = self.file_calculator.calculate_optimal_chunk_size(file_path)
            
        # Determine file type
        _, ext = os.path.splitext(file_path)
        file_type = ext.lstrip('.').lower()
        
        results = []
        total_chunks = 0
        
        try:
            # Process based on file type
            if file_type in ['csv', 'txt']:
                # Get total number of rows for progress bar
                if show_progress:
                    with open(file_path, 'r') as f:
                        total_rows = sum(1 for _ in f)
                        total_chunks = total_rows // chunk_size + 1
                
                # Process chunks
                reader = pd.read_csv(file_path, chunksize=chunk_size)
                for i, chunk in enumerate(tqdm(reader, total=total_chunks, disable=not show_progress)):
                    # Clean up before processing new chunk
                    if i > 0 and i % 10 == 0:  # Every 10 chunks
                        self.memory_monitor.cleanup_memory()
                    
                    # Process the chunk
                    result = process_chunk_fn(chunk)
                    results.append(result)
            
            elif file_type in ['excel', 'xlsx', 'xls']:
                # Excel requires a different approach - read in chunks manually
                xls = pd.ExcelFile(file_path)
                sheet_name = xls.sheet_names[0]  # Assume first sheet
                
                # Get total number of rows
                total_rows = xls.book.sheet_by_name(sheet_name).nrows - 1  # Exclude header
                if show_progress:
                    total_chunks = total_rows // chunk_size + 1
                
                # Process in chunks
                for i in tqdm(range(0, total_rows, chunk_size), total=total_chunks, disable=not show_progress):
                    # Calculate actual chunk size
                    end_row = min(i + chunk_size, total_rows)
                    
                    # Read chunk
                    chunk = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        skiprows=range(1, i + 1),  # Skip header + previous chunks
                        nrows=end_row - i
                    )
                    
                    # Clean up before processing new chunk
                    if i > 0 and i % 5 == 0:  # Every 5 chunks for Excel (more memory intensive)
                        self.memory_monitor.cleanup_memory()
                    
                    # Process the chunk
                    result = process_chunk_fn(chunk)
                    results.append(result)
            
            elif file_type == 'parquet':
                # Use dask for parquet files
                ddf = dd.read_parquet(file_path)
                total_rows = len(ddf)
                
                if show_progress:
                    total_chunks = int(np.ceil(total_rows / chunk_size))
                    
                # Process in chunks
                for i in tqdm(range(0, total_rows, chunk_size), total=total_chunks, disable=not show_progress):
                    end_i = min(i + chunk_size, total_rows)
                    chunk = ddf.iloc[i:end_i].compute()
                    
                    # Clean up
                    if i > 0 and i % 10 == 0:
                        self.memory_monitor.cleanup_memory()
                    
                    # Process the chunk
                    result = process_chunk_fn(chunk)
                    results.append(result)
                    
            else:
                logger.warning(f"Unsupported file type for chunked processing: {file_type}")
                # Try reading the whole file if it's not too large
                df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)
                result = process_chunk_fn(df)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path} in chunks: {str(e)}")
            raise
        
        finally:
            # Stop memory tracking
            self.memory_monitor.end_tracking_step(f"process_file_{os.path.basename(file_path)}")
            
        return results
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize a pandas DataFrame to reduce memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
        
        # Start memory tracking
        self.memory_monitor.start_tracking_step("optimize_dataframe")
        
        # Make a copy of the input DataFrame
        result = df.copy()
        
        # Optimize numeric columns
        for col in result.select_dtypes(include=['int']).columns:
            col_min, col_max = result[col].min(), result[col].max()
            
            # Find the smallest integer type that can represent the data
            if col_min >= 0:
                if col_max < 256:
                    result[col] = result[col].astype(np.uint8)
                elif col_max < 65536:
                    result[col] = result[col].astype(np.uint16)
                elif col_max < 4294967296:
                    result[col] = result[col].astype(np.uint32)
                else:
                    result[col] = result[col].astype(np.uint64)
            else:
                if col_min > -128 and col_max < 128:
                    result[col] = result[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32768:
                    result[col] = result[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483648:
                    result[col] = result[col].astype(np.int32)
                else:
                    result[col] = result[col].astype(np.int64)
        
        # Optimize float columns
        for col in result.select_dtypes(include=['float']).columns:
            result[col] = result[col].astype(np.float32)
        
        # Optimize object (string) columns by using categories for repeated values
        for col in result.select_dtypes(include=['object']).columns:
            # If column has few unique values relative to total, convert to category
            unique_count = result[col].nunique()
            if unique_count / len(result) < 0.5:  # 50% unique threshold
                result[col] = result[col].astype('category')
        
        # End memory tracking
        self.memory_monitor.end_tracking_step("optimize_dataframe")
        
        # Report memory savings
        end_mem = result.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
        
        return result
    
    def stream_large_dataframe(self, df: Union[pd.DataFrame, str]) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a large DataFrame in chunks to process it without loading it all in memory.
        
        Args:
            df: Either a DataFrame or a path to a file
            
        Yields:
            Chunks of the DataFrame
        """
        # Start memory tracking
        if isinstance(df, str):
            operation_name = f"stream_file_{os.path.basename(df)}"
        else:
            operation_name = "stream_dataframe"
        
        self.memory_monitor.start_tracking_step(operation_name)
        
        try:
            # Calculate optimal chunk size based on available memory
            _, usage_percent, total_ram_mb = self.memory_monitor.get_memory_usage()
            available_ram_mb = total_ram_mb * (1 - usage_percent) * 0.3  # Use 30% of available RAM
            
            if isinstance(df, str):
                # It's a file path
                file_path = df
                _, ext = os.path.splitext(file_path)
                file_type = ext.lstrip('.').lower()
                
                # Determine optimal chunk size
                chunk_size = self.file_calculator.calculate_optimal_chunk_size(
                    file_path, 
                    target_memory_usage_mb=available_ram_mb
                )
                
                # Stream from file
                if file_type in ['csv', 'txt']:
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                        yield chunk
                        gc.collect()  # Clean up after yielding
                
                elif file_type in ['excel', 'xlsx', 'xls']:
                    xls = pd.ExcelFile(file_path)
                    sheet_name = xls.sheet_names[0]  # Assume first sheet
                    
                    # Get total rows
                    total_rows = xls.book.sheet_by_name(sheet_name).nrows - 1  # Exclude header
                    
                    # Process in chunks
                    for i in range(0, total_rows, chunk_size):
                        end_row = min(i + chunk_size, total_rows)
                        chunk = pd.read_excel(
                            file_path,
                            sheet_name=sheet_name,
                            skiprows=range(1, i + 1),  # Skip header + previous chunks
                            nrows=end_row - i
                        )
                        yield chunk
                        gc.collect()  # Clean up after yielding
                
                elif file_type == 'parquet':
                    ddf = dd.read_parquet(file_path)
                    total_rows = len(ddf)
                    
                    # Process in chunks
                    for i in range(0, total_rows, chunk_size):
                        end_i = min(i + chunk_size, total_rows)
                        chunk = ddf.iloc[i:end_i].compute()
                        yield chunk
                        gc.collect()  # Clean up after yielding
                
                else:
                    logger.warning(f"Unsupported file type for streaming: {file_type}")
                    # Try reading the whole file as last resort
                    df = pd.read_csv(file_path) if file_type == 'csv' else pd.read_excel(file_path)
                    yield df
                    
            else:
                # It's already a DataFrame - chunk it
                total_rows = len(df)
                
                # Estimate memory per row
                sample_size = min(1000, total_rows)
                if sample_size > 0:
                    sample = df.iloc[:sample_size]
                    mem_per_row_bytes = sample.memory_usage(deep=True).sum() / sample_size
                    mem_per_row_mb = mem_per_row_bytes / (1024 * 1024)
                    
                    # Calculate optimal chunk size
                    chunk_size = max(100, int(available_ram_mb / mem_per_row_mb))
                else:
                    chunk_size = 10000  # Default
                
                # Yield chunks
                for i in range(0, total_rows, chunk_size):
                    end_i = min(i + chunk_size, total_rows)
                    yield df.iloc[i:end_i]
                    
                    # Clean up every few chunks
                    if i > 0 and i % (chunk_size * 5) == 0:
                        gc.collect()
        
        finally:
            # End memory tracking
            self.memory_monitor.end_tracking_step(operation_name)
    
    def cache_result(self, key: str, data: Any, estimated_size_mb: Optional[float] = None) -> bool:
        """
        Cache a result with smart memory management.
        
        Args:
            key: Unique key for the cached item
            data: Data to cache
            estimated_size_mb: Estimated size in MB, calculated if not provided
            
        Returns:
            Whether caching was successful
        """
        # Calculate size if not provided
        if estimated_size_mb is None:
            if isinstance(data, pd.DataFrame):
                estimated_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            else:
                # Rough estimate for other types
                import sys
                estimated_size_mb = sys.getsizeof(data) / (1024 * 1024)
        
        # Check if we need to free space
        if self._cache_size + estimated_size_mb > self.max_cache_size_mb:
            self._free_cache_space(estimated_size_mb)
            
        # Store in cache if we have space
        if self._cache_size + estimated_size_mb <= self.max_cache_size_mb:
            self._cache[key] = {
                'data': data,
                'size_mb': estimated_size_mb,
                'last_accessed': pd.Timestamp.now()
            }
            self._cache_size += estimated_size_mb
            logger.debug(f"Cached item '{key}' ({estimated_size_mb:.2f} MB), total cache: {self._cache_size:.2f} MB")
            return True
        else:
            logger.warning(f"Could not cache item '{key}' ({estimated_size_mb:.2f} MB), insufficient cache space")
            return False
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached result.
        
        Args:
            key: Unique key for the cached item
            
        Returns:
            Cached data if found, None otherwise
        """
        if key in self._cache:
            # Update last accessed time
            self._cache[key]['last_accessed'] = pd.Timestamp.now()
            logger.debug(f"Cache hit for '{key}'")
            return self._cache[key]['data']
        else:
            logger.debug(f"Cache miss for '{key}'")
            return None
    
    def _free_cache_space(self, needed_space_mb: float) -> None:
        """
        Free up cache space by removing least recently used items.
        
        Args:
            needed_space_mb: Space needed in MB
        """
        if not self._cache:
            return
            
        # Sort cache items by last accessed time (oldest first)
        sorted_items = sorted(
            self._cache.items(), 
            key=lambda x: x[1]['last_accessed']
        )
        
        space_freed = 0.0
        items_to_remove = []
        
        # Identify items to remove until we free enough space
        for key, item in sorted_items:
            items_to_remove.append(key)
            space_freed += item['size_mb']
            if space_freed >= needed_space_mb:
                break
                
        # Remove the identified items
        for key in items_to_remove:
            freed_mb = self._cache[key]['size_mb']
            del self._cache[key]
            self._cache_size -= freed_mb
            logger.debug(f"Removed '{key}' from cache, freed {freed_mb:.2f} MB")
            
        # Force garbage collection
        gc.collect()
        
    def clear_cache(self) -> float:
        """
        Clear all cached items.
        
        Returns:
            Amount of memory freed in MB
        """
        freed_mb = self._cache_size
        self._cache.clear()
        self._cache_size = 0.0
        gc.collect()
        logger.info(f"Cache cleared, freed {freed_mb:.2f} MB")
        return freed_mb