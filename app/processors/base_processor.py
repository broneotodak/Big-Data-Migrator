"""
Abstract base class for all file processors in the system.
Provides common functionality for memory-safe file reading and processing.
"""
from abc import ABC, abstractmethod
import os
import time
import gc
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO, Iterator, Callable

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer

logger = get_logger(__name__)

class BaseProcessor(ABC):
    """
    Abstract base class for all file processors.
    
    This class provides common functionality for:
    1. Memory-safe file reading with chunking
    2. Progress tracking for large file operations
    3. Automatic memory cleanup after processing
    4. Error recovery and partial processing capabilities
    5. Real-time memory usage reporting
    """
    
    def __init__(self, 
                memory_monitor: Optional[MemoryMonitor] = None,
                resource_optimizer: Optional[ResourceOptimizer] = None,
                chunk_size_mb: float = 50.0,
                show_progress: bool = True):
        """
        Initialize the base processor with memory management components.
        
        Args:
            memory_monitor: MemoryMonitor instance for tracking memory usage
            resource_optimizer: ResourceOptimizer instance for optimizing memory usage
            chunk_size_mb: Default chunk size for file reading in MB
            show_progress: Whether to show progress bars
        """
        # Initialize memory management components
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.resource_optimizer = resource_optimizer or ResourceOptimizer(
            memory_monitor=self.memory_monitor
        )
        
        # Configuration settings
        self.chunk_size_mb = chunk_size_mb
        self.show_progress = show_progress
        self.default_dtypes = {
            'int': 'Int64',  # pandas nullable integer
            'float': 'float32',
            'str': 'string',  # pandas string array (more memory efficient)
            'bool': 'boolean',  # pandas boolean array
            'date': 'datetime64[ns]',
            'category': 'category',
        }
        
        # Statistics and results
        self.stats = {
            'processed_chunks': 0,
            'total_chunks': 0,
            'rows_processed': 0,
            'processing_time': 0,
            'memory_used_mb': 0,
            'memory_peak_mb': 0,
            'errors': [],
        }
    
    @abstractmethod
    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process the file at the given path.
        
        Args:
            file_path: Path to the file to process
            **kwargs: Additional processor-specific parameters
            
        Returns:
            Dictionary with processing results
        """
        pass
        
    @abstractmethod
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about the file without fully processing it.
        
        Args:
            file_path: Path to the file to get information for
            
        Returns:
            Dictionary with file information
        """
        pass
        
    def estimate_memory_requirement(self, file_path: str) -> float:
        """
        Estimate the memory requirement for processing the file.
        
        Args:
            file_path: Path to the file to estimate memory for
            
        Returns:
            Estimated memory requirement in MB
        """
        # Default implementation based on file size with safety factor
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb * 2.5  # Assume 2.5x memory requirement as a safety factor
    
    def get_optimal_chunk_size(self, file_path: str) -> float:
        """
        Get the optimal chunk size for processing the file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Optimal chunk size in MB
        """
        # Use resource optimizer to determine optimal chunk size
        available_memory = self.memory_monitor.get_available_memory_mb()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Default to a percentage of available memory, but not smaller than 10MB
        # or larger than the file itself
        chunk_size = min(
            max(10.0, available_memory * 0.3),  # At most 30% of available memory
            file_size_mb  # No larger than the file itself
        )
        
        # Use the resource optimizer to refine the chunk size
        optimal_chunk_size = self.resource_optimizer.get_optimal_chunk_size(
            file_size_mb=file_size_mb,
            initial_chunk_size_mb=chunk_size
        )
        
        logger.debug(f"Optimal chunk size for {file_path}: {optimal_chunk_size:.2f} MB")
        return optimal_chunk_size
    
    def check_memory_safety(self, required_mb: float) -> bool:
        """
        Check if there's enough memory to safely process a chunk.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            True if there's enough memory, False otherwise
        """
        available_mb = self.memory_monitor.get_available_memory_mb()
        is_safe = available_mb >= (required_mb * 1.2)  # 20% safety margin
        
        if not is_safe:
            logger.warning(f"Memory safety check failed: {available_mb:.2f} MB available, {required_mb:.2f} MB required")
            
        return is_safe
    
    def cleanup_memory(self):
        """
        Clean up memory after processing a chunk.
        """
        # Update memory usage statistics
        current_usage = self.memory_monitor.get_memory_usage_mb()
        self.stats['memory_used_mb'] = current_usage
        
        peak_usage = self.memory_monitor.get_peak_memory_usage()
        if peak_usage > self.stats['memory_peak_mb']:
            self.stats['memory_peak_mb'] = peak_usage
            
        # Force garbage collection
        gc.collect()
        
        # Log memory usage
        logger.debug(f"Memory after cleanup: {self.memory_monitor.get_memory_usage_mb():.2f} MB")
    
    def chunk_file(self, file_path: str, chunk_size_bytes: int) -> Iterator[bytes]:
        """
        Read a file in chunks to prevent loading the entire file into memory.
        
        Args:
            file_path: Path to the file to read
            chunk_size_bytes: Size of each chunk in bytes
            
        Yields:
            Chunks of the file as bytes
        """
        file_size = os.path.getsize(file_path)
        self.stats['total_chunks'] = max(1, file_size // chunk_size_bytes)
        
        with open(file_path, 'rb') as f:
            for _ in range(self.stats['total_chunks']):
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    break
                    
                self.stats['processed_chunks'] += 1
                yield chunk
                
                # Clean up memory after each chunk
                self.cleanup_memory()
    
    def detect_data_type(self, sample_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, str]:
        """
        Detect optimal data types for columns in a DataFrame or array.
        
        Args:
            sample_data: Sample data to determine types for
            
        Returns:
            Dictionary mapping column names to optimal data types
        """
        if not isinstance(sample_data, pd.DataFrame):
            # Convert to DataFrame if it's not already
            if isinstance(sample_data, np.ndarray):
                sample_data = pd.DataFrame(sample_data)
            else:
                return {}
        
        # Initialize type mapping
        optimized_types = {}
        
        for col in sample_data.columns:
            # Skip if column is already optimized
            if pd.api.types.is_integer_dtype(sample_data[col]):
                # Check if can be represented as smaller integer
                if sample_data[col].min() >= 0:
                    if sample_data[col].max() <= 255:
                        optimized_types[col] = 'uint8'
                    elif sample_data[col].max() <= 65535:
                        optimized_types[col] = 'uint16'
                    else:
                        optimized_types[col] = 'uint32'
                else:
                    if sample_data[col].min() >= -128 and sample_data[col].max() <= 127:
                        optimized_types[col] = 'int8'
                    elif sample_data[col].min() >= -32768 and sample_data[col].max() <= 32767:
                        optimized_types[col] = 'int16'
                    else:
                        optimized_types[col] = 'int32'
            
            elif pd.api.types.is_float_dtype(sample_data[col]):
                # Use float32 for most float columns to save memory
                optimized_types[col] = 'float32'
            
            elif pd.api.types.is_string_dtype(sample_data[col]) or pd.api.types.is_object_dtype(sample_data[col]):
                # Check if column has few unique values
                unique_count = sample_data[col].nunique()
                total_count = len(sample_data[col])
                
                if unique_count / total_count < 0.5 and unique_count <= 100:
                    optimized_types[col] = 'category'
                else:
                    optimized_types[col] = 'string'
            
            elif pd.api.types.is_datetime64_dtype(sample_data[col]):
                optimized_types[col] = 'datetime64[ns]'
            
            elif pd.api.types.is_bool_dtype(sample_data[col]):
                optimized_types[col] = 'boolean'
        
        return optimized_types
    
    def sample_data(self, data: Union[pd.DataFrame, np.ndarray], 
                   sample_size: int = 10000) -> Union[pd.DataFrame, np.ndarray]:
        """
        Generate a representative sample of data for analysis.
        
        Args:
            data: Data to sample from
            sample_size: Number of samples to take
            
        Returns:
            Sampled data
        """
        if isinstance(data, pd.DataFrame):
            total_rows = len(data)
            
            if total_rows <= sample_size:
                return data
            
            # Stratified sampling for better representation
            if total_rows > 100000:
                # First downsample to 100k for efficiency
                initial_sample = data.sample(n=100000, random_state=42)
                return initial_sample.sample(n=sample_size, random_state=42)
            else:
                return data.sample(n=sample_size, random_state=42)
        
        elif isinstance(data, np.ndarray):
            total_rows = data.shape[0]
            
            if total_rows <= sample_size:
                return data
            
            # Simple random sampling for numpy arrays
            indices = np.random.choice(total_rows, size=sample_size, replace=False)
            return data[indices]
        
        return data
    
    def profile_data(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate profile statistics for DataFrame columns.
        
        Args:
            data: DataFrame to profile
            
        Returns:
            Dictionary with column statistics
        """
        profile = {}
        
        # Safety check
        if not isinstance(data, pd.DataFrame) or len(data) == 0:
            return profile
            
        # Sample data if it's too large
        if len(data) > 100000:
            data_sample = self.sample_data(data, 100000)
        else:
            data_sample = data
        
        # Generate profile for each column
        for col in data_sample.columns:
            col_profile = {}
            
            # Basic information
            col_profile['data_type'] = str(data_sample[col].dtype)
            col_profile['missing_count'] = int(data_sample[col].isna().sum())
            col_profile['missing_percentage'] = float(data_sample[col].isna().mean() * 100)
            
            # Get non-missing values for further analysis
            non_missing = data_sample[col].dropna()
            
            if len(non_missing) > 0:
                # Type-specific statistics
                if pd.api.types.is_numeric_dtype(non_missing):
                    col_profile['min'] = float(non_missing.min())
                    col_profile['max'] = float(non_missing.max())
                    col_profile['mean'] = float(non_missing.mean())
                    col_profile['median'] = float(non_missing.median())
                    col_profile['std'] = float(non_missing.std())
                    
                    # Check for outliers using IQR method
                    q1 = float(non_missing.quantile(0.25))
                    q3 = float(non_missing.quantile(0.75))
                    iqr = q3 - q1
                    outlier_low = q1 - (1.5 * iqr)
                    outlier_high = q3 + (1.5 * iqr)
                    outliers = non_missing[(non_missing < outlier_low) | (non_missing > outlier_high)]
                    col_profile['outlier_count'] = int(len(outliers))
                    col_profile['outlier_percentage'] = float(len(outliers) / len(non_missing) * 100)
                    
                elif pd.api.types.is_string_dtype(non_missing) or pd.api.types.is_object_dtype(non_missing):
                    # Convert to strings for consistent processing
                    non_missing = non_missing.astype(str)
                    
                    col_profile['min_length'] = int(non_missing.str.len().min())
                    col_profile['max_length'] = int(non_missing.str.len().max())
                    col_profile['avg_length'] = float(non_missing.str.len().mean())
                    
                    # Sample of unique values
                    unique_values = non_missing.unique()
                    col_profile['unique_count'] = int(len(unique_values))
                    col_profile['unique_percentage'] = float(len(unique_values) / len(non_missing) * 100)
                    
                    # Most common values
                    if len(unique_values) < 100:  # Only compute for columns with reasonable cardinality
                        value_counts = non_missing.value_counts(normalize=True)
                        col_profile['top_values'] = [
                            {"value": str(value), "percentage": float(percentage * 100)}
                            for value, percentage in value_counts.head(10).items()
                        ]
                
                elif pd.api.types.is_datetime64_dtype(non_missing):
                    col_profile['min'] = non_missing.min().isoformat()
                    col_profile['max'] = non_missing.max().isoformat()
                    # Time range in days
                    time_range = (non_missing.max() - non_missing.min()).total_seconds() / (24 * 3600)
                    col_profile['time_range_days'] = float(time_range)
                    
                    # Date patterns
                    has_weekends = (non_missing.dt.dayofweek >= 5).any()
                    col_profile['has_weekend_dates'] = bool(has_weekends)
                    
                    # Check for seasonal patterns
                    month_counts = non_missing.dt.month.value_counts(normalize=True)
                    col_profile['month_distribution'] = {
                        str(month): float(count * 100)
                        for month, count in month_counts.items()
                    }
            
            # Add column profile to overall profile
            profile[col] = col_profile
        
        return profile
    
    def detect_relationships(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect potential relationships between columns.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of dictionaries with relationship information
        """
        relationships = []
        
        # Safety check
        if not isinstance(data, pd.DataFrame) or len(data) == 0:
            return relationships
        
        # Sample data if it's too large
        if len(data) > 50000:
            data = self.sample_data(data, 50000)
        
        try:
            # Find duplicate column values
            cols = data.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    # Skip comparisons between different types
                    if data[cols[i]].dtype != data[cols[j]].dtype:
                        continue
                        
                    # Check for exact matches between columns
                    if data[cols[i]].equals(data[cols[j]]):
                        relationships.append({
                            'type': 'duplicate_columns',
                            'columns': [cols[i], cols[j]],
                            'confidence': 1.0,
                            'description': f"Columns {cols[i]} and {cols[j]} have identical values"
                        })
                        continue
                    
                    # For numeric columns, check correlation
                    if pd.api.types.is_numeric_dtype(data[cols[i]]) and pd.api.types.is_numeric_dtype(data[cols[j]]):
                        try:
                            corr = data[cols[i]].corr(data[cols[j]])
                            if abs(corr) > 0.9:
                                relationships.append({
                                    'type': 'high_correlation',
                                    'columns': [cols[i], cols[j]],
                                    'correlation': float(corr),
                                    'confidence': float(abs(corr)),
                                    'description': f"High {'positive' if corr > 0 else 'negative'} correlation ({corr:.2f})"
                                })
                        except:
                            pass
            
            # Detect potential primary-foreign key relationships
            for col1 in cols:
                # Candidates for primary keys have high cardinality and few nulls
                is_pk_candidate = (data[col1].nunique() / len(data) > 0.9) and (data[col1].isna().mean() < 0.01)
                
                if is_pk_candidate:
                    for col2 in cols:
                        if col1 == col2:
                            continue
                            
                        # Check if values in col2 are a subset of col1
                        if set(data[col2].dropna().unique()).issubset(set(data[col1].unique())):
                            relationships.append({
                                'type': 'potential_foreign_key',
                                'primary_key': col1,
                                'foreign_key': col2,
                                'confidence': float(data[col2].nunique() / data[col1].nunique()),
                                'description': f"Column {col2} may be a foreign key to {col1}"
                            })
            
        except Exception as e:
            logger.warning(f"Error detecting relationships: {str(e)}")
            
        return relationships
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality of a DataFrame.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Dictionary with data quality metrics
        """
        # Safety check
        if not isinstance(data, pd.DataFrame) or len(data) == 0:
            return {"overall_score": 0, "issues": []}
            
        issues = []
        scores = {
            "completeness": 0,
            "consistency": 0,
            "validity": 0,
            "uniqueness": 0
        }
        
        # Completeness: Measure presence of nulls
        null_percentages = data.isna().mean()
        avg_null_percentage = float(null_percentages.mean())
        scores["completeness"] = 1.0 - avg_null_percentage
        
        # Flag columns with high null percentages
        high_null_cols = [(col, pct) for col, pct in null_percentages.items() if pct > 0.1]
        for col, pct in high_null_cols:
            issues.append({
                "type": "high_null_percentage",
                "column": col,
                "percentage": float(pct * 100),
                "severity": "high" if pct > 0.5 else "medium" if pct > 0.2 else "low"
            })
            
        # Validity: Check for outliers in numeric columns
        validity_scores = []
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                non_null = data[col].dropna()
                
                if len(non_null) > 0:
                    # Use IQR to detect outliers
                    q1 = non_null.quantile(0.25)
                    q3 = non_null.quantile(0.75)
                    iqr = q3 - q1
                    outlier_low = q1 - (1.5 * iqr)
                    outlier_high = q3 + (1.5 * iqr)
                    outlier_pct = ((non_null < outlier_low) | (non_null > outlier_high)).mean()
                    
                    validity_scores.append(1.0 - outlier_pct)
                    
                    # Flag columns with high outlier percentage
                    if outlier_pct > 0.05:
                        issues.append({
                            "type": "high_outlier_percentage",
                            "column": col,
                            "percentage": float(outlier_pct * 100),
                            "severity": "high" if outlier_pct > 0.2 else "medium" if outlier_pct > 0.1 else "low"
                        })
        
        scores["validity"] = sum(validity_scores) / max(1, len(validity_scores))
        
        # Uniqueness: Check for duplicate values in columns
        uniqueness_scores = []
        
        for col in data.columns:
            non_null = data[col].dropna()
            if len(non_null) > 0:
                uniqueness = non_null.nunique() / len(non_null)
                uniqueness_scores.append(uniqueness)
        
        scores["uniqueness"] = sum(uniqueness_scores) / max(1, len(uniqueness_scores))
        
        # Consistency: Check for mixed data types within columns
        for col in data.columns:
            if pd.api.types.is_object_dtype(data[col]):
                sample = data[col].dropna().sample(min(1000, len(data[col].dropna())))
                
                # Check if column contains mixed types (numbers and strings)
                num_numeric = sum(1 for x in sample if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()))
                num_string = sum(1 for x in sample if isinstance(x, str) and not x.isdigit())
                
                if num_numeric > 0 and num_string > 0:
                    pct_mixed = min(num_numeric, num_string) / len(sample)
                    if pct_mixed > 0.05:
                        issues.append({
                            "type": "mixed_data_types",
                            "column": col,
                            "percentage": float(pct_mixed * 100),
                            "severity": "high" if pct_mixed > 0.2 else "medium" if pct_mixed > 0.1 else "low"
                        })
        
        # Set consistency score based on number of mixed type issues
        mixed_type_issues = len([i for i in issues if i["type"] == "mixed_data_types"])
        scores["consistency"] = 1.0 - (mixed_type_issues / max(1, len(data.columns)))
        
        # Overall score is the average of individual scores
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "overall_score": float(overall_score),
            "scores": {k: float(v) for k, v in scores.items()},
            "issues": issues
        }
    
    def _update_stats(self, rows_processed: int = 0):
        """
        Update processing statistics.
        
        Args:
            rows_processed: Number of rows processed in this update
        """
        self.stats['rows_processed'] += rows_processed
        
    def report_progress(self, current: int, total: int, prefix: str = ""):
        """
        Report processing progress.
        
        Args:
            current: Current position
            total: Total size
            prefix: Prefix for progress message
        """
        if self.show_progress:
            percentage = (current / total) * 100 if total > 0 else 0
            memory_usage = self.memory_monitor.get_memory_usage_mb()
            
            logger.info(f"{prefix} Progress: {percentage:.1f}% ({current}/{total}) | Memory: {memory_usage:.1f} MB")
        
    def _temp_file(self) -> Tuple[BinaryIO, str]:
        """
        Create a temporary file for intermediate results.
        
        Returns:
            Tuple of (file object, file path)
        """
        # Create a temporary file that will be automatically deleted when closed
        fd, path = tempfile.mkstemp(suffix='.tmp')
        return os.fdopen(fd, 'wb'), path
        
    def _log_error(self, error_msg: str, file_path: str = None, critical: bool = False):
        """
        Log an error during processing.
        
        Args:
            error_msg: Error message
            file_path: Associated file path
            critical: Whether this is a critical error
        """
        error_info = {
            "message": error_msg,
            "file": file_path,
            "time": time.time(),
            "critical": critical
        }
        
        self.stats['errors'].append(error_info)
        
        if critical:
            logger.error(f"Critical error: {error_msg} ({file_path})")
        else:
            logger.warning(f"Error: {error_msg} ({file_path})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory()
        return False  # Don't suppress exceptions