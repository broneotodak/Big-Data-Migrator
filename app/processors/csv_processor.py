"""
CSV file processor with memory-efficient chunking for large files.
"""
import os
import io
import csv
from typing import Dict, Generator, List, Optional, Any

import pandas as pd
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.processors.base_processor import BaseProcessor

logger = get_logger(__name__)

class LargeCSVProcessor(BaseProcessor):
    """
    Process large CSV files with memory-efficient chunking.
    
    This class handles multi-GB CSV files by using pandas' built-in
    chunking capabilities to avoid loading the entire file into memory.
    """
    
    def __init__(self, **kwargs):
        """Initialize the CSV processor with shared parameters."""
        super().__init__(**kwargs)
        # CSV-specific settings
        self.delimiter = kwargs.get('delimiter', ',')
        self.quotechar = kwargs.get('quotechar', '"')
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.date_format = kwargs.get('date_format', None)
        # Try to infer CSV dialect if possible
        self.infer_dialect = kwargs.get('infer_dialect', True)
        
    def read_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Read a CSV file in memory-efficient chunks.
        
        Args:
            file_path: Path to the CSV file
            
        Yields:
            DataFrame chunks from the CSV file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        # Get file info
        file_info = self.get_file_info(file_path)
        
        # Determine optimal chunk size if not provided
        chunk_size = self.chunk_size
        if chunk_size is None:
            chunk_size = self.file_calculator.calculate_optimal_chunk_size(file_path, file_type="csv")
            logger.info(f"Auto-calculated chunk size: {chunk_size} rows")
        
        # Infer CSV dialect if requested
        delimiter = self.delimiter
        quotechar = self.quotechar
        
        if self.infer_dialect:
            dialect = self._infer_csv_dialect(file_path)
            delimiter = dialect.get('delimiter', delimiter)
            quotechar = dialect.get('quotechar', quotechar)
            logger.info(f"Inferred CSV dialect: delimiter='{delimiter}', quotechar='{quotechar}'")
        
        # Setup reader with inferred or provided parameters
        try:
            total_rows = file_info.get('estimated_rows', 0)
            reader = pd.read_csv(
                file_path,
                chunksize=chunk_size,
                delimiter=delimiter,
                quotechar=quotechar,
                encoding=self.encoding,
                low_memory=True,
                on_bad_lines='warn'
            )
            
            # Process chunks
            for i, chunk in enumerate(tqdm(reader, total=total_rows//chunk_size+1 if total_rows > 0 else None, 
                                         desc=f"Reading {os.path.basename(file_path)}", 
                                         disable=not self.show_progress)):
                # Optimize memory usage for this chunk
                optimized_chunk = self.optimize_dtypes(chunk)
                
                # Convert date columns if format is specified
                if self.date_format:
                    for col in optimized_chunk.columns:
                        # Simple heuristic to identify potential date columns
                        if 'date' in col.lower() or 'time' in col.lower():
                            try:
                                optimized_chunk[col] = pd.to_datetime(
                                    optimized_chunk[col], 
                                    format=self.date_format,
                                    errors='ignore'
                                )
                            except:
                                pass  # Skip if column can't be parsed as dates
                
                yield optimized_chunk
                
                # Clean up memory periodically
                if i > 0 and i % 10 == 0:
                    self.memory_monitor.cleanup_memory()
                    
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise
            
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a CSV file without reading its full contents.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with file metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "file_type": "csv",
            "delimiter": self.delimiter,
            "encoding": self.encoding,
        }
        
        try:
            # Count lines efficiently without loading the entire file
            with open(file_path, 'rb') as f:
                # Count newlines in the file
                line_count = sum(1 for _ in f)
                file_info["total_lines"] = line_count
                file_info["estimated_rows"] = max(0, line_count - 1)  # Subtract header
                
            # Infer dialect if requested
            if self.infer_dialect:
                dialect = self._infer_csv_dialect(file_path)
                file_info["delimiter"] = dialect.get('delimiter', self.delimiter)
                file_info["quotechar"] = dialect.get('quotechar', self.quotechar)
                
            # Read first few lines to get column info
            sample = pd.read_csv(file_path, nrows=5)
            file_info["columns"] = list(sample.columns)
            file_info["column_count"] = len(sample.columns)
            file_info["dtypes"] = {col: str(sample[col].dtype) for col in sample.columns}
            
            # Estimate memory requirements
            file_info["estimated_memory_mb"] = self.file_calculator.estimate_file_memory_usage(file_path, "csv")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting CSV file info for {file_path}: {str(e)}")
            # Return basic file info even if detailed analysis fails
            return file_info
            
    def _infer_csv_dialect(self, file_path: str, sample_size: int = 10240) -> Dict[str, str]:
        """
        Infer CSV dialect by examining a sample of the file.
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of bytes to sample
            
        Returns:
            Dictionary with dialect information
        """
        try:
            # Read a sample of the file
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size).decode(self.encoding, errors='replace')
            
            # Use csv.Sniffer to detect the dialect
            dialect = csv.Sniffer().sniff(sample)
            
            return {
                'delimiter': dialect.delimiter,
                'quotechar': dialect.quotechar,
                'has_header': csv.Sniffer().has_header(sample)
            }
            
        except Exception as e:
            logger.warning(f"Failed to infer CSV dialect: {str(e)}. Using default settings.")
            return {
                'delimiter': self.delimiter,
                'quotechar': self.quotechar,
                'has_header': True
            }
            
    def split_file(self, file_path: str, output_dir: str, max_rows_per_file: int) -> List[str]:
        """
        Split a large CSV file into smaller chunks.
        
        Args:
            file_path: Path to the large CSV file
            output_dir: Directory to save split files
            max_rows_per_file: Maximum rows per split file
            
        Returns:
            List of paths to the created split files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        output_files = []
        file_count = 0
        
        try:
            # Get file info to determine optimal splitting
            file_info = self.get_file_info(file_path)
            delimiter = file_info.get('delimiter', self.delimiter)
            total_rows = file_info.get('estimated_rows', 0)
            
            # Read header
            df_header = pd.read_csv(file_path, nrows=0)
            header = list(df_header.columns)
            
            # Process file in chunks
            reader = pd.read_csv(
                file_path,
                chunksize=min(max_rows_per_file, 100000),  # Use smaller chunks for processing
                delimiter=delimiter,
                quotechar=self.quotechar,
                encoding=self.encoding
            )
            
            current_rows = 0
            current_file_path = os.path.join(output_dir, f"{file_base}_part_{file_count:03d}.csv")
            
            # Setup progress bar
            pbar = tqdm(total=total_rows, desc="Splitting file", disable=not self.show_progress)
            
            # Write header to first file
            pd.DataFrame(columns=header).to_csv(current_file_path, index=False)
            output_files.append(current_file_path)
            
            for chunk in reader:
                # Check if we need to start a new file
                if current_rows + len(chunk) > max_rows_per_file and current_rows > 0:
                    # Start a new file
                    file_count += 1
                    current_rows = 0
                    current_file_path = os.path.join(output_dir, f"{file_base}_part_{file_count:03d}.csv")
                    pd.DataFrame(columns=header).to_csv(current_file_path, index=False)
                    output_files.append(current_file_path)
                
                # Append to current file (using mode='a' to append)
                chunk.to_csv(current_file_path, mode='a', header=False, index=False)
                current_rows += len(chunk)
                pbar.update(len(chunk))
            
            pbar.close()
            logger.info(f"Split CSV file into {len(output_files)} parts")
            return output_files
            
        except Exception as e:
            logger.error(f"Error splitting CSV file: {str(e)}")
            raise