"""
CSV file processor with memory-efficient chunking for large files.
"""
import os
import io
import csv
import time
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
        chunk_size = self.get_optimal_chunk_size(file_path)
        if chunk_size is None:
            chunk_size = 10000  # Default fallback
            logger.info(f"Using default chunk size: {chunk_size} rows")
        else:
            # Convert MB to approximate rows (assuming ~100 bytes per row)
            chunk_size = int(chunk_size * 1024 * 1024 / 100)
            logger.info(f"Auto-calculated chunk size: {chunk_size} rows")
        
        # Infer CSV dialect if requested
        delimiter = self.delimiter
        quotechar = self.quotechar
        skiprows = None
        
        if self.infer_dialect:
            dialect = self._infer_csv_dialect(file_path)
            delimiter = dialect.get('delimiter', delimiter)
            quotechar = dialect.get('quotechar', quotechar)
            logger.info(f"Inferred CSV dialect: delimiter='{delimiter}', quotechar='{quotechar}'")
        
        # Check for empty rows at the beginning and find actual header
        try:
            # Read first few rows without header to detect structure
            test_df = pd.read_csv(
                file_path,
                nrows=10,
                header=None,
                delimiter=delimiter,
                quotechar=quotechar,
                encoding=self.encoding,
                low_memory=True,
                on_bad_lines='warn'
            )
            
            # Find the first non-empty row that could be a header
            header_row_index = None
            for idx, row in test_df.iterrows():
                # Check if row has mostly string values (likely header)
                non_null_values = row.dropna()
                if len(non_null_values) > 0:
                    # Check if it looks like a header (contains strings, not just numbers/dates)
                    string_like = sum(1 for val in non_null_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').replace('/', '').isdigit())
                    if string_like > len(non_null_values) * 0.5:  # More than 50% are string-like
                        header_row_index = idx
                        break
            
            # If we found a header row that's not the first row, skip rows before it
            if header_row_index is not None and header_row_index > 0:
                skiprows = list(range(header_row_index))
                logger.info(f"Detected header at row {header_row_index}, skipping {len(skiprows)} empty rows")
            
        except Exception as e:
            logger.warning(f"Could not detect empty rows, using default reading: {e}")
        
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
                on_bad_lines='warn',
                skiprows=skiprows  # Skip empty rows if detected
            )
            
            # Process chunks
            for i, chunk in enumerate(tqdm(reader, total=total_rows//chunk_size+1 if total_rows > 0 else None, 
                                         desc=f"Reading {os.path.basename(file_path)}", 
                                         disable=not self.show_progress)):
                # Optimize memory usage for this chunk
                try:
                    # Use base class method to detect optimal types
                    optimal_types = self.detect_data_type(chunk)
                    # Apply the types
                    for col, dtype in optimal_types.items():
                        if col in chunk.columns:
                            try:
                                chunk[col] = chunk[col].astype(dtype)
                            except:
                                pass  # Skip if conversion fails
                    optimized_chunk = chunk
                except:
                    optimized_chunk = chunk  # Use original if optimization fails
                
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
                
                # Filter out summary rows
                filtered_chunk = self._filter_summary_rows(optimized_chunk)
                
                yield filtered_chunk
                
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
                
            # Read first few lines to get column info with empty row detection
            try:
                # Check for empty rows at the beginning and find actual header
                skiprows = None
                delimiter = file_info.get("delimiter", self.delimiter)
                quotechar = file_info.get("quotechar", self.quotechar)
                
                # Read first few rows without header to detect structure
                test_df = pd.read_csv(
                    file_path,
                    nrows=10,
                    header=None,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    encoding=self.encoding,
                    low_memory=True,
                    on_bad_lines='warn'
                )
                
                # Find the first non-empty row that could be a header
                header_row_index = None
                for idx, row in test_df.iterrows():
                    # Check if row has mostly string values (likely header)
                    non_null_values = row.dropna()
                    if len(non_null_values) > 0:
                        # Check if it looks like a header (contains strings, not just numbers/dates)
                        string_like = sum(1 for val in non_null_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').replace('/', '').isdigit())
                        if string_like > len(non_null_values) * 0.5:  # More than 50% are string-like
                            header_row_index = idx
                            break
                
                # If we found a header row that's not the first row, skip rows before it
                if header_row_index is not None and header_row_index > 0:
                    skiprows = list(range(header_row_index))
                    logger.info(f"Detected header at row {header_row_index}, skipping {len(skiprows)} empty rows")
                
                # Now read with proper header detection
                sample = pd.read_csv(file_path, nrows=5, skiprows=skiprows)
                file_info["columns"] = list(sample.columns)
                file_info["column_count"] = len(sample.columns)
                file_info["dtypes"] = {col: str(sample[col].dtype) for col in sample.columns}
                
            except Exception as e:
                logger.warning(f"Could not read CSV sample with header detection: {e}")
                # Fallback to simple reading
                try:
                    sample = pd.read_csv(file_path, nrows=5)
                    file_info["columns"] = list(sample.columns)
                    file_info["column_count"] = len(sample.columns)
                    file_info["dtypes"] = {col: str(sample[col].dtype) for col in sample.columns}
                except Exception as e2:
                    logger.error(f"Could not read CSV file at all: {e2}")
                    file_info["columns"] = []
                    file_info["column_count"] = 0
                    file_info["dtypes"] = {}
            
            # Estimate memory requirements
            file_info["estimated_memory_mb"] = self.estimate_memory_requirement(file_path)
            
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

    def sample_data(self, file_path: str, sample_size: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get a sample of data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of rows to sample
            
        Returns:
            DataFrame with sample data or None if error
        """
        try:
            # Infer CSV dialect if requested
            delimiter = self.delimiter
            quotechar = self.quotechar
            skiprows = None
            
            if self.infer_dialect:
                dialect = self._infer_csv_dialect(file_path)
                delimiter = dialect.get('delimiter', delimiter)
                quotechar = dialect.get('quotechar', quotechar)
            
            # Check for empty rows at the beginning and find actual header
            try:
                # Read first few rows without header to detect structure
                test_df = pd.read_csv(
                    file_path,
                    nrows=10,
                    header=None,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    encoding=self.encoding,
                    low_memory=True,
                    on_bad_lines='warn'
                )
                
                # Find the first non-empty row that could be a header
                header_row_index = None
                for idx, row in test_df.iterrows():
                    # Check if row has mostly string values (likely header)
                    non_null_values = row.dropna()
                    if len(non_null_values) > 0:
                        # Check if it looks like a header (contains strings, not just numbers/dates)
                        string_like = sum(1 for val in non_null_values if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').replace('/', '').isdigit())
                        if string_like > len(non_null_values) * 0.5:  # More than 50% are string-like
                            header_row_index = idx
                            break
                
                # If we found a header row that's not the first row, skip rows before it
                if header_row_index is not None and header_row_index > 0:
                    skiprows = list(range(header_row_index))
                    logger.info(f"Detected header at row {header_row_index}, skipping {len(skiprows)} empty rows")
                
            except Exception as e:
                logger.warning(f"Could not detect empty rows in sample, using default reading: {e}")
            
            # Read a sample from the beginning of the file
            sample_df = pd.read_csv(
                file_path,
                nrows=sample_size,
                delimiter=delimiter,
                quotechar=quotechar,
                encoding=self.encoding,
                low_memory=True,
                on_bad_lines='warn',
                skiprows=skiprows  # Skip empty rows if detected
            )
            
            # Filter out summary rows from sample
            sample_df = self._filter_summary_rows(sample_df)
            
            logger.debug(f"Sampled {len(sample_df)} rows from {os.path.basename(file_path)}")
            return sample_df
            
        except Exception as e:
            logger.error(f"Error sampling data from {file_path}: {str(e)}")
            return None

    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a CSV file using the provided function or default processing.
        This method implements the abstract method from BaseProcessor.
        
        Args:
            file_path: Path to the CSV file to process
            **kwargs: Additional processing parameters including:
                - process_fn: Function to apply to each chunk
                - output_path: Path to save processed output
                
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Extract parameters
            process_fn = kwargs.get('process_fn')
            output_path = kwargs.get('output_path')
            
            # Initialize results
            total_rows_processed = 0
            processed_chunks = []
            
            # Track memory usage
            self.memory_monitor.start_tracking_step(f"csv_processing_{os.path.basename(file_path)}")
            
            # Process file in chunks
            for chunk in self.read_file(file_path):
                chunk_rows = len(chunk)
                
                # Apply processing function if provided
                if process_fn:
                    try:
                        chunk = process_fn(chunk)
                    except Exception as e:
                        logger.warning(f"Error applying process function to chunk: {str(e)}")
                
                processed_chunks.append(chunk)
                total_rows_processed += chunk_rows
                
                # Update memory peak tracking
                self.memory_monitor.update_step_peak(f"csv_processing_{os.path.basename(file_path)}")
            
            # Combine results if we have chunks to process
            if processed_chunks:
                combined_df = pd.concat(processed_chunks, ignore_index=True)
                
                # Save output if path provided
                if output_path:
                    combined_df.to_csv(output_path, index=False)
                    logger.info(f"Saved processed CSV to {output_path}")
            else:
                combined_df = pd.DataFrame()
            
            # End memory tracking
            memory_stats = self.memory_monitor.end_tracking_step(f"csv_processing_{os.path.basename(file_path)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "status": "completed",
                "file_path": file_path,
                "output_path": output_path,
                "data": combined_df,
                "stats": {
                    "total_rows_processed": total_rows_processed,
                    "total_chunks": len(processed_chunks),
                    "processing_time": processing_time,
                    "memory_peak": memory_stats.get("peak_mb", 0),
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "file_path": file_path,
                "stats": {
                    "total_rows_processed": 0,
                    "processing_time": time.time() - start_time,
                    "memory_peak": 0
                }
            }

    def _filter_summary_rows(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out summary/total rows from a data chunk.
        
        Args:
            chunk: DataFrame chunk to filter
            
        Returns:
            Filtered DataFrame with summary rows removed
        """
        try:
            original_length = len(chunk)
            
            # Define patterns that indicate summary rows
            summary_indicators = [
                'total:', 'total', 'sum:', 'sum', 'grand total', 
                'subtotal', 'summary', 'balance', 'amount due'
            ]
            
            # Create a mask to identify rows to keep (not summary rows)
            keep_mask = pd.Series([True] * len(chunk), index=chunk.index)
            
            # Check each column for summary indicators
            for col in chunk.columns:
                if chunk[col].dtype == 'object' or chunk[col].dtype.name == 'string':
                    # Check if any cell in this column contains summary indicators
                    for indicator in summary_indicators:
                        # Case-insensitive check for summary indicators
                        mask = chunk[col].astype(str).str.lower().str.contains(indicator, na=False)
                        keep_mask = keep_mask & ~mask
            
            # Also filter out rows where most columns are empty (often summary rows)
            # Count non-null values per row
            non_null_count = chunk.count(axis=1)
            total_columns = len(chunk.columns)
            
            # If a row has very few non-null values compared to total columns, it might be a summary row
            # But be careful not to filter legitimate sparse data
            sparse_threshold = max(2, total_columns * 0.3)  # At least 30% of columns should have data
            sparse_mask = non_null_count >= sparse_threshold
            
            # Apply both filters
            final_mask = keep_mask & sparse_mask
            filtered_chunk = chunk[final_mask].copy()
            
            filtered_count = original_length - len(filtered_chunk)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} summary/total rows from chunk")
            
            return filtered_chunk
            
        except Exception as e:
            logger.warning(f"Error filtering summary rows: {str(e)}, returning original chunk")
            return chunk