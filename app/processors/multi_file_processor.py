"""
Multi-file processor with memory pooling for processing multiple files efficiently.
"""
import os
import time
from typing import Dict, Generator, List, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.processors.base_processor import BaseProcessor
from app.processors.csv_processor import LargeCSVProcessor
from app.processors.pdf_processor import PDFProcessor
from app.processors.excel_processor import ExcelProcessor
from app.processors.docx_processor import DocxProcessor
from app.processors.image_processor import ImageProcessor
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer

logger = get_logger(__name__)

class MultiFileProcessor(BaseProcessor):
    """
    Process multiple files efficiently with memory pooling.
    
    This class orchestrates the processing of multiple files by:
    1. Allocating memory efficiently across files
    2. Prioritizing processing order based on file characteristics
    3. Maintaining overall memory constraints
    """
    
    def __init__(self, **kwargs):
        """Initialize the multi-file processor with shared parameters."""
        super().__init__(**kwargs)
        # Multi-file specific settings
        self.max_concurrent = kwargs.get('max_concurrent', 1)  # Default to sequential processing
        self.prioritize_by = kwargs.get('prioritize_by', 'size')  # 'size', 'type', or 'name'
        self.reverse_priority = kwargs.get('reverse_priority', False)  # False=smallest first
        self.batch_size = kwargs.get('batch_size', 5)  # Max files to process in one batch
        
        # Process CSV, XLSX, PDF, DOCX, images
        self.file_type_map = {
            '.csv': LargeCSVProcessor,
            '.xlsx': ExcelProcessor,
            '.xls': ExcelProcessor,
            '.pdf': PDFProcessor,
            '.docx': DocxProcessor,
            '.doc': DocxProcessor,
            '.png': ImageProcessor,
            '.jpg': ImageProcessor,
            '.jpeg': ImageProcessor,
            '.tiff': ImageProcessor,
            '.tif': ImageProcessor,
            '.bmp': ImageProcessor,
        }
        
        # Initialize processor cache
        self._processor_cache = {}
        # Lock for thread safety
        self._lock = threading.RLock()
        
    def read_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Read a single file using the appropriate processor.
        
        Args:
            file_path: Path to the file
            
        Yields:
            DataFrame chunks from the file
        """
        # Get appropriate processor for file type
        processor = self._get_processor_for_file(file_path)
        
        if processor:
            # Use the processor to read the file
            for chunk in processor.read_file(file_path):
                yield chunk
        else:
            # Return error DataFrame if no processor available
            error_df = pd.DataFrame({
                'error': [f"No processor available for file: {file_path}"],
                'file_path': [file_path]
            })
            yield error_df
            
    def process_files(self, file_paths: List[str],
                     process_fn: Optional[callable] = None,
                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple files with memory optimization.
        
        Args:
            file_paths: List of file paths to process
            process_fn: Optional function to apply to each chunk
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not file_paths:
            return {"status": "error", "message": "No files provided"}
        
        # Reset processing stats
        self.processing_stats = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "failed_files": 0,
            "total_rows_processed": 0,
            "total_processing_time": 0,
            "memory_peak": 0,
            "file_stats": {},
            "errors": [],
            "started_at": time.time(),
            "completed_at": None,
            "status": "processing",
        }
        
        # Start memory tracking
        self.memory_monitor.start_tracking_step("multi_file_processing")
        
        try:
            # Optimize processing order
            ordered_files = self._prioritize_files(file_paths)
            
            # Process files in optimized batches
            results = {}
            
            # Determine concurrency based on available memory
            available_memory = self.memory_monitor.get_available_memory()
            
            if self.max_concurrent > 1:
                # Check if we have enough memory for concurrent processing
                memory_per_process = available_memory / self.max_concurrent
                if memory_per_process < 500:  # At least 500MB per process
                    logger.warning(f"Limited memory available ({available_memory}MB). "
                                 f"Reducing concurrency from {self.max_concurrent} to 1.")
                    self.max_concurrent = 1
            
            # Process either sequentially or in parallel
            if self.max_concurrent <= 1:
                # Sequential processing
                results = self._process_files_sequential(ordered_files, process_fn, output_dir)
            else:
                # Parallel processing
                results = self._process_files_parallel(ordered_files, process_fn, output_dir)
            
            # Update final stats
            self.processing_stats["completed_at"] = time.time()
            self.processing_stats["total_processing_time"] = self.processing_stats["completed_at"] - self.processing_stats["started_at"]
            self.processing_stats["status"] = "completed"
            
            return {
                "results": results,
                "stats": self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Error in multi-file processing: {str(e)}")
            self.processing_stats["status"] = "failed"
            self.processing_stats["errors"].append(str(e))
            return {
                "status": "error",
                "message": str(e),
                "stats": self.processing_stats
            }
            
        finally:
            # Clean up memory and end tracking
            self.memory_monitor.cleanup_memory(aggressive=True)
            self.memory_monitor.end_tracking_step("multi_file_processing")
            
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a file using the appropriate processor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        # Get appropriate processor for file type
        processor = self._get_processor_for_file(file_path)
        
        if processor:
            # Use the processor to get file info
            return processor.get_file_info(file_path)
        else:
            # Return basic file info if no processor available
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_bytes": os.path.getsize(file_path),
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "file_type": "unknown",
                "error": "No processor available for this file type"
            }
            
    def _get_processor_for_file(self, file_path: str) -> Optional[BaseProcessor]:
        """
        Get the appropriate processor for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor instance or None if no processor available
        """
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Check if we already have a processor for this file type
        with self._lock:
            if ext in self._processor_cache:
                return self._processor_cache[ext]
            
            # If not, create a new processor if we have one for this file type
            if ext in self.file_type_map:
                processor_class = self.file_type_map[ext]
                
                # Create processor with shared memory monitor
                processor = processor_class(
                    memory_monitor=self.memory_monitor,
                    resource_optimizer=self.resource_optimizer,
                    show_progress=self.show_progress
                )
                
                # Cache the processor for reuse
                self._processor_cache[ext] = processor
                return processor
                
        # No processor available for this file type
        return None
        
    def _prioritize_files(self, file_paths: List[str]) -> List[str]:
        """
        Prioritize files for processing based on specified criteria.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Prioritized list of file paths
        """
        if not file_paths:
            return []
            
        # Get file info for all files
        file_info_list = []
        for file_path in file_paths:
            try:
                # Get basic file info (minimal overhead)
                file_info = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_size_bytes": os.path.getsize(file_path),
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                }
                
                # Get file extension
                _, ext = os.path.splitext(file_path.lower())
                file_info["file_type"] = ext.lstrip('.')
                
                file_info_list.append(file_info)
            except Exception as e:
                logger.warning(f"Error getting info for file {file_path}: {str(e)}")
        
        # Sort files based on priority criteria
        if self.prioritize_by == 'size':
            # Sort by file size
            file_info_list.sort(key=lambda x: x["file_size_bytes"], reverse=self.reverse_priority)
        elif self.prioritize_by == 'type':
            # Sort by file type
            file_info_list.sort(key=lambda x: x["file_type"], reverse=self.reverse_priority)
        elif self.prioritize_by == 'name':
            # Sort by file name
            file_info_list.sort(key=lambda x: x["file_name"], reverse=self.reverse_priority)
        
        # Return sorted file paths
        return [info["file_path"] for info in file_info_list]
        
    def _process_files_sequential(self, file_paths: List[str], 
                                process_fn: Optional[callable], 
                                output_dir: Optional[str]) -> Dict[str, Any]:
        """
        Process files sequentially.
        
        Args:
            file_paths: List of file paths to process
            process_fn: Optional function to apply to each chunk
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        # Process files one by one
        for file_path in tqdm(file_paths, desc="Processing files", disable=not self.show_progress):
            try:
                # Get processor for this file
                processor = self._get_processor_for_file(file_path)
                
                if processor:
                    # Process the file
                    file_result = processor.process_file(file_path, process_fn=process_fn, 
                                                       output_path=os.path.join(output_dir, os.path.basename(file_path)) if output_dir else None)
                    
                    # Store results and update stats
                    results[file_path] = file_result
                    self.processing_stats["processed_files"] += 1
                    self.processing_stats["total_rows_processed"] += file_result["stats"]["total_rows_processed"]
                    self.processing_stats["file_stats"][file_path] = file_result["stats"]
                    
                    # Update memory peak
                    if file_result["stats"].get("memory_peak", 0) > self.processing_stats["memory_peak"]:
                        self.processing_stats["memory_peak"] = file_result["stats"]["memory_peak"]
                else:
                    # Log error for unsupported file type
                    error_msg = f"No processor available for file: {file_path}"
                    logger.error(error_msg)
                    self.processing_stats["failed_files"] += 1
                    self.processing_stats["errors"].append(error_msg)
                
                # Clean up memory after each file
                self.memory_monitor.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                self.processing_stats["failed_files"] += 1
                self.processing_stats["errors"].append(f"File {file_path}: {str(e)}")
                
                # Clean up memory after error
                self.memory_monitor.cleanup_memory(aggressive=True)
        
        return results
        
    def _process_files_parallel(self, file_paths: List[str],
                              process_fn: Optional[callable],
                              output_dir: Optional[str]) -> Dict[str, Any]:
        """
        Process files in parallel with memory constraints.
        
        Args:
            file_paths: List of file paths to process
            process_fn: Optional function to apply to each chunk
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        # Calculate memory allocation per worker
        available_memory = self.memory_monitor.get_available_memory()
        memory_per_worker = max(500, available_memory / self.max_concurrent)  # At least 500MB per worker
        
        # Process in batches to control memory usage
        file_batches = [file_paths[i:i+self.batch_size] 
                      for i in range(0, len(file_paths), self.batch_size)]
        
        for batch_idx, batch in enumerate(file_batches):
            batch_results = {}
            batch_futures = {}
            
            with ThreadPoolExecutor(max_workers=min(self.max_concurrent, len(batch))) as executor:
                # Submit all files in this batch
                for file_path in batch:
                    future = executor.submit(self._process_single_file, 
                                          file_path, process_fn, output_dir)
                    batch_futures[future] = file_path
                
                # Process results as they complete
                for future in tqdm(as_completed(batch_futures), 
                                 total=len(batch_futures),
                                 desc=f"Batch {batch_idx+1}/{len(file_batches)}",
                                 disable=not self.show_progress):
                    file_path = batch_futures[future]
                    try:
                        file_result = future.result()
                        
                        # Store results
                        batch_results[file_path] = file_result
                        
                        # Update stats
                        with self._lock:
                            if "stats" in file_result:
                                self.processing_stats["processed_files"] += 1
                                self.processing_stats["total_rows_processed"] += file_result["stats"].get("total_rows_processed", 0)
                                self.processing_stats["file_stats"][file_path] = file_result["stats"]
                                
                                # Update memory peak
                                if file_result["stats"].get("memory_peak", 0) > self.processing_stats["memory_peak"]:
                                    self.processing_stats["memory_peak"] = file_result["stats"]["memory_peak"]
                            else:
                                self.processing_stats["failed_files"] += 1
                                self.processing_stats["errors"].append(f"No stats for file: {file_path}")
                    
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        with self._lock:
                            self.processing_stats["failed_files"] += 1
                            self.processing_stats["errors"].append(f"File {file_path}: {str(e)}")
            
            # Update results with this batch
            results.update(batch_results)
            
            # Clean up memory between batches
            self.memory_monitor.cleanup_memory(aggressive=True)
        
        return results
        
    def _process_single_file(self, file_path: str, 
                           process_fn: Optional[callable], 
                           output_dir: Optional[str]) -> Dict[str, Any]:
        """
        Process a single file (for parallel processing).
        
        Args:
            file_path: Path to the file
            process_fn: Optional function to apply to each chunk
            output_dir: Optional directory to save results
            
        Returns:
            Processing result for the file
        """
        try:
            # Get processor for this file
            processor = self._get_processor_for_file(file_path)
            
            if processor:
                # Process the file
                output_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, os.path.basename(file_path))
                    
                return processor.process_file(file_path, process_fn=process_fn, 
                                            output_path=output_path)
            else:
                # Return error for unsupported file type
                return {
                    "error": f"No processor available for file: {file_path}",
                    "stats": {
                        "status": "failed",
                        "error": "Unsupported file type"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in _process_single_file for {file_path}: {str(e)}")
            return {
                "error": str(e),
                "stats": {
                    "status": "failed",
                    "error": str(e)
                }
            }