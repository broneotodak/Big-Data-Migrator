"""
Processing orchestrator for managing and monitoring file processing operations.
"""
import os
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.memory.resource_optimizer import ResourceOptimizer
from app.processors.multi_file_processor import MultiFileProcessor

# Add PriorityError as a subclass of queue.Empty for priority queue specific errors
class PriorityError(queue.Empty):
    """Exception raised when priority queue operations fail due to corruption or other issues."""
    pass

# Monkey patch queue.PriorityQueue to catch "No lowest priority node found" error
original_get = queue.PriorityQueue.get
def safe_priority_get(self, *args, **kwargs):
    try:
        return original_get(self, *args, **kwargs)
    except Exception as e:
        if "No lowest priority node found" in str(e):
            raise PriorityError("No lowest priority node found, queue may be corrupted")
        raise
queue.PriorityQueue.get = safe_priority_get

logger = get_logger(__name__)

@dataclass
class ProcessingTask:
    """Data class representing a processing task."""
    task_id: str
    file_paths: List[str]
    task_type: str  # 'process', 'analyze', 'export', etc.
    priority: int = 0  # Lower number = higher priority
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class ProcessingOrchestrator:
    """
    Orchestrates file processing tasks with memory management and prioritization.
    
    This class:
    1. Manages memory allocation across multiple files and processing tasks
    2. Implements intelligent processing order based on file characteristics
    3. Provides real-time processing status and progress tracking
    4. Handles processing failures and recovery mechanisms
    """
    
    def __init__(self, 
                max_memory_percent: float = 80.0,  # Max % of available memory to use
                max_concurrent_tasks: int = 3,
                max_retries: int = 3,
                recovery_enabled: bool = True,
                show_progress: bool = True):
        """
        Initialize the processing orchestrator.
        
        Args:
            max_memory_percent: Maximum percentage of available memory to use
            max_concurrent_tasks: Maximum number of concurrent tasks
            max_retries: Maximum number of retry attempts for failed tasks
            recovery_enabled: Whether to enable recovery for failed tasks
            show_progress: Whether to show progress bars
        """
        # Memory management
        self.memory_monitor = MemoryMonitor(warning_threshold=max_memory_percent/100.0)
        self.resource_optimizer = ResourceOptimizer(
            memory_monitor=self.memory_monitor
        )
        
        # Settings
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_retries = max_retries
        self.recovery_enabled = recovery_enabled
        self.show_progress = show_progress
        
        # Create multi-file processor
        self.multi_processor = MultiFileProcessor(
            memory_monitor=self.memory_monitor,
            resource_optimizer=self.resource_optimizer,
            show_progress=show_progress
        )
        
        # Task management
        self._task_queue = queue.PriorityQueue()
        self._active_tasks = {}  # task_id -> ProcessingTask
        self._completed_tasks = {}  # task_id -> ProcessingTask
        self._failed_tasks = {}  # task_id -> (ProcessingTask, retry_count)
        
        # Status tracking
        self.processing_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_tasks": 0,
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_processing_time": 0,
            "memory_peak": 0,
        }
        
        # Locks for thread safety
        self._task_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # Processing thread
        self._processing_thread = None
        self._stop_event = threading.Event()
        
    def start(self):
        """Start the processing orchestrator."""
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning("Processing orchestrator already running")
            return
            
        # Reset stop event
        self._stop_event.clear()
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_task_queue,
            daemon=True,
            name="ProcessingOrchestratorThread"
        )
        self._processing_thread.start()
        
        logger.info("Processing orchestrator started")
        
    def stop(self, wait=True):
        """
        Stop the processing orchestrator.
        
        Args:
            wait: Whether to wait for active tasks to complete
        """
        # Signal processing thread to stop
        self._stop_event.set()
        
        if wait and self._processing_thread and self._processing_thread.is_alive():
            # Wait for processing thread to finish
            self._processing_thread.join()
            
        logger.info("Processing orchestrator stopped")
        
    def add_task(self, task: ProcessingTask) -> str:
        """
        Add a processing task to the queue.
        
        Args:
            task: The processing task to add
            
        Returns:
            Task ID
        """
        with self._task_lock:
            # Update task status
            task.status = "pending"
            
            # Add task to queue with priority
            self._task_queue.put((task.priority, task.task_id, task))
            
            # Update stats
            with self._stats_lock:
                self.processing_stats["total_tasks"] += 1
                self.processing_stats["total_files"] += len(task.file_paths)
            
            logger.info(f"Added task {task.task_id} with {len(task.file_paths)} files")
            
            return task.task_id
            
    def process_files(self, file_paths: List[str], 
                     process_fn: Optional[Callable] = None,
                     output_dir: Optional[str] = None,
                     task_id: Optional[str] = None,
                     priority: int = 0,
                     metadata: Optional[Dict] = None) -> str:
        """
        Add a file processing task to the queue.
        
        Args:
            file_paths: List of file paths to process
            process_fn: Function to apply to each file chunk
            output_dir: Directory to save output files
            task_id: Optional task ID (auto-generated if not provided)
            priority: Task priority (lower number = higher priority)
            metadata: Optional metadata for the task
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time())}_{len(file_paths)}"
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            file_paths=file_paths,
            task_type="process",
            priority=priority,
            metadata={
                "process_fn": process_fn.__name__ if process_fn else None,
                "output_dir": output_dir,
                **(metadata or {})
            }
        )
        
        # Add task to queue
        return self.add_task(task)
        
    def analyze_files(self, file_paths: List[str],
                     task_id: Optional[str] = None,
                     priority: int = 0,
                     metadata: Optional[Dict] = None) -> str:
        """
        Add a file analysis task to the queue.
        
        Args:
            file_paths: List of file paths to analyze
            task_id: Optional task ID (auto-generated if not provided)
            priority: Task priority (lower number = higher priority)
            metadata: Optional metadata for the task
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"analysis_{int(time.time())}_{len(file_paths)}"
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            file_paths=file_paths,
            task_type="analyze",
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add task to queue
        return self.add_task(task)
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dictionary
        """
        with self._task_lock:
            # Check active tasks
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                return self._create_task_status_dict(task)
                
            # Check completed tasks
            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                return self._create_task_status_dict(task)
                
            # Check failed tasks
            if task_id in self._failed_tasks:
                task, retry_count = self._failed_tasks[task_id]
                status_dict = self._create_task_status_dict(task)
                status_dict["retry_count"] = retry_count
                return status_dict
            
            # Task not found
            return {
                "task_id": task_id,
                "status": "not_found",
                "error": "Task not found"
            }
    
    def get_all_tasks_status(self) -> Dict[str, List[Dict]]:
        """
        Get status of all tasks.
        
        Returns:
            Dictionary with lists of active, completed, and failed tasks
        """
        with self._task_lock:
            # Get active tasks
            active_tasks = [
                self._create_task_status_dict(task)
                for task in self._active_tasks.values()
            ]
            
            # Get completed tasks
            completed_tasks = [
                self._create_task_status_dict(task)
                for task in self._completed_tasks.values()
            ]
            
            # Get failed tasks
            failed_tasks = [
                {**self._create_task_status_dict(task), "retry_count": retry_count}
                for task, retry_count in self._failed_tasks.values()
            ]
            
            return {
                "active": active_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "stats": self.processing_stats.copy()
            }
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._task_lock:
            # Task cannot be cancelled if already active
            if task_id in self._active_tasks:
                return False
                
            # Check if task is in failed tasks
            if task_id in self._failed_tasks:
                task, _ = self._failed_tasks.pop(task_id)
                with self._stats_lock:
                    self.processing_stats["failed_tasks"] -= 1
                return True
                
            # We can't directly remove from the queue, so we'll mark it for removal
            # when it comes up for processing
            for i in range(self._task_queue.qsize()):
                try:
                    priority, tid, task = self._task_queue.get()
                    if tid != task_id:
                        # Put it back
                        self._task_queue.put((priority, tid, task))
                    else:
                        # Found the task to cancel, don't put it back
                        with self._stats_lock:
                            self.processing_stats["total_tasks"] -= 1
                            self.processing_stats["total_files"] -= len(task.file_paths)
                        return True
                except queue.Empty:
                    break
            
            return False
    
    def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was queued for retry, False otherwise
        """
        with self._task_lock:
            # Check if task is in failed tasks
            if task_id in self._failed_tasks:
                task, retry_count = self._failed_tasks.pop(task_id)
                
                # Reset task status
                task.status = "pending"
                task.start_time = None
                task.end_time = None
                task.error = None
                
                # Add task to queue with slightly lower priority to avoid starvation
                self._task_queue.put((task.priority + 1, task.task_id, task))
                
                with self._stats_lock:
                    self.processing_stats["failed_tasks"] -= 1
                
                logger.info(f"Requeued failed task {task_id}")
                return True
                
            return False
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        with self._stats_lock:
            return self.processing_stats.copy()
            
    def _process_task_queue(self):
        """
        Process tasks from the queue in a background thread.
        """
        logger.info("Processing thread started")
        
        while not self._stop_event.is_set():
            try:
                # Get task from queue with 1 second timeout
                try:
                    priority, task_id, task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    # No tasks, continue checking stop event
                    continue
                
                # Additional validation to ensure we have valid task data
                if task is None or task_id is None:
                    logger.warning("Retrieved invalid task data from queue, skipping")
                    continue
                
                # Check if we have capacity to process this task
                if len(self._active_tasks) >= self.max_concurrent_tasks:
                    # Put task back in queue with same priority
                    try:
                        self._task_queue.put((priority, task_id, task))
                    except Exception as qe:
                        logger.error(f"Failed to requeue task {task_id}: {str(qe)}")
                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.5)
                    continue
                
                # Process the task
                self._execute_task(task)
                
            except PriorityError as pe:
                # Handle specific priority queue errors
                logger.error(f"Priority queue error: {str(pe)}")
                logger.warning("No lowest priority node found, queue may be corrupted")
                # Add recovery attempt - recreate queue if needed
                try:
                    with self._task_lock:
                        if self._task_queue.qsize() > 0:
                            logger.info("Attempting to recover task queue...")
                            # Create a new queue
                            new_queue = queue.PriorityQueue()
                            # Try to recover tasks
                            recovered = 0
                            while True:
                                try:
                                    item = self._task_queue.get_nowait()
                                    new_queue.put(item)
                                    recovered += 1
                                except queue.Empty:
                                    break
                                except Exception:
                                    # Skip problematic items
                                    continue
                            # Replace the queue
                            self._task_queue = new_queue
                            logger.info(f"Task queue recovered with {recovered} tasks")
                except Exception as recovery_error:
                    logger.error(f"Failed to recover task queue: {str(recovery_error)}")
                # Sleep before retrying
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in processing thread: {str(e)}")
                # Sleep briefly to avoid busy-waiting
                time.sleep(1.0)
        
        logger.info("Processing thread stopped")
    
    def _execute_task(self, task: ProcessingTask):
        """
        Execute a single task.
        
        Args:
            task: The task to execute
        """
        # Mark task as active
        with self._task_lock:
            self._active_tasks[task.task_id] = task
            with self._stats_lock:
                self.processing_stats["active_tasks"] += 1
        
        # Update task status
        task.status = "running"
        task.start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id} of type {task.task_type}")
            
            # Execute task based on type
            if task.task_type == "process":
                result = self._execute_processing_task(task)
            elif task.task_type == "analyze":
                result = self._execute_analysis_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Mark task as completed
            task.status = "completed"
            task.end_time = time.time()
            task.result = result
            
            # Update stats
            with self._stats_lock:
                self.processing_stats["completed_tasks"] += 1
                if "stats" in result:
                    self.processing_stats["processed_files"] += result["stats"].get("processed_files", 0)
                    
                    # Update memory peak if higher
                    if result["stats"].get("memory_peak", 0) > self.processing_stats["memory_peak"]:
                        self.processing_stats["memory_peak"] = result["stats"]["memory_peak"]
            
            # Log completion
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Mark task as failed
            task.status = "failed"
            task.end_time = time.time()
            task.error = str(e)
            
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            # Handle failure
            self._handle_task_failure(task)
            
        finally:
            # Update total processing time
            if task.start_time and task.end_time:
                with self._stats_lock:
                    self.processing_stats["total_processing_time"] += task.end_time - task.start_time
            
            # Remove from active tasks
            with self._task_lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                    
                # Add to completed or failed tasks collection
                if task.status == "completed":
                    self._completed_tasks[task.task_id] = task
                
                with self._stats_lock:
                    self.processing_stats["active_tasks"] -= 1
    
    def _handle_task_failure(self, task: ProcessingTask):
        """
        Handle a failed task.
        
        Args:
            task: The failed task
        """
        with self._task_lock:
            # Check if task should be retried
            retry_count = 0
            if task.task_id in self._failed_tasks:
                _, retry_count = self._failed_tasks[task.task_id]
            
            if self.recovery_enabled and retry_count < self.max_retries:
                # Increment retry count
                retry_count += 1
                
                # Store task with retry count
                self._failed_tasks[task.task_id] = (task, retry_count)
                
                # Log retry
                logger.info(f"Task {task.task_id} will be retried (attempt {retry_count} of {self.max_retries})")
                
                # Queue for retry with delay based on retry count
                # Add a small delay to avoid immediate retry
                retry_delay = min(5, retry_count * 2)  # 2, 4, 6... seconds up to 5 max
                
                # Start retry thread
                retry_thread = threading.Timer(
                    retry_delay,
                    self._retry_task,
                    args=(task, retry_count)
                )
                retry_thread.daemon = True
                retry_thread.start()
                
            else:
                # No more retries, mark as failed
                self._failed_tasks[task.task_id] = (task, retry_count)
                
                with self._stats_lock:
                    self.processing_stats["failed_tasks"] += 1
                    
                logger.warning(f"Task {task.task_id} failed permanently after {retry_count} retries")
    
    def _retry_task(self, task: ProcessingTask, retry_count: int):
        """
        Retry a failed task after a delay.
        
        Args:
            task: The task to retry
            retry_count: Current retry count
        """
        with self._task_lock:
            # Check if task is still in failed tasks
            if task.task_id not in self._failed_tasks:
                return
                
            # Reset task status
            task.status = "pending"
            task.start_time = None
            task.end_time = None
            task.error = None
            
            # Add task to queue with slightly lower priority to avoid starvation
            self._task_queue.put((task.priority + retry_count, task.task_id, task))
            
            logger.info(f"Requeued failed task {task.task_id} for retry {retry_count}")
    
    def _execute_processing_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """
        Execute a file processing task.
        
        Args:
            task: The task to execute
            
        Returns:
            Dictionary with processing results
        """
        # Extract parameters from task metadata
        process_fn = task.metadata.get("process_fn")
        output_dir = task.metadata.get("output_dir")
        
        # Process files
        return self.multi_processor.process_files(
            task.file_paths,
            process_fn=process_fn,
            output_dir=output_dir
        )
    
    def _execute_analysis_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """
        Execute a file analysis task.
        
        Args:
            task: The task to execute
            
        Returns:
            Dictionary with analysis results
        """
        # Analyze files
        results = {}
        file_stats = {}
        error_files = []
        
        for file_path in tqdm(task.file_paths, desc="Analyzing files", disable=not self.show_progress):
            try:
                # Get file info
                file_info = self.multi_processor.get_file_info(file_path)
                
                # Add to results
                results[file_path] = file_info
                file_stats[file_path] = {
                    "file_size_mb": file_info.get("file_size_mb", 0),
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                error_files.append({
                    "file_path": file_path,
                    "error": str(e)
                })
                file_stats[file_path] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Return analysis results
        return {
            "results": results,
            "stats": {
                "total_files": len(task.file_paths),
                "processed_files": len(task.file_paths) - len(error_files),
                "failed_files": len(error_files),
                "errors": error_files,
                "file_stats": file_stats,
                "memory_peak": self.memory_monitor.get_peak_memory_usage()
            }
        }
    
    def _create_task_status_dict(self, task: ProcessingTask) -> Dict[str, Any]:
        """
        Create a dictionary with task status information.
        
        Args:
            task: The task
            
        Returns:
            Dictionary with task status
        """
        status_dict = {
            "task_id": task.task_id,
            "status": task.status,
            "task_type": task.task_type,
            "file_count": len(task.file_paths),
        }
        
        # Add timing information if available
        if task.start_time:
            status_dict["start_time"] = task.start_time
            status_dict["elapsed_time"] = (task.end_time or time.time()) - task.start_time
            
        # Add completion information if available
        if task.end_time:
            status_dict["end_time"] = task.end_time
            status_dict["processing_time"] = task.end_time - task.start_time
            
        # Add error information if available
        if task.error:
            status_dict["error"] = task.error
            
        # Add result summary if available
        if task.result:
            if "stats" in task.result:
                status_dict["stats"] = task.result["stats"]
                
        return status_dict

    async def process_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Process a single file with its content.
        
        Args:
            filename: Name of the file
            content: File content as bytes
            
        Returns:
            Processing result dictionary compatible with ProcessingResponse
        """
        import uuid
        from datetime import datetime
        
        start_time = time.time()
        job_id = f"job_{int(start_time)}_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now()
        
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the file content to a temporary file
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            
            # Track memory usage for this operation
            self.memory_monitor.start_tracking_step(f"process_file_{filename}")
            
            # Process the file using multi-file processor
            result = self.multi_processor.process_file(temp_file_path)
            
            # End memory tracking
            memory_stats = self.memory_monitor.end_tracking_step(f"process_file_{filename}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            
            # Extract relevant data from the result
            rows_processed = result.get("stats", {}).get("total_rows_processed", 0)
            memory_peak_mb = memory_stats.get("peak_mb", 0)
            
            # Prepare response compatible with ProcessingResponse model
            response = {
                "job_id": job_id,
                "status": "completed",
                "file_path": temp_file_path,
                "output_path": None,  # No output file for this operation
                "processing_time_ms": processing_time_ms,
                "rows_processed": rows_processed,
                "file_size_mb": len(content) / (1024 * 1024),
                "memory_usage_peak_mb": memory_peak_mb,
                "error_message": None,
                "warnings": [],
                "metadata": {
                    "filename": filename,
                    "file_extension": os.path.splitext(filename)[1].lower(),
                    "processor_used": type(self.multi_processor).__name__,
                    "data_info": result.get("data_info", {}),
                    "statistics": result.get("statistics", {})
                },
                "created_at": created_at,
                "completed_at": datetime.now()
            }
            
            logger.info(f"Successfully processed file {filename} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # End memory tracking if it was started
            try:
                self.memory_monitor.end_tracking_step(f"process_file_{filename}")
            except:
                pass
            
            error_msg = f"Error processing file {filename}: {str(e)}"
            logger.error(error_msg)
            
            # Calculate processing time even for errors
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "job_id": job_id,
                "status": "failed",
                "file_path": None,
                "output_path": None,
                "processing_time_ms": processing_time_ms,
                "rows_processed": 0,
                "file_size_mb": len(content) / (1024 * 1024),
                "memory_usage_peak_mb": 0,
                "error_message": str(e),
                "warnings": [],
                "metadata": {
                    "filename": filename,
                    "error_details": error_msg
                },
                "created_at": created_at,
                "completed_at": datetime.now()
            }

    async def process_request(self, request) -> Dict[str, Any]:
        """
        Process a ProcessingRequest from the API.
        
        Args:
            request: ProcessingRequest object with processing parameters
            
        Returns:
            Processing result dictionary compatible with ProcessingResponse
        """
        import uuid
        from datetime import datetime
        
        start_time = time.time()
        job_id = f"job_{int(start_time)}_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now()
        
        try:
            # Validate that we have a file to process
            if not request.file_path:
                raise ValueError("No file path provided in the request")
                
            if not os.path.exists(request.file_path):
                raise FileNotFoundError(f"File not found: {request.file_path}")
            
            # Track memory usage for this operation
            self.memory_monitor.start_tracking_step(f"process_request_{job_id}")
            
            # Get file size
            file_size_bytes = os.path.getsize(request.file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Process the file using multi-file processor
            process_kwargs = {}
            
            # Add custom configuration if provided
            if request.custom_config:
                process_kwargs.update(request.custom_config)
            
            # Set output path based on request
            if request.export_path:
                os.makedirs(request.export_path, exist_ok=True)
                output_filename = os.path.basename(request.file_path)
                if request.output_format.value == "csv":
                    output_filename = os.path.splitext(output_filename)[0] + ".csv"
                elif request.output_format.value == "excel":
                    output_filename = os.path.splitext(output_filename)[0] + ".xlsx"
                elif request.output_format.value == "json":
                    output_filename = os.path.splitext(output_filename)[0] + ".json"
                
                process_kwargs["output_path"] = os.path.join(request.export_path, output_filename)
            
            # Override chunk size if specified
            if request.chunk_size:
                process_kwargs["chunk_size"] = request.chunk_size
            
            # Set processing mode parameters
            if request.processing_mode.value == "memory_optimized":
                process_kwargs["memory_efficient"] = True
                process_kwargs["use_chunking"] = True
            elif request.processing_mode.value == "speed_optimized":
                process_kwargs["memory_efficient"] = False
                process_kwargs["parallel_processing"] = True
            # balanced mode uses default settings
            
            # Process the file
            result = self.multi_processor.process_file(request.file_path, **process_kwargs)
            
            # End memory tracking
            memory_stats = self.memory_monitor.end_tracking_step(f"process_request_{job_id}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_time_ms = processing_time * 1000
            
            # Extract relevant data from the result
            rows_processed = result.get("stats", {}).get("total_rows_processed", 0)
            memory_peak_mb = memory_stats.get("peak_mb", 0)
            
            # Handle different output formats
            output_path = process_kwargs.get("output_path")
            if request.output_format.value == "supabase":
                # TODO: Implement Supabase export
                logger.warning("Supabase export not yet implemented")
                output_path = None
            
            # Prepare response compatible with ProcessingResponse model
            response = {
                "job_id": job_id,
                "status": "completed",
                "file_path": request.file_path,
                "output_path": output_path,
                "processing_time_ms": processing_time_ms,
                "rows_processed": rows_processed,
                "file_size_mb": file_size_mb,
                "memory_usage_peak_mb": memory_peak_mb,
                "error_message": None,
                "warnings": [],
                "metadata": {
                    "processing_mode": request.processing_mode.value,
                    "output_format": request.output_format.value,
                    "chunk_size": request.chunk_size,
                    "skip_validation": request.skip_validation,
                    "supabase_table_name": request.supabase_table_name,
                    "processor_used": type(self.multi_processor).__name__,
                    "data_info": result.get("data_info", {}),
                    "statistics": result.get("statistics", {})
                },
                "created_at": created_at,
                "completed_at": datetime.now()
            }
            
            logger.info(f"Successfully processed request for {request.file_path} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # End memory tracking if it was started
            try:
                self.memory_monitor.end_tracking_step(f"process_request_{job_id}")
            except:
                pass
            
            error_msg = f"Error processing request for {request.file_path}: {str(e)}"
            logger.error(error_msg)
            
            # Calculate processing time even for errors
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "job_id": job_id,
                "status": "failed",
                "file_path": request.file_path,
                "output_path": None,
                "processing_time_ms": processing_time_ms,
                "rows_processed": 0,
                "file_size_mb": 0,
                "memory_usage_peak_mb": 0,
                "error_message": str(e),
                "warnings": [],
                "metadata": {
                    "processing_mode": request.processing_mode.value if hasattr(request, 'processing_mode') else "unknown",
                    "error_details": error_msg
                },
                "created_at": created_at,
                "completed_at": datetime.now()
            }