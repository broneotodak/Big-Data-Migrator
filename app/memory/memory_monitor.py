"""
Memory monitoring capabilities for real-time RAM usage tracking and management.
"""
import os
import gc
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import psutil
import pandas as pd

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class MemoryMonitor:
    """
    Monitors real-time RAM usage and provides memory management utilities.
    
    This class tracks system memory, calculates safe file size limits, provides
    warnings when memory usage exceeds thresholds, and implements cleanup methods.
    """
    
    # Default warning thresholds
    WARNING_THRESHOLD = 0.75  # 75% RAM usage triggers warning
    CRITICAL_THRESHOLD = 0.90  # 90% RAM usage is critical
    
    # Memory tracking constants
    MEMORY_OVERHEAD_FACTOR = 1.5  # Account for Python memory overhead
    PANDAS_MEMORY_MULTIPLIER = 5  # Pandas typically uses 5x the raw file size in memory
    
    def __init__(self, 
                 warning_threshold: float = WARNING_THRESHOLD, 
                 critical_threshold: float = CRITICAL_THRESHOLD,
                 monitor_interval: float = 5.0,
                 auto_monitoring: bool = False):
        """
        Initialize the memory monitor.
        
        Args:
            warning_threshold: RAM usage percentage that triggers warnings
            critical_threshold: RAM usage percentage that triggers critical alerts
            monitor_interval: Seconds between memory checks when auto-monitoring
            auto_monitoring: Whether to automatically start background monitoring
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitor_interval = monitor_interval
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._process = psutil.Process(os.getpid())
        self._memory_usage_history = []  # List of (timestamp, usage_mb) tuples
        self._step_memory_usage = {}  # Dict of processing steps and their memory usage
        
        # Start auto-monitoring if requested
        if auto_monitoring:
            self.start_monitoring()
            
    def get_memory_usage(self) -> Tuple[float, float, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Tuple containing:
            - Current system memory usage in MB
            - Percentage of total system RAM in use
            - Total available system RAM in MB
        """
        # Get system memory info
        system_memory = psutil.virtual_memory()
        total_ram_mb = system_memory.total / (1024 * 1024)
        
        # Get system-wide memory usage (not just our process)
        used_ram_mb = system_memory.used / (1024 * 1024)
        usage_percent = system_memory.percent / 100  # psutil returns percentage, convert to decimal
        
        # Get current process memory usage for internal tracking
        process_usage_mb = self._process.memory_info().rss / (1024 * 1024)
        
        # Record process usage in history (for process-specific tracking)
        self._memory_usage_history.append((time.time(), process_usage_mb))
        
        # Return system-wide usage for main metrics
        return used_ram_mb, usage_percent, total_ram_mb
    
    def current_time_ms(self) -> int:
        """
        Get current time in milliseconds.
        
        Returns:
            Current timestamp in milliseconds
        """
        return int(time.time() * 1000)
    
    def get_max_safe_file_size(self, file_type: str = "csv") -> float:
        """
        Calculate the maximum safe file size based on available memory.
        
        Args:
            file_type: Type of file ("csv", "excel", "json", etc.)
        
        Returns:
            Maximum recommended file size in MB
        """
        # Get available memory
        _, usage_percent, total_ram_mb = self.get_memory_usage()
        available_ram_mb = total_ram_mb * (1 - usage_percent)
        
        # Apply safety factor based on file type
        # Different file types need different amounts of memory for processing
        safety_factors = {
            "csv": 0.5,
            "excel": 0.4,  # Excel files need more overhead
            "json": 0.6,
            "parquet": 0.7,  # Parquet is more efficient
            "default": 0.5
        }
        
        safety_factor = safety_factors.get(file_type.lower(), safety_factors["default"])
        
        # Calculate max safe size considering pandas memory multiplier
        safe_size_mb = (available_ram_mb * safety_factor) / self.PANDAS_MEMORY_MULTIPLIER
        
        logger.info(f"Maximum safe {file_type} file size: {safe_size_mb:.2f} MB")
        return safe_size_mb
    
    def check_memory_status(self) -> str:
        """
        Check memory status and return appropriate warning level.
        
        Returns:
            Status string: "normal", "warning", or "critical"
        """
        _, usage_percent, _ = self.get_memory_usage()
        
        if usage_percent >= self.critical_threshold:
            logger.warning(f"CRITICAL: Memory usage at {usage_percent*100:.1f}% - Immediate action required")
            return "critical"
        elif usage_percent >= self.warning_threshold:
            logger.warning(f"WARNING: Memory usage at {usage_percent*100:.1f}% - Consider cleanup")
            return "warning"
        else:
            return "normal"
    
    def cleanup_memory(self, aggressive: bool = False) -> float:
        """
        Perform memory cleanup and garbage collection.
        
        Args:
            aggressive: Whether to perform more aggressive cleanup
        
        Returns:
            Amount of memory freed in MB
        """
        # Record memory before cleanup
        before_mb, _, _ = self.get_memory_usage()
        
        # Run garbage collection
        gc.collect()
        
        if aggressive:
            # For more aggressive cleanup, we can suggest clearing pandas cache
            # and releasing unused dataframes
            import pandas as pd
            for obj in gc.get_objects():
                if isinstance(obj, pd.DataFrame):
                    if hasattr(obj, "_memory_usage"):
                        del obj
            gc.collect()
        
        # Record memory after cleanup
        after_mb, _, _ = self.get_memory_usage()
        freed_mb = before_mb - after_mb
        
        logger.info(f"Memory cleanup complete. Freed {freed_mb:.2f} MB")
        return freed_mb
    
    def start_tracking_step(self, step_name: str) -> None:
        """
        Start tracking memory usage for a specific processing step.
        
        Args:
            step_name: Name of the processing step
        """
        current_mb, _, _ = self.get_memory_usage()
        self._step_memory_usage[step_name] = {"start": current_mb, "peak": current_mb}
        logger.debug(f"Started tracking memory for step: {step_name}")
    
    def end_tracking_step(self, step_name: str) -> Dict[str, float]:
        """
        End tracking memory usage for a specific processing step.
        
        Args:
            step_name: Name of the processing step
        
        Returns:
            Dict with memory usage statistics for the step
        """
        if step_name not in self._step_memory_usage:
            logger.warning(f"Step {step_name} was not being tracked")
            return {}
            
        current_mb, _, _ = self.get_memory_usage()
        start_mb = self._step_memory_usage[step_name]["start"]
        peak_mb = self._step_memory_usage[step_name]["peak"]
        
        result = {
            "start_mb": start_mb,
            "end_mb": current_mb,
            "peak_mb": max(peak_mb, current_mb),
            "delta_mb": current_mb - start_mb
        }
        
        self._step_memory_usage[step_name].update({"end": current_mb, "delta": current_mb - start_mb})
        logger.info(f"Memory usage for step '{step_name}': {result['delta_mb']:.2f} MB")
        
        return result
    
    def update_step_peak(self, step_name: str) -> None:
        """Update the peak memory usage for a step that's being tracked"""
        if step_name in self._step_memory_usage:
            current_mb, _, _ = self.get_memory_usage()
            self._step_memory_usage[step_name]["peak"] = max(
                self._step_memory_usage[step_name]["peak"], 
                current_mb
            )
    
    def get_memory_report(self) -> Dict:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Dictionary containing memory statistics and history
        """
        current_mb, usage_percent, total_ram_mb = self.get_memory_usage()
        
        # Get process-specific memory for additional tracking
        process_usage_mb = self._process.memory_info().rss / (1024 * 1024)
        
        report = {
            "current_usage_mb": current_mb,  # System-wide used memory
            "usage_percent": usage_percent * 100,  # System-wide usage percentage
            "total_ram_mb": total_ram_mb,
            "available_mb": total_ram_mb * (1 - usage_percent),  # System-wide available memory
            "process_memory_mb": process_usage_mb,  # Our process memory usage
            "step_memory": self._step_memory_usage,
            "history": self._memory_usage_history[-50:] if self._memory_usage_history else []
        }
        
        return report
    
    def get_available_memory_mb(self) -> float:
        """
        Get available memory in MB.
        
        Returns:
            Available memory in MB
        """
        _, usage_percent, total_ram_mb = self.get_memory_usage()
        available_mb = total_ram_mb * (1 - usage_percent)
        return available_mb
    
    def get_peak_memory_usage(self) -> float:
        """
        Get peak memory usage from all tracked steps.
        
        Returns:
            Peak memory usage in MB
        """
        peak_mb = 0
        for step_data in self._step_memory_usage.values():
            if "peak" in step_data:
                peak_mb = max(peak_mb, step_data["peak"])
        
        # If no steps tracked, return current usage
        if peak_mb == 0:
            current_mb, _, _ = self.get_memory_usage()
            peak_mb = current_mb
            
        return peak_mb
    
    def _monitor_memory(self) -> None:
        """Background thread function for continuous memory monitoring"""
        logger.info("Starting background memory monitoring")
        
        while not self._stop_monitoring.is_set():
            status = self.check_memory_status()
            
            # Update peak memory usage for any active steps
            for step_name in self._step_memory_usage:
                if "end" not in self._step_memory_usage[step_name]:
                    self.update_step_peak(step_name)
            
            # If memory status is critical, attempt cleanup
            if status == "critical":
                self.cleanup_memory(aggressive=True)
                
            # Sleep for the specified interval
            time.sleep(self.monitor_interval)
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring thread"""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Memory monitoring already running")
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory, 
            daemon=True,
            name="MemoryMonitorThread"
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring thread"""
        if self._monitoring_thread is None:
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
        
    def __del__(self):
        """Cleanup when the object is deleted"""
        self.stop_monitoring()