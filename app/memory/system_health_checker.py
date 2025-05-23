"""
System health monitoring and capacity planning utilities.
"""
import os
import platform
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import psutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class SystemHealthChecker:
    """
    Monitors system resources and provides recommendations.
    
    This class monitors CPU, RAM, and disk usage, provides system recommendations
    before processing, implements automatic resource scaling, and generates
    system capacity reports.
    """
    
    # Resource thresholds
    CPU_WARNING_THRESHOLD = 80.0  # Percentage
    RAM_WARNING_THRESHOLD = 85.0  # Percentage
    DISK_WARNING_THRESHOLD = 90.0  # Percentage
    
    # Minimum requirements for processing
    MIN_FREE_RAM_MB = 1000.0  # Minimum 1GB free RAM
    MIN_FREE_DISK_MB = 2000.0  # Minimum 2GB free disk
    
    def __init__(self, 
                monitoring_interval: float = 5.0,
                history_length: int = 60):
        """
        Initialize the system health checker.
        
        Args:
            monitoring_interval: Seconds between health checks
            history_length: Number of data points to keep in history
        """
        self.monitoring_interval = monitoring_interval
        self.history_length = history_length
        self._monitoring = False
        self._history = {
            "timestamps": [],
            "cpu": [],
            "ram": [],
            "ram_available_mb": [],
            "disk": [],
            "disk_available_mb": []
        }
        
    def get_system_info(self) -> Dict:
        """
        Get general system information.
        
        Returns:
            Dictionary with system specifications
        """
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_cores": {
                "physical": psutil.cpu_count(logical=False),
                "logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024)
            },
            "disk": {
                "total_mb": disk.total / (1024 * 1024),
                "free_mb": disk.free / (1024 * 1024)
            },
            "datetime": datetime.now().isoformat()
        }
        
        return info
        
    def check_current_usage(self) -> Dict:
        """
        Check current system resource usage.
        
        Returns:
            Dictionary with current resource usage stats
        """
        # Get current resource usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Store in history if we're monitoring
        if self._monitoring:
            now = datetime.now().isoformat()
            
            # Keep history length in check
            if len(self._history["timestamps"]) >= self.history_length:
                for key in self._history:
                    self._history[key] = self._history[key][-self.history_length:]
            
            self._history["timestamps"].append(now)
            self._history["cpu"].append(cpu_percent)
            self._history["ram"].append(memory.percent)
            self._history["ram_available_mb"].append(memory.available / (1024 * 1024))
            self._history["disk"].append(disk.percent)
            self._history["disk_available_mb"].append(disk.free / (1024 * 1024))
            
        # Return current stats
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_mb": disk.free / (1024 * 1024),
            "datetime": datetime.now().isoformat()
        }
        
    def get_usage_history(self) -> Dict:
        """
        Get historical resource usage data.
        
        Returns:
            Dictionary with usage history
        """
        return self._history
        
    def analyze_system_health(self) -> Dict:
        """
        Analyze system health status.
        
        Returns:
            Dictionary with health analysis and recommendations
        """
        current = self.check_current_usage()
        
        # Determine health status for each component
        cpu_status = "good" if current["cpu_percent"] < self.CPU_WARNING_THRESHOLD else "warning"
        ram_status = "good" if current["memory_percent"] < self.RAM_WARNING_THRESHOLD else "warning"
        disk_status = "good" if current["disk_percent"] < self.DISK_WARNING_THRESHOLD else "warning"
        
        # Generate recommendations based on status
        recommendations = []
        
        if cpu_status == "warning":
            recommendations.append("CPU usage is high. Consider closing other applications or reducing processing load.")
            
        if ram_status == "warning":
            recommendations.append("Memory usage is high. Consider freeing up RAM or increasing chunk size for processing.")
            
        if disk_status == "warning":
            recommendations.append("Disk space is running low. Free up space before processing large files.")
            
        if current["memory_available_mb"] < self.MIN_FREE_RAM_MB:
            recommendations.append(f"Less than {self.MIN_FREE_RAM_MB} MB RAM available. Processing may be severely impacted.")
            
        if current["disk_free_mb"] < self.MIN_FREE_DISK_MB:
            recommendations.append(f"Less than {self.MIN_FREE_DISK_MB} MB disk space available. Processing may fail.")
            
        # Overall status is the worst of all components
        overall_status = "good"
        if "warning" in [cpu_status, ram_status, disk_status]:
            overall_status = "warning"
            
        result = {
            "overall": overall_status,
            "components": {
                "cpu": cpu_status,
                "ram": ram_status,
                "disk": disk_status
            },
            "recommendations": recommendations,
            "current_usage": current
        }
        
        return result
        
    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        self._monitoring = True
        logger.info("System health monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring"""
        self._monitoring = False
        logger.info("System health monitoring stopped")
        
    def generate_resource_recommendations(self, file_size_mb: float, operation_type: str = "general") -> Dict:
        """
        Generate resource usage recommendations for processing a file.
        
        Args:
            file_size_mb: Size of the file in MB
            operation_type: Type of operation ("general", "etl", "llm", "export")
            
        Returns:
            Dictionary with recommendations
        """
        # Get current system state
        health = self.analyze_system_health()
        current = health["current_usage"]
        
        # Different operation types require different resource estimates
        operation_factors = {
            "general": 5.0,  # General processing needs ~5x file size in RAM
            "etl": 6.0,      # ETL operations need more memory
            "llm": 10.0,     # LLM operations are very memory intensive
            "export": 3.0    # Export operations need less memory
        }
        
        factor = operation_factors.get(operation_type.lower(), operation_factors["general"])
        
        # Estimate resource needs
        estimated_ram_mb = file_size_mb * factor
        estimated_disk_mb = file_size_mb * 2  # Temporary files
        
        # Check if resources are sufficient
        ram_sufficient = current["memory_available_mb"] >= estimated_ram_mb
        disk_sufficient = current["disk_free_mb"] >= estimated_disk_mb
        
        # Generate recommendations
        recommendations = []
        
        if not ram_sufficient:
            deficit_mb = estimated_ram_mb - current["memory_available_mb"]
            recommendations.append(f"Insufficient RAM: Need {estimated_ram_mb:.1f} MB, have {current['memory_available_mb']:.1f} MB. "
                                 f"Free up {deficit_mb:.1f} MB or process in smaller chunks.")
            
        if not disk_sufficient:
            deficit_mb = estimated_disk_mb - current["disk_free_mb"]
            recommendations.append(f"Insufficient disk space: Need {estimated_disk_mb:.1f} MB, have {current['disk_free_mb']:.1f} MB. "
                                 f"Free up {deficit_mb:.1f} MB before processing.")
                                 
        if current["cpu_percent"] > self.CPU_WARNING_THRESHOLD:
            recommendations.append(f"CPU usage is high ({current['cpu_percent']:.1f}%). Processing may be slower than expected.")
            
        # Calculate optimal chunk size if needed
        optimal_chunk_size_mb = None
        optimal_chunks_count = None
        
        if not ram_sufficient:
            # If RAM is insufficient, calculate chunk size
            safe_ram_mb = current["memory_available_mb"] * 0.7  # Use 70% of available RAM
            optimal_chunk_size_mb = safe_ram_mb / factor
            optimal_chunks_count = file_size_mb / optimal_chunk_size_mb
            
            recommendations.append(f"Process in {optimal_chunks_count:.1f} chunks of {optimal_chunk_size_mb:.1f} MB each.")
            
        result = {
            "sufficient_resources": ram_sufficient and disk_sufficient,
            "estimated_needs": {
                "ram_mb": estimated_ram_mb,
                "disk_mb": estimated_disk_mb
            },
            "current_resources": {
                "ram_available_mb": current["memory_available_mb"],
                "disk_free_mb": current["disk_free_mb"]
            },
            "recommendations": recommendations
        }
        
        if optimal_chunk_size_mb is not None:
            result["chunk_recommendation"] = {
                "optimal_chunk_size_mb": optimal_chunk_size_mb,
                "optimal_chunks_count": optimal_chunks_count
            }
            
        return result
        
    def generate_system_capacity_report(self, 
                                       include_history: bool = True,
                                       include_charts: bool = False,
                                       output_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive system capacity report.
        
        Args:
            include_history: Whether to include historical data
            include_charts: Whether to generate charts
            output_path: Optional path to save the report
            
        Returns:
            Dictionary with system capacity report
        """
        # Get basic system info
        system_info = self.get_system_info()
        current_usage = self.check_current_usage()
        health_analysis = self.analyze_system_health()
        
        # Get process info
        process = psutil.Process(os.getpid())
        process_info = {
            "memory_info": {
                "rss_mb": process.memory_info().rss / (1024 * 1024),
                "vms_mb": process.memory_info().vms / (1024 * 1024),
            },
            "cpu_percent": process.cpu_percent(interval=1.0),
            "threads": len(process.threads()),
            "open_files": len(process.open_files()),
            "started": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        # Create basic report
        report = {
            "report_time": datetime.now().isoformat(),
            "system_info": system_info,
            "current_usage": current_usage,
            "health_analysis": health_analysis,
            "process_info": process_info,
            "recommendations": health_analysis["recommendations"]
        }
        
        # Add history if requested
        if include_history and self._history["timestamps"]:
            report["history"] = self.get_usage_history()
            
        # Generate charts if requested
        if include_charts and self._history["timestamps"]:
            chart_paths = self._generate_usage_charts(output_path)
            report["charts"] = chart_paths
            
        # Save report if path provided
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            logger.info(f"System capacity report saved to {output_path}")
            
        return report
        
    def _generate_usage_charts(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate usage history charts.
        
        Args:
            output_dir: Directory to save charts (None for memory-only)
            
        Returns:
            Dictionary with chart paths or data
        """
        if not self._history["timestamps"]:
            return {}
            
        # Convert timestamps to datetime objects
        timestamps = [datetime.fromisoformat(ts) for ts in self._history["timestamps"]]
        
        # Create charts directory if saving
        chart_paths = {}
        if output_dir:
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
        # Create CPU usage chart
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, self._history["cpu"], 'b-', label='CPU Usage')
        plt.axhline(y=self.CPU_WARNING_THRESHOLD, color='r', linestyle='--', label=f'Warning Threshold ({self.CPU_WARNING_THRESHOLD}%)')
        plt.title('CPU Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            cpu_chart_path = os.path.join(charts_dir, 'cpu_usage.png')
            plt.savefig(cpu_chart_path)
            chart_paths['cpu'] = cpu_chart_path
            
        plt.close()
        
        # Create RAM usage chart
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, self._history["ram"], 'g-', label='RAM Usage')
        plt.axhline(y=self.RAM_WARNING_THRESHOLD, color='r', linestyle='--', label=f'Warning Threshold ({self.RAM_WARNING_THRESHOLD}%)')
        plt.title('RAM Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('RAM Usage (%)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            ram_chart_path = os.path.join(charts_dir, 'ram_usage.png')
            plt.savefig(ram_chart_path)
            chart_paths['ram'] = ram_chart_path
            
        plt.close()
        
        # Create Disk usage chart
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, self._history["disk"], 'm-', label='Disk Usage')
        plt.axhline(y=self.DISK_WARNING_THRESHOLD, color='r', linestyle='--', label=f'Warning Threshold ({self.DISK_WARNING_THRESHOLD}%)')
        plt.title('Disk Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Disk Usage (%)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            disk_chart_path = os.path.join(charts_dir, 'disk_usage.png')
            plt.savefig(disk_chart_path)
            chart_paths['disk'] = disk_chart_path
            
        plt.close()
        
        return chart_paths
        
    def perform_stress_test(self, 
                          duration_sec: int = 30, 
                          file_size_mb: float = 100.0) -> Dict:
        """
        Perform a stress test to see how the system handles load.
        
        Args:
            duration_sec: Duration of the test in seconds
            file_size_mb: Simulated file size in MB
            
        Returns:
            Dictionary with stress test results
        """
        logger.info(f"Starting stress test for {duration_sec} seconds with simulated {file_size_mb} MB file...")
        
        # Start monitoring
        self.start_monitoring()
        
        # Record initial state
        initial_usage = self.check_current_usage()
        
        # Perform stress test
        stress_results = []
        
        start_time = time.time()
        end_time = start_time + duration_sec
        
        # Simulate load with progressbar
        with tqdm(total=duration_sec, desc="Stress test") as pbar:
            while time.time() < end_time:
                # Check usage
                usage = self.check_current_usage()
                stress_results.append(usage)
                
                # Simulate memory allocation to create load
                # We're creating a temporary large list to simulate memory usage
                # The size is scaled down to avoid crashing the system
                test_data_size = int(file_size_mb * 1024 * 50)  # Scale down to avoid actual crash
                test_data = [0] * test_data_size
                
                # Simulate CPU usage
                for _ in range(1000000):
                    _ = 1 + 1
                
                # Clean up
                del test_data
                
                # Update progress
                elapsed = time.time() - start_time
                pbar.update(min(1, elapsed - pbar.n))
                
                # Sleep briefly
                time.sleep(0.1)
        
        # Record final state
        final_usage = self.check_current_usage()
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate peak resource usage
        peak_cpu = max(point["cpu_percent"] for point in stress_results)
        peak_ram = max(point["memory_percent"] for point in stress_results)
        peak_ram_used_mb = max(point["memory_available_mb"] for point in stress_results)
        
        # Generate report
        report = {
            "duration_sec": duration_sec,
            "simulated_file_size_mb": file_size_mb,
            "initial_state": initial_usage,
            "final_state": final_usage,
            "peak_usage": {
                "cpu_percent": peak_cpu,
                "ram_percent": peak_ram,
                "ram_used_mb": peak_ram_used_mb
            },
            "resource_delta": {
                "cpu_percent": final_usage["cpu_percent"] - initial_usage["cpu_percent"],
                "ram_percent": final_usage["memory_percent"] - initial_usage["memory_percent"],
                "ram_mb": initial_usage["memory_available_mb"] - final_usage["memory_available_mb"]
            }
        }
        
        # Determine max safe file size
        if report["resource_delta"]["ram_mb"] > 0:
            # Based on RAM delta, estimate max file size
            ram_scaling_factor = file_size_mb / report["resource_delta"]["ram_mb"]
            max_safe_file_mb = (initial_usage["memory_available_mb"] * 0.7) * ram_scaling_factor
            
            report["estimated_max_file_size_mb"] = max_safe_file_mb
            logger.info(f"Estimated maximum safe file size: {max_safe_file_mb:.1f} MB")
        
        return report