"""
Resource Management System

This module provides comprehensive resource management for the video processing pipeline,
including GPU allocation, memory management, and dynamic resource scaling.
"""

import asyncio
import torch
import psutil
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import gc
import weakref

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    GPU_MEMORY = "gpu_memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceRequest:
    """Resource allocation request."""
    cpu_cores: float = 0.0
    memory_mb: float = 0.0
    gpu_devices: List[int] = field(default_factory=list)
    gpu_memory_mb: float = 0.0
    disk_space_mb: float = 0.0
    network_bandwidth_mbps: float = 0.0
    priority: int = 1  # 1=high, 5=low
    timeout_seconds: float = 300.0


@dataclass
class ResourceAllocation:
    """Allocated resources for a request."""
    allocation_id: str
    request: ResourceRequest
    allocated_cpu_cores: float
    allocated_memory_mb: float
    allocated_gpu_devices: List[int]
    allocated_gpu_memory_mb: float
    allocated_disk_space_mb: float
    timestamp: float
    expires_at: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        return self.expires_at is not None and time.time() > self.expires_at


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""
    cpu_usage_percent: float
    cpu_cores_available: float
    memory_usage_mb: float
    memory_available_mb: float
    gpu_usage_percent: Dict[int, float]
    gpu_memory_usage_mb: Dict[int, float]
    gpu_memory_available_mb: Dict[int, float]
    disk_usage_percent: float
    disk_available_mb: float
    network_usage_mbps: float
    timestamp: float


class GPUResourceManager:
    """GPU-specific resource management."""
    
    def __init__(self):
        self.device_locks = {}
        self.device_allocations = {}
        self.device_capabilities = {}
        
        # Initialize GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device_locks[i] = threading.RLock()
                self.device_allocations[i] = []
                
                # Get device capabilities
                props = torch.cuda.get_device_properties(i)
                self.device_capabilities[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory / (1024 ** 2),  # MB
                    'multiprocessor_count': props.multiprocessor_count,
                    'compute_capability': (props.major, props.minor)
                }
        
        logger.info(f"Initialized GPU manager with {len(self.device_capabilities)} devices")
    
    def get_available_devices(self) -> List[int]:
        """Get list of available GPU devices."""
        return list(self.device_capabilities.keys())
    
    def get_device_memory_usage(self, device_id: int) -> Tuple[float, float]:
        """Get GPU memory usage for a device."""
        if device_id not in self.device_capabilities:
            return 0.0, 0.0
        
        try:
            torch.cuda.set_device(device_id)
            used_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 2)  # MB
            total_memory = self.device_capabilities[device_id]['total_memory']
            return used_memory, total_memory - used_memory
        except Exception as e:
            logger.error(f"Failed to get GPU {device_id} memory usage: {e}")
            return 0.0, 0.0
    
    def allocate_device(self, device_id: int, memory_mb: float) -> bool:
        """Allocate GPU device with memory requirement."""
        if device_id not in self.device_capabilities:
            return False
        
        with self.device_locks[device_id]:
            used_memory, available_memory = self.get_device_memory_usage(device_id)
            
            if available_memory >= memory_mb:
                # Reserve memory (simplified)
                self.device_allocations[device_id].append({
                    'memory_mb': memory_mb,
                    'timestamp': time.time()
                })
                return True
        
        return False
    
    def deallocate_device(self, device_id: int, memory_mb: float):
        """Deallocate GPU device memory."""
        if device_id not in self.device_capabilities:
            return
        
        with self.device_locks[device_id]:
            # Remove allocation (simplified)
            allocations = self.device_allocations[device_id]
            for i, alloc in enumerate(allocations):
                if alloc['memory_mb'] == memory_mb:
                    allocations.pop(i)
                    break
    
    def get_optimal_device(self, memory_requirement_mb: float) -> Optional[int]:
        """Get optimal GPU device for memory requirement."""
        best_device = None
        best_score = -1
        
        for device_id in self.device_capabilities:
            used_memory, available_memory = self.get_device_memory_usage(device_id)
            
            if available_memory >= memory_requirement_mb:
                # Score based on available memory and utilization
                utilization = used_memory / self.device_capabilities[device_id]['total_memory']
                score = available_memory * (1.0 - utilization)
                
                if score > best_score:
                    best_score = score
                    best_device = device_id
        
        return best_device


class MemoryManager:
    """System memory management."""
    
    def __init__(self, max_memory_mb: float = 8192.0):
        self.max_memory_mb = max_memory_mb
        self.allocations = {}
        self.allocation_lock = threading.RLock()
        
        # Memory monitoring
        self.memory_pressure_threshold = 0.85  # 85%
        self.cleanup_callbacks = []
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage (used, available) in MB."""
        memory = psutil.virtual_memory()
        used_mb = (memory.total - memory.available) / (1024 ** 2)
        available_mb = memory.available / (1024 ** 2)
        return used_mb, available_mb
    
    def can_allocate(self, memory_mb: float) -> bool:
        """Check if memory can be allocated."""
        used_memory, available_memory = self.get_memory_usage()
        
        # Check system availability
        if available_memory < memory_mb:
            return False
        
        # Check against configured limit
        with self.allocation_lock:
            total_allocated = sum(self.allocations.values())
            if total_allocated + memory_mb > self.max_memory_mb:
                return False
        
        return True
    
    def allocate_memory(self, allocation_id: str, memory_mb: float) -> bool:
        """Allocate memory for an allocation ID."""
        if not self.can_allocate(memory_mb):
            return False
        
        with self.allocation_lock:
            self.allocations[allocation_id] = memory_mb
        
        # Check memory pressure
        self._check_memory_pressure()
        
        return True
    
    def deallocate_memory(self, allocation_id: str):
        """Deallocate memory for an allocation ID."""
        with self.allocation_lock:
            if allocation_id in self.allocations:
                del self.allocations[allocation_id]
        
        # Force garbage collection
        gc.collect()
    
    def _check_memory_pressure(self):
        """Check and respond to memory pressure."""
        used_memory, available_memory = self.get_memory_usage()
        total_memory = used_memory + available_memory
        memory_usage_ratio = used_memory / total_memory
        
        if memory_usage_ratio > self.memory_pressure_threshold:
            logger.warning(f"High memory pressure: {memory_usage_ratio:.1%}")
            
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Memory cleanup callback failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def register_cleanup_callback(self, callback):
        """Register callback for memory pressure cleanup."""
        self.cleanup_callbacks.append(callback)


class ResourceManager:
    """Main resource management system."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize sub-managers
        self.gpu_manager = GPUResourceManager()
        self.memory_manager = MemoryManager(config.max_memory_usage)
        
        # Resource tracking
        self.allocations = {}
        self.allocation_counter = 0
        self.allocation_lock = threading.RLock()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history = []
        self.max_history_size = 1000
        
        logger.info("ResourceManager initialized")
    
    async def allocate_resources(self, request: ResourceRequest) -> ResourceAllocation:
        """Allocate resources based on request."""
        allocation_id = f"alloc_{self.allocation_counter}"
        self.allocation_counter += 1
        
        # Check resource availability
        if not await self._check_resource_availability(request):
            raise RuntimeError("Insufficient resources available")
        
        # Allocate CPU (simplified - would use process affinity in production)
        allocated_cpu = min(request.cpu_cores, psutil.cpu_count())
        
        # Allocate memory
        if not self.memory_manager.allocate_memory(allocation_id, request.memory_mb):
            raise RuntimeError("Failed to allocate memory")
        
        # Allocate GPU devices
        allocated_gpu_devices = []
        allocated_gpu_memory = 0.0
        
        if request.gpu_devices:
            for device_id in request.gpu_devices:
                if self.gpu_manager.allocate_device(device_id, request.gpu_memory_mb):
                    allocated_gpu_devices.append(device_id)
                    allocated_gpu_memory += request.gpu_memory_mb
        
        # Create allocation record
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            request=request,
            allocated_cpu_cores=allocated_cpu,
            allocated_memory_mb=request.memory_mb,
            allocated_gpu_devices=allocated_gpu_devices,
            allocated_gpu_memory_mb=allocated_gpu_memory,
            allocated_disk_space_mb=request.disk_space_mb,
            timestamp=time.time(),
            expires_at=time.time() + request.timeout_seconds if request.timeout_seconds > 0 else None
        )
        
        with self.allocation_lock:
            self.allocations[allocation_id] = allocation
        
        logger.info(f"Allocated resources: {allocation_id}")
        return allocation
    
    async def deallocate_resources(self, allocation: ResourceAllocation):
        """Deallocate resources from an allocation."""
        allocation_id = allocation.allocation_id
        
        with self.allocation_lock:
            if allocation_id not in self.allocations:
                logger.warning(f"Allocation {allocation_id} not found for deallocation")
                return
            
            del self.allocations[allocation_id]
        
        # Deallocate memory
        self.memory_manager.deallocate_memory(allocation_id)
        
        # Deallocate GPU devices
        for device_id in allocation.allocated_gpu_devices:
            self.gpu_manager.deallocate_device(device_id, allocation.allocated_gpu_memory_mb)
        
        logger.info(f"Deallocated resources: {allocation_id}")
    
    async def _check_resource_availability(self, request: ResourceRequest) -> bool:
        """Check if requested resources are available."""
        
        # Check CPU
        if request.cpu_cores > psutil.cpu_count():
            return False
        
        # Check memory
        if not self.memory_manager.can_allocate(request.memory_mb):
            return False
        
        # Check GPU devices
        if request.gpu_devices:
            for device_id in request.gpu_devices:
                if device_id not in self.gpu_manager.device_capabilities:
                    return False
                
                used_memory, available_memory = self.gpu_manager.get_device_memory_usage(device_id)
                if available_memory < request.gpu_memory_mb:
                    return False
        
        return True
    
    def get_current_usage(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / (1024 ** 2)
        memory_available_mb = memory.available / (1024 ** 2)
        
        # GPU metrics
        gpu_usage = {}
        gpu_memory_usage = {}
        gpu_memory_available = {}
        
        for device_id in self.gpu_manager.get_available_devices():
            try:
                # GPU utilization (simplified)
                gpu_usage[device_id] = 0.0  # Would use nvidia-ml-py in production
                
                # GPU memory
                used_mem, avail_mem = self.gpu_manager.get_device_memory_usage(device_id)
                gpu_memory_usage[device_id] = used_mem
                gpu_memory_available[device_id] = avail_mem
                
            except Exception as e:
                logger.error(f"Failed to get GPU {device_id} metrics: {e}")
                gpu_usage[device_id] = 0.0
                gpu_memory_usage[device_id] = 0.0
                gpu_memory_available[device_id] = 0.0
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_available_mb = disk.free / (1024 ** 2)
        
        return ResourceMetrics(
            cpu_usage_percent=cpu_percent,
            cpu_cores_available=cpu_count,
            memory_usage_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            gpu_usage_percent=gpu_usage,
            gpu_memory_usage_mb=gpu_memory_usage,
            gpu_memory_available_mb=gpu_memory_available,
            disk_usage_percent=disk_percent,
            disk_available_mb=disk_available_mb,
            network_usage_mbps=0.0,  # Would implement network monitoring
            timestamp=time.time()
        )
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Resource monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.get_current_usage()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Check for resource issues
                self._check_resource_health(metrics)
                
                # Clean up expired allocations
                self._cleanup_expired_allocations()
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(interval_seconds)
    
    def _check_resource_health(self, metrics: ResourceMetrics):
        """Check resource health and log warnings."""
        
        # CPU health
        if metrics.cpu_usage_percent > 90:
            logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        # Memory health
        memory_usage_percent = (metrics.memory_usage_mb / 
                               (metrics.memory_usage_mb + metrics.memory_available_mb)) * 100
        if memory_usage_percent > 85:
            logger.warning(f"High memory usage: {memory_usage_percent:.1f}%")
        
        # GPU health
        for device_id, usage in metrics.gpu_usage_percent.items():
            if usage > 90:
                logger.warning(f"High GPU {device_id} usage: {usage:.1f}%")
            
            gpu_mem_used = metrics.gpu_memory_usage_mb[device_id]
            gpu_mem_total = gpu_mem_used + metrics.gpu_memory_available_mb[device_id]
            gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
            
            if gpu_mem_percent > 90:
                logger.warning(f"High GPU {device_id} memory usage: {gpu_mem_percent:.1f}%")
        
        # Disk health
        if metrics.disk_usage_percent > 90:
            logger.warning(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
    
    def _cleanup_expired_allocations(self):
        """Clean up expired resource allocations."""
        current_time = time.time()
        expired_allocations = []
        
        with self.allocation_lock:
            for allocation_id, allocation in list(self.allocations.items()):
                if allocation.is_expired:
                    expired_allocations.append(allocation)
        
        # Clean up expired allocations
        for allocation in expired_allocations:
            logger.info(f"Cleaning up expired allocation: {allocation.allocation_id}")
            asyncio.create_task(self.deallocate_resources(allocation))
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        with self.allocation_lock:
            total_allocations = len(self.allocations)
            
            # Calculate totals
            total_cpu = sum(a.allocated_cpu_cores for a in self.allocations.values())
            total_memory = sum(a.allocated_memory_mb for a in self.allocations.values())
            total_gpu_memory = sum(a.allocated_gpu_memory_mb for a in self.allocations.values())
            
            # Get oldest allocation
            oldest_allocation = min(
                (a.timestamp for a in self.allocations.values()),
                default=time.time()
            )
        
        return {
            'total_allocations': total_allocations,
            'total_allocated_cpu_cores': total_cpu,
            'total_allocated_memory_mb': total_memory,
            'total_allocated_gpu_memory_mb': total_gpu_memory,
            'oldest_allocation_age_seconds': time.time() - oldest_allocation,
            'metrics_history_size': len(self.metrics_history)
        }
    
    def optimize_allocations(self) -> List[str]:
        """Optimize current resource allocations."""
        optimization_suggestions = []
        
        current_metrics = self.get_current_usage()
        
        # Suggest GPU optimization
        if current_metrics.gpu_usage_percent:
            underutilized_gpus = [
                device_id for device_id, usage in current_metrics.gpu_usage_percent.items()
                if usage < 30.0
            ]
            
            if underutilized_gpus:
                optimization_suggestions.append(
                    f"Consider consolidating work from underutilized GPUs: {underutilized_gpus}"
                )
        
        # Suggest memory optimization
        memory_total = current_metrics.memory_usage_mb + current_metrics.memory_available_mb
        memory_usage = current_metrics.memory_usage_mb / memory_total
        
        if memory_usage > 0.8:
            optimization_suggestions.append(
                "Consider reducing batch sizes or enabling memory optimization"
            )
        
        return optimization_suggestions
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring() 