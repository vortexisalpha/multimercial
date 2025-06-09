"""
Video Advertisement Placement Pipeline

This module provides the main orchestration pipeline for end-to-end video advertisement
placement, including async processing, resource management, monitoring, and deployment.

Features:
- Modular architecture with pluggable components
- Efficient memory management and GPU utilization
- Progress tracking and error recovery
- Configurable quality/performance trade-offs
- Batch and streaming processing modes
- Comprehensive logging and monitoring
- Integration with cloud storage and APIs
- Scalable deployment architecture
"""

from .main_pipeline import (
    VideoAdPlacementPipeline, PipelineConfig, ProcessingResult,
    FrameResult, ProcessingEstimate, VideoInfo, VideoCharacteristics,
    PlacementConfig, ResourceUsage, QualityLevel
)

from .resource_manager import (
    ResourceManager, GPUResourceManager, MemoryManager,
    ResourceAllocation, ResourceMetrics
)

from .monitoring import (
    PipelineMonitor, MetricsCollector, HealthChecker,
    PerformanceProfiler, AlertManager
)

from .checkpointing import (
    CheckpointManager, ProcessingCheckpoint, CheckpointStrategy,
    RecoveryManager
)

from .streaming import (
    VideoStream, StreamProcessor, RealtimeProcessor,
    BufferManager, StreamConfig
)

from .cloud_integration import (
    CloudStorageManager, APIGateway, DistributedProcessor,
    CloudConfig, DeploymentManager
)

__version__ = "1.0.0"

__all__ = [
    # Main pipeline
    "VideoAdPlacementPipeline", "PipelineConfig", "ProcessingResult",
    "FrameResult", "ProcessingEstimate", "VideoInfo", "VideoCharacteristics",
    "PlacementConfig", "ResourceUsage", "QualityLevel",
    
    # Resource management
    "ResourceManager", "GPUResourceManager", "MemoryManager",
    "ResourceAllocation", "ResourceMetrics",
    
    # Monitoring and metrics
    "PipelineMonitor", "MetricsCollector", "HealthChecker",
    "PerformanceProfiler", "AlertManager",
    
    # Checkpointing and recovery
    "CheckpointManager", "ProcessingCheckpoint", "CheckpointStrategy",
    "RecoveryManager",
    
    # Streaming processing
    "VideoStream", "StreamProcessor", "RealtimeProcessor",
    "BufferManager", "StreamConfig",
    
    # Cloud integration
    "CloudStorageManager", "APIGateway", "DistributedProcessor",
    "CloudConfig", "DeploymentManager",
] 