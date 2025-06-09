"""
Main Video Advertisement Placement Pipeline

This module implements the core video processing pipeline that orchestrates all system
components for end-to-end video advertisement placement with advanced features including
async processing, resource management, monitoring, and error recovery.
"""

import asyncio
import torch
import cv2
import numpy as np
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncIterator, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

# Import all system components
from ..depth_estimation import DepthEstimator, MarigoldDepthEstimator, DepthProEstimator
from ..object_detection import YOLOv9Detector, ObjectDetectionConfig
from ..scene_understanding import (
    PlaneDetector, TemporalPlaneTracker, TemporalConsistencyManager,
    PlaneDetectionConfig, TemporalConfig, CameraParameters
)
from ..rendering import VideoRenderer, RenderingConfig, RenderingQuality
from .resource_manager import ResourceManager, ResourceAllocation, ResourceMetrics
from .monitoring import PipelineMonitor, MetricsCollector
from .checkpointing import CheckpointManager, ProcessingCheckpoint
from .streaming import VideoStream, BufferManager

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Processing quality levels for quality/performance trade-offs."""
    ULTRA_LOW = "ultra_low"      # Fastest processing, lowest quality
    LOW = "low"                  # Fast processing, low quality
    MEDIUM = "medium"            # Balanced processing and quality
    HIGH = "high"                # Slower processing, high quality
    ULTRA_HIGH = "ultra_high"    # Slowest processing, highest quality


class ProcessingMode(Enum):
    """Processing mode options."""
    BATCH = "batch"              # Process entire video at once
    STREAMING = "streaming"      # Process frame by frame
    REALTIME = "realtime"        # Real-time processing with strict timing
    DISTRIBUTED = "distributed"  # Distributed processing across multiple nodes


@dataclass
class VideoInfo:
    """Video file information and metadata."""
    path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    bitrate: int
    file_size: int
    has_audio: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoCharacteristics:
    """Video content characteristics for optimization."""
    complexity_score: float  # 0.0-1.0, higher = more complex
    motion_intensity: float  # 0.0-1.0, higher = more motion
    scene_changes: int       # Number of scene changes
    dominant_colors: List[Tuple[int, int, int]]
    lighting_stability: float  # 0.0-1.0, higher = more stable
    object_density: float    # 0.0-1.0, higher = more objects
    depth_variation: float   # 0.0-1.0, higher = more depth variation
    recommended_quality: QualityLevel


@dataclass
class PlacementConfig:
    """Configuration for TV advertisement placement."""
    target_wall_types: List[str] = field(default_factory=lambda: ["wall"])
    min_wall_area: float = 2.0  # Minimum wall area in square meters
    max_wall_distance: float = 5.0  # Maximum distance to wall
    tv_width: float = 1.2  # TV width in meters
    tv_height: float = 0.8  # TV height in meters
    placement_stability_threshold: float = 0.8
    quality_threshold: float = 0.7
    temporal_smoothing: bool = True
    lighting_adaptation: bool = True
    occlusion_handling: bool = True


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    cpu_usage: float
    memory_usage: float  # MB
    gpu_usage: float
    gpu_memory_usage: float  # MB
    disk_io: float  # MB/s
    network_io: float  # MB/s
    processing_time: float  # seconds
    peak_memory: float  # MB
    energy_consumption: float  # Joules (estimated)


@dataclass
class FrameResult:
    """Result for a single frame processing."""
    frame_id: int
    timestamp: float
    success: bool
    tv_placements: List[Any]  # TV placement results
    quality_score: float
    processing_time: float
    resource_usage: ResourceUsage
    error_message: Optional[str] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete processing result for video."""
    output_path: str
    processing_time: float
    total_frames: int
    successful_frames: int
    failed_frames: int
    quality_metrics: Dict[str, float]
    resource_usage: ResourceUsage
    error_log: List[str]
    checkpoints: List[str] = field(default_factory=list)
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        return self.successful_frames / self.total_frames if self.total_frames > 0 else 0.0


@dataclass
class ProcessingEstimate:
    """Processing time and resource estimates."""
    estimated_time: float  # seconds
    estimated_memory: float  # MB
    estimated_gpu_memory: float  # MB
    estimated_disk_space: float  # MB
    confidence: float  # 0.0-1.0
    bottleneck_component: str
    optimization_suggestions: List[str]


@dataclass
class PipelineConfig:
    """Comprehensive pipeline configuration."""
    
    # Quality and performance
    quality_level: QualityLevel = QualityLevel.MEDIUM
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    max_workers: int = 4
    use_gpu: bool = True
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    
    # Memory management
    max_memory_usage: float = 8192.0  # MB
    batch_size: int = 32
    prefetch_frames: int = 16
    cache_size: int = 1000  # Number of cached items
    
    # Processing components
    depth_estimator_type: str = "marigold"  # "marigold", "depth_pro"
    object_detector_config: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    plane_detection_config: PlaneDetectionConfig = field(default_factory=PlaneDetectionConfig)
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
    rendering_config: RenderingConfig = field(default_factory=RenderingConfig)
    
    # Error handling and recovery
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_interval: int = 100  # frames
    auto_recovery: bool = True
    fallback_quality: QualityLevel = QualityLevel.LOW
    
    # Monitoring and logging
    enable_profiling: bool = True
    log_level: str = "INFO"
    metrics_interval: float = 5.0  # seconds
    health_check_interval: float = 10.0  # seconds
    
    # Output configuration
    output_formats: List[str] = field(default_factory=lambda: ["mp4"])
    output_qualities: List[str] = field(default_factory=lambda: ["1080p"])
    intermediate_outputs: bool = False
    save_debug_data: bool = False
    
    # Cloud and distributed processing
    enable_cloud_storage: bool = False
    cloud_provider: str = "aws"  # "aws", "gcp", "azure"
    distributed_processing: bool = False
    scaling_policy: str = "auto"  # "auto", "manual"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.max_memory_usage <= 0:
            raise ValueError("max_memory_usage must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        
        return True


class VideoAdPlacementPipeline:
    """
    Main video advertisement placement pipeline with comprehensive orchestration.
    
    This pipeline coordinates all system components for end-to-end video processing,
    including depth estimation, object detection, scene understanding, temporal
    consistency, and video rendering with advanced features for production use.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the video processing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        config.validate()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self._initialize_components()
        
        # Initialize management systems
        self.resource_manager = ResourceManager(config)
        self.monitor = PipelineMonitor(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.buffer_manager = BufferManager(config)
        
        # State management
        self.is_processing = False
        self.current_video_info: Optional[VideoInfo] = None
        self.processing_stats = {
            'total_videos_processed': 0,
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'total_errors': 0
        }
        
        # Resource tracking
        self._resource_allocations: List[ResourceAllocation] = []
        self._cleanup_callbacks: List[callable] = []
        
        self.logger.info("VideoAdPlacementPipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize all processing components."""
        try:
            # Depth estimation
            if self.config.depth_estimator_type == "marigold":
                self.depth_estimator = MarigoldDepthEstimator()
            elif self.config.depth_estimator_type == "depth_pro":
                self.depth_estimator = DepthProEstimator()
            else:
                raise ValueError(f"Unknown depth estimator: {self.config.depth_estimator_type}")
            
            # Object detection
            self.object_detector = YOLOv9Detector(self.config.object_detector_config)
            
            # Scene understanding
            self.plane_detector = PlaneDetector(self.config.plane_detection_config)
            self.temporal_tracker = TemporalPlaneTracker()
            self.consistency_manager = TemporalConsistencyManager(self.config.temporal_config)
            
            # Video rendering
            self.video_renderer = VideoRenderer(self.config.rendering_config)
            
            # Thread pools for async processing
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
            self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.config.max_workers))
            
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def process_video(self, 
                           input_path: str,
                           output_path: str,
                           advertisement: torch.Tensor,
                           placement_config: PlacementConfig) -> ProcessingResult:
        """
        Process a complete video with advertisement placement.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            advertisement: Advertisement content tensor
            placement_config: TV placement configuration
            
        Returns:
            Complete processing result
        """
        start_time = time.time()
        self.is_processing = True
        
        try:
            # Extract video information
            video_info = await self._extract_video_info(input_path)
            self.current_video_info = video_info
            
            # Analyze video characteristics for optimization
            characteristics = await self._analyze_video_characteristics(input_path)
            
            # Optimize pipeline configuration
            optimized_config = self.optimize_pipeline(characteristics)
            
            # Estimate processing requirements
            estimate = self.estimate_processing_time(video_info)
            self.logger.info(f"Processing estimate: {estimate.estimated_time:.1f}s, "
                           f"Memory: {estimate.estimated_memory:.0f}MB")
            
            # Allocate resources
            allocation = await self.resource_manager.allocate_resources(estimate)
            self._resource_allocations.append(allocation)
            
            # Initialize monitoring
            self.monitor.start_monitoring(video_info, estimate)
            
            # Process based on mode
            if self.config.processing_mode == ProcessingMode.BATCH:
                result = await self._process_batch(
                    input_path, output_path, advertisement, placement_config
                )
            elif self.config.processing_mode == ProcessingMode.STREAMING:
                result = await self._process_streaming(
                    input_path, output_path, advertisement, placement_config
                )
            elif self.config.processing_mode == ProcessingMode.REALTIME:
                result = await self._process_realtime(
                    input_path, output_path, advertisement, placement_config
                )
            else:
                raise ValueError(f"Unsupported processing mode: {self.config.processing_mode}")
            
            # Finalize result
            result.processing_time = time.time() - start_time
            result.performance_profile = self.monitor.get_performance_profile()
            
            # Update statistics
            self.processing_stats['total_videos_processed'] += 1
            self.processing_stats['total_frames_processed'] += result.total_frames
            self.processing_stats['total_processing_time'] += result.processing_time
            
            self.logger.info(f"Video processing completed in {result.processing_time:.1f}s, "
                           f"Success rate: {result.success_rate:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            # Create error result
            result = ProcessingResult(
                output_path=output_path,
                processing_time=time.time() - start_time,
                total_frames=0,
                successful_frames=0,
                failed_frames=0,
                quality_metrics={},
                resource_usage=ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0),
                error_log=[str(e)]
            )
            return result
            
        finally:
            # Cleanup resources
            await self._cleanup_resources()
            self.is_processing = False
    
    async def process_video_stream(self,
                                  input_stream: VideoStream,
                                  output_stream: VideoStream,
                                  advertisement: torch.Tensor) -> AsyncIterator[FrameResult]:
        """
        Process video stream in real-time with frame-by-frame results.
        
        Args:
            input_stream: Input video stream
            output_stream: Output video stream
            advertisement: Advertisement content tensor
            
        Yields:
            Frame processing results
        """
        self.is_processing = True
        frame_id = 0
        
        try:
            # Initialize streaming components
            await self._initialize_streaming(input_stream, output_stream)
            
            async for frame in input_stream:
                frame_start_time = time.time()
                
                try:
                    # Process single frame
                    result = await self._process_frame(
                        frame, frame_id, advertisement
                    )
                    
                    # Write to output stream
                    if result.success and output_stream:
                        await output_stream.write_frame(result.intermediate_results['rendered_frame'])
                    
                    # Update monitoring
                    self.monitor.update_frame_metrics(result)
                    
                    yield result
                    
                except Exception as e:
                    # Create error result for this frame
                    error_result = FrameResult(
                        frame_id=frame_id,
                        timestamp=time.time(),
                        success=False,
                        tv_placements=[],
                        quality_score=0.0,
                        processing_time=time.time() - frame_start_time,
                        resource_usage=ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0),
                        error_message=str(e)
                    )
                    
                    self.logger.error(f"Frame {frame_id} processing failed: {e}")
                    yield error_result
                
                frame_id += 1
                
                # Check for termination conditions
                if not self.is_processing:
                    break
                    
        finally:
            await self._cleanup_streaming()
            self.is_processing = False
    
    def estimate_processing_time(self, video_info: VideoInfo) -> ProcessingEstimate:
        """
        Estimate processing time and resource requirements.
        
        Args:
            video_info: Video information
            
        Returns:
            Processing estimates
        """
        # Base processing time per frame (seconds)
        base_time_per_frame = {
            QualityLevel.ULTRA_LOW: 0.01,
            QualityLevel.LOW: 0.02,
            QualityLevel.MEDIUM: 0.05,
            QualityLevel.HIGH: 0.10,
            QualityLevel.ULTRA_HIGH: 0.20
        }[self.config.quality_level]
        
        # Resolution scaling factor
        resolution_factor = (video_info.width * video_info.height) / (1920 * 1080)
        
        # Complexity scaling (simplified estimation)
        complexity_factor = 1.0  # Would be based on video analysis
        
        # Calculate estimates
        time_per_frame = base_time_per_frame * resolution_factor * complexity_factor
        estimated_time = time_per_frame * video_info.total_frames
        
        # Memory estimates (MB)
        frame_memory = (video_info.width * video_info.height * 3 * 4) / (1024 * 1024)  # RGB float32
        estimated_memory = frame_memory * self.config.batch_size * 3  # Buffer overhead
        estimated_gpu_memory = estimated_memory * 2  # GPU processing overhead
        
        # Disk space for output
        estimated_disk_space = video_info.file_size * 1.2  # 20% overhead
        
        # Identify bottleneck
        bottleneck_component = "gpu" if self.config.use_gpu else "cpu"
        
        # Optimization suggestions
        suggestions = []
        if estimated_time > 300:  # 5 minutes
            suggestions.append("Consider using lower quality level for faster processing")
        if estimated_memory > self.config.max_memory_usage:
            suggestions.append("Reduce batch size to fit memory constraints")
        if not self.config.use_gpu:
            suggestions.append("Enable GPU processing for significant speedup")
        
        return ProcessingEstimate(
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            estimated_gpu_memory=estimated_gpu_memory,
            estimated_disk_space=estimated_disk_space,
            confidence=0.8,  # Would be improved with historical data
            bottleneck_component=bottleneck_component,
            optimization_suggestions=suggestions
        )
    
    def optimize_pipeline(self, video_characteristics: VideoCharacteristics) -> PipelineConfig:
        """
        Automatically optimize pipeline configuration for specific content.
        
        Args:
            video_characteristics: Video content analysis
            
        Returns:
            Optimized pipeline configuration
        """
        optimized_config = self.config
        
        # Adjust quality based on content complexity
        if video_characteristics.complexity_score < 0.3:
            # Simple content - can use higher quality
            if optimized_config.quality_level == QualityLevel.MEDIUM:
                optimized_config.quality_level = QualityLevel.HIGH
        elif video_characteristics.complexity_score > 0.8:
            # Complex content - may need lower quality for performance
            if optimized_config.quality_level == QualityLevel.HIGH:
                optimized_config.quality_level = QualityLevel.MEDIUM
        
        # Adjust batch size based on motion intensity
        if video_characteristics.motion_intensity > 0.7:
            # High motion - smaller batches for better temporal consistency
            optimized_config.batch_size = max(8, optimized_config.batch_size // 2)
        
        # Adjust temporal smoothing based on lighting stability
        if video_characteristics.lighting_stability < 0.5:
            # Unstable lighting - enable more aggressive smoothing
            optimized_config.temporal_config.lighting_smoothing_alpha = 0.9
        
        # Use recommended quality if specified
        if video_characteristics.recommended_quality:
            optimized_config.quality_level = video_characteristics.recommended_quality
        
        self.logger.info(f"Pipeline optimized for content complexity: {video_characteristics.complexity_score:.2f}")
        
        return optimized_config
    
    async def _process_batch(self,
                           input_path: str,
                           output_path: str,
                           advertisement: torch.Tensor,
                           placement_config: PlacementConfig) -> ProcessingResult:
        """Process video in batch mode."""
        self.logger.info("Starting batch processing")
        
        # Initialize result tracking
        frame_results = []
        error_log = []
        
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        try:
            frame_id = 0
            batch_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch_frames.append((frame_id, frame))
                
                # Process batch when full or at end
                if len(batch_frames) >= self.config.batch_size or frame_id == total_frames - 1:
                    batch_results = await self._process_frame_batch(
                        batch_frames, advertisement, placement_config
                    )
                    
                    # Write results and collect metrics
                    for result in batch_results:
                        frame_results.append(result)
                        
                        if result.success:
                            # Write frame to output
                            rendered_frame = result.intermediate_results.get('rendered_frame')
                            if rendered_frame is not None:
                                out.write(rendered_frame)
                        else:
                            error_log.append(f"Frame {result.frame_id}: {result.error_message}")
                    
                    # Clear batch
                    batch_frames = []
                    
                    # Update progress
                    progress = (frame_id + 1) / total_frames
                    self.monitor.update_progress(progress)
                    
                    # Checkpoint if needed
                    if frame_id % self.config.checkpoint_interval == 0:
                        await self._create_checkpoint(frame_id, frame_results)
                
                frame_id += 1
            
            # Calculate metrics
            successful_frames = sum(1 for r in frame_results if r.success)
            failed_frames = len(frame_results) - successful_frames
            
            # Aggregate resource usage
            total_resource_usage = self._aggregate_resource_usage(frame_results)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(frame_results)
            
            return ProcessingResult(
                output_path=output_path,
                processing_time=0.0,  # Will be set by caller
                total_frames=total_frames,
                successful_frames=successful_frames,
                failed_frames=failed_frames,
                quality_metrics=quality_metrics,
                resource_usage=total_resource_usage,
                error_log=error_log
            )
            
        finally:
            cap.release()
            out.release()
    
    async def _process_streaming(self,
                               input_path: str,
                               output_path: str,
                               advertisement: torch.Tensor,
                               placement_config: PlacementConfig) -> ProcessingResult:
        """Process video in streaming mode."""
        self.logger.info("Starting streaming processing")
        
        # Create video streams
        input_stream = VideoStream(input_path, mode='read')
        output_stream = VideoStream(output_path, mode='write')
        
        frame_results = []
        
        try:
            async for frame_result in self.process_video_stream(
                input_stream, output_stream, advertisement
            ):
                frame_results.append(frame_result)
                
                # Update progress
                if self.current_video_info:
                    progress = (frame_result.frame_id + 1) / self.current_video_info.total_frames
                    self.monitor.update_progress(progress)
            
            # Calculate final metrics
            successful_frames = sum(1 for r in frame_results if r.success)
            failed_frames = len(frame_results) - successful_frames
            
            total_resource_usage = self._aggregate_resource_usage(frame_results)
            quality_metrics = self._calculate_quality_metrics(frame_results)
            
            return ProcessingResult(
                output_path=output_path,
                processing_time=0.0,
                total_frames=len(frame_results),
                successful_frames=successful_frames,
                failed_frames=failed_frames,
                quality_metrics=quality_metrics,
                resource_usage=total_resource_usage,
                error_log=[r.error_message for r in frame_results if r.error_message]
            )
            
        finally:
            await input_stream.close()
            await output_stream.close()
    
    async def _process_realtime(self,
                              input_path: str,
                              output_path: str,
                              advertisement: torch.Tensor,
                              placement_config: PlacementConfig) -> ProcessingResult:
        """Process video in real-time mode with strict timing constraints."""
        self.logger.info("Starting real-time processing")
        
        # Real-time processing requires different approach
        # This would typically connect to live video streams
        # For now, simulate by processing with timing constraints
        
        target_fps = 30.0  # Target processing FPS
        frame_time_budget = 1.0 / target_fps
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_results = []
        dropped_frames = 0
        
        try:
            frame_id = 0
            
            while True:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with time budget
                try:
                    result = await asyncio.wait_for(
                        self._process_frame(frame, frame_id, advertisement),
                        timeout=frame_time_budget
                    )
                    frame_results.append(result)
                    
                except asyncio.TimeoutError:
                    # Frame dropped due to time constraint
                    dropped_frames += 1
                    self.logger.warning(f"Frame {frame_id} dropped due to timing constraint")
                
                frame_id += 1
                
                # Maintain real-time pace
                processing_time = time.time() - frame_start_time
                sleep_time = max(0, frame_time_budget - processing_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            successful_frames = sum(1 for r in frame_results if r.success)
            failed_frames = len(frame_results) - successful_frames
            
            total_resource_usage = self._aggregate_resource_usage(frame_results)
            quality_metrics = self._calculate_quality_metrics(frame_results)
            quality_metrics['dropped_frames'] = dropped_frames
            quality_metrics['real_time_performance'] = (total_frames - dropped_frames) / total_frames
            
            return ProcessingResult(
                output_path=output_path,
                processing_time=0.0,
                total_frames=total_frames,
                successful_frames=successful_frames,
                failed_frames=failed_frames,
                quality_metrics=quality_metrics,
                resource_usage=total_resource_usage,
                error_log=[]
            )
            
        finally:
            cap.release()
    
    async def _process_frame_batch(self,
                                 batch_frames: List[Tuple[int, np.ndarray]],
                                 advertisement: torch.Tensor,
                                 placement_config: PlacementConfig) -> List[FrameResult]:
        """Process a batch of frames efficiently."""
        results = []
        
        for frame_id, frame in batch_frames:
            try:
                result = await self._process_frame(frame, frame_id, advertisement)
                results.append(result)
            except Exception as e:
                error_result = FrameResult(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    success=False,
                    tv_placements=[],
                    quality_score=0.0,
                    processing_time=0.0,
                    resource_usage=ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0),
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def _process_frame(self,
                           frame: np.ndarray,
                           frame_id: int,
                           advertisement: torch.Tensor) -> FrameResult:
        """Process a single frame through the complete pipeline."""
        start_time = time.time()
        
        try:
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            if len(frame_tensor.shape) == 3:
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            
            # 1. Depth estimation
            depth_map = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.depth_estimator.estimate_depth,
                frame_tensor
            )
            
            # 2. Object detection
            detections = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.object_detector.detect_objects,
                frame_tensor
            )
            
            # 3. Plane detection
            planes = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.plane_detector.detect_planes,
                depth_map,
                frame_tensor
            )
            
            # 4. TV placement with temporal consistency
            camera_params = CameraParameters(fx=500, fy=500, cx=320, cy=240)  # Example params
            tv_placements = self.consistency_manager.track_tv_placement(
                detections, [], depth_map, frame_tensor
            )
            
            # 5. Video rendering
            rendered_frame = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.video_renderer.render_frame,
                frame_tensor,
                tv_placements,
                advertisement
            )
            
            # Calculate quality score
            quality_score = self._calculate_frame_quality(tv_placements, depth_map)
            
            # Measure resource usage
            resource_usage = self._measure_frame_resources()
            
            return FrameResult(
                frame_id=frame_id,
                timestamp=time.time(),
                success=True,
                tv_placements=tv_placements,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                resource_usage=resource_usage,
                intermediate_results={
                    'depth_map': depth_map,
                    'detections': detections,
                    'planes': planes,
                    'rendered_frame': rendered_frame
                }
            )
            
        except Exception as e:
            return FrameResult(
                frame_id=frame_id,
                timestamp=time.time(),
                success=False,
                tv_placements=[],
                quality_score=0.0,
                processing_time=time.time() - start_time,
                resource_usage=ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0),
                error_message=str(e)
            )
    
    async def _extract_video_info(self, video_path: str) -> VideoInfo:
        """Extract comprehensive video information."""
        cap = cv2.VideoCapture(video_path)
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file info
            file_path = Path(video_path)
            file_size = file_path.stat().st_size
            
            return VideoInfo(
                path=video_path,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                codec="unknown",  # Would extract from metadata
                bitrate=0,  # Would extract from metadata
                file_size=file_size,
                has_audio=False  # Would detect audio tracks
            )
            
        finally:
            cap.release()
    
    async def _analyze_video_characteristics(self, video_path: str) -> VideoCharacteristics:
        """Analyze video content characteristics for optimization."""
        # This would perform comprehensive video analysis
        # For now, return default characteristics
        
        return VideoCharacteristics(
            complexity_score=0.5,
            motion_intensity=0.5,
            scene_changes=10,
            dominant_colors=[(128, 128, 128)],
            lighting_stability=0.7,
            object_density=0.3,
            depth_variation=0.6,
            recommended_quality=QualityLevel.MEDIUM
        )
    
    def _calculate_frame_quality(self, tv_placements: List[Any], depth_map: torch.Tensor) -> float:
        """Calculate quality score for a single frame."""
        if not tv_placements:
            return 0.0
        
        # Simple quality calculation based on placement confidence
        confidences = [p.confidence for p in tv_placements if hasattr(p, 'confidence')]
        return np.mean(confidences) if confidences else 0.0
    
    def _measure_frame_resources(self) -> ResourceUsage:
        """Measure current resource usage."""
        process = psutil.Process()
        
        return ResourceUsage(
            cpu_usage=process.cpu_percent(),
            memory_usage=process.memory_info().rss / (1024 * 1024),  # MB
            gpu_usage=0.0,  # Would measure GPU usage
            gpu_memory_usage=0.0,  # Would measure GPU memory
            disk_io=0.0,  # Would measure disk I/O
            network_io=0.0,  # Would measure network I/O
            processing_time=0.0,  # Set by caller
            peak_memory=process.memory_info().peak_wss / (1024 * 1024) if hasattr(process.memory_info(), 'peak_wss') else 0.0,
            energy_consumption=0.0  # Would estimate energy usage
        )
    
    def _aggregate_resource_usage(self, frame_results: List[FrameResult]) -> ResourceUsage:
        """Aggregate resource usage across all frames."""
        if not frame_results:
            return ResourceUsage(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_cpu = sum(r.resource_usage.cpu_usage for r in frame_results)
        total_memory = sum(r.resource_usage.memory_usage for r in frame_results)
        total_gpu = sum(r.resource_usage.gpu_usage for r in frame_results)
        total_gpu_memory = sum(r.resource_usage.gpu_memory_usage for r in frame_results)
        total_processing_time = sum(r.processing_time for r in frame_results)
        
        num_frames = len(frame_results)
        
        return ResourceUsage(
            cpu_usage=total_cpu / num_frames,
            memory_usage=total_memory / num_frames,
            gpu_usage=total_gpu / num_frames,
            gpu_memory_usage=total_gpu_memory / num_frames,
            disk_io=0.0,
            network_io=0.0,
            processing_time=total_processing_time,
            peak_memory=max(r.resource_usage.peak_memory for r in frame_results),
            energy_consumption=0.0
        )
    
    def _calculate_quality_metrics(self, frame_results: List[FrameResult]) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        if not frame_results:
            return {}
        
        successful_results = [r for r in frame_results if r.success]
        
        if not successful_results:
            return {'overall_quality': 0.0}
        
        quality_scores = [r.quality_score for r in successful_results]
        processing_times = [r.processing_time for r in successful_results]
        
        return {
            'overall_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_std': np.std(quality_scores),
            'avg_processing_time': np.mean(processing_times),
            'temporal_consistency': self._calculate_temporal_consistency(successful_results)
        }
    
    def _calculate_temporal_consistency(self, frame_results: List[FrameResult]) -> float:
        """Calculate temporal consistency score."""
        if len(frame_results) < 2:
            return 1.0
        
        # Simple consistency based on quality score variation
        quality_scores = [r.quality_score for r in frame_results]
        consistency = 1.0 - np.std(quality_scores)
        return max(0.0, consistency)
    
    async def _create_checkpoint(self, frame_id: int, frame_results: List[FrameResult]):
        """Create processing checkpoint."""
        checkpoint = ProcessingCheckpoint(
            frame_id=frame_id,
            timestamp=time.time(),
            results=frame_results[-self.config.checkpoint_interval:],  # Last N results
            pipeline_state=self._get_pipeline_state()
        )
        
        await self.checkpoint_manager.save_checkpoint(checkpoint)
        self.logger.info(f"Checkpoint created at frame {frame_id}")
    
    def _get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state for checkpointing."""
        return {
            'config': self.config.__dict__,
            'processing_stats': self.processing_stats,
            'resource_allocations': len(self._resource_allocations)
        }
    
    async def _initialize_streaming(self, input_stream: VideoStream, output_stream: VideoStream):
        """Initialize streaming processing components."""
        self.buffer_manager.initialize_buffers(input_stream.info)
        await input_stream.start()
        await output_stream.start()
    
    async def _cleanup_streaming(self):
        """Cleanup streaming processing components."""
        self.buffer_manager.cleanup_buffers()
    
    async def _cleanup_resources(self):
        """Cleanup allocated resources."""
        for allocation in self._resource_allocations:
            await self.resource_manager.deallocate_resources(allocation)
        
        self._resource_allocations.clear()
        
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            **self.processing_stats,
            'pipeline_config': self.config.__dict__,
            'component_stats': {
                'depth_estimator': getattr(self.depth_estimator, 'stats', {}),
                'object_detector': getattr(self.object_detector, 'stats', {}),
                'plane_detector': getattr(self.plane_detector, 'stats', {}),
                'consistency_manager': self.consistency_manager.get_performance_stats()
            },
            'resource_usage': self.resource_manager.get_current_usage(),
            'health_status': self.monitor.get_health_status()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        self.logger.info("Shutting down VideoAdPlacementPipeline")
        
        self.is_processing = False
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Cleanup resources
        await self._cleanup_resources()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Pipeline shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False) 