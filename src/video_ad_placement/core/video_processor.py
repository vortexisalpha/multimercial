"""
Video Processor Module

Main processing pipeline that orchestrates all components for intelligent
video advertisement placement including depth estimation, object detection,
scene understanding, camera estimation, rendering, and temporal consistency.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch

from .depth_estimation import DepthEstimator, DepthEstimationConfig
from .object_detection import ObjectDetector, ObjectDetectionConfig
from .scene_understanding import SceneAnalyzer, SceneUnderstandingConfig
from .camera_estimation import CameraEstimator, CameraEstimationConfig
from .rendering_engine import RenderingEngine, RenderingConfig, AdAsset
from .temporal_consistency import TemporalStabilizer, TemporalConsistencyConfig

from ..utils.logging_utils import get_logger
from ..utils.video_utils import VideoReader, VideoWriter, extract_frames, save_video
from ..utils.performance_utils import PerformanceMonitor, ProfilerContext

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Video processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    OFFLINE = "offline"


class QualityLevel(Enum):
    """Processing quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ProcessingResult:
    """Result of video processing operation."""
    success: bool
    output_path: Optional[str] = None
    processing_time: float = 0.0
    frames_processed: int = 0
    average_fps: float = 0.0
    errors: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: np.ndarray
    depth_map: Optional[np.ndarray] = None
    detections: List[Any] = None
    scene_layout: Optional[Any] = None
    camera_params: Optional[Any] = None
    temporal_info: Optional[Any] = None
    processing_time: float = 0.0


class VideoProcessorConfig:
    """Configuration for the video processing pipeline."""
    
    def __init__(
        self,
        processing_mode: ProcessingMode = ProcessingMode.BATCH,
        quality_level: QualityLevel = QualityLevel.HIGH,
        enable_gpu_acceleration: bool = True,
        max_video_resolution: Tuple[int, int] = (1920, 1080),
        target_fps: float = 30.0,
        enable_progress_tracking: bool = True,
        enable_performance_monitoring: bool = True,
        save_intermediate_results: bool = False,
        output_format: str = "mp4",
        compression_quality: float = 0.8,
        **kwargs
    ):
        self.processing_mode = processing_mode
        self.quality_level = quality_level
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.max_video_resolution = max_video_resolution
        self.target_fps = target_fps
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_performance_monitoring = enable_performance_monitoring
        self.save_intermediate_results = save_intermediate_results
        self.output_format = output_format
        self.compression_quality = compression_quality
        self.kwargs = kwargs


class VideoProcessor:
    """
    Main video processing pipeline for AI-powered advertisement placement.
    
    This class orchestrates all components of the system to provide a unified
    interface for processing videos with intelligent advertisement placement.
    
    Attributes:
        config: Main processing configuration
        depth_estimator: Depth estimation component
        object_detector: Object detection component
        scene_analyzer: Scene understanding component
        camera_estimator: Camera parameter estimation component
        rendering_engine: 3D rendering and compositing component
        temporal_stabilizer: Temporal consistency component
        performance_monitor: Performance monitoring system
        is_initialized: Initialization status
    """

    def __init__(
        self,
        config: Optional[VideoProcessorConfig] = None,
        depth_config: Optional[DepthEstimationConfig] = None,
        detection_config: Optional[ObjectDetectionConfig] = None,
        scene_config: Optional[SceneUnderstandingConfig] = None,
        camera_config: Optional[CameraEstimationConfig] = None,
        rendering_config: Optional[RenderingConfig] = None,
        temporal_config: Optional[TemporalConsistencyConfig] = None
    ):
        """
        Initialize the video processor with component configurations.
        
        Args:
            config: Main video processing configuration
            depth_config: Depth estimation configuration
            detection_config: Object detection configuration
            scene_config: Scene understanding configuration
            camera_config: Camera estimation configuration
            rendering_config: Rendering engine configuration
            temporal_config: Temporal consistency configuration
        """
        self.config = config or VideoProcessorConfig()
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor() if self.config.enable_performance_monitoring else None
        
        # Initialize components
        logger.info("Initializing VideoProcessor components")
        self._initialize_components(
            depth_config, detection_config, scene_config,
            camera_config, rendering_config, temporal_config
        )
        
        self.is_initialized = True
        logger.info("VideoProcessor initialization completed")

    def _initialize_components(
        self,
        depth_config: Optional[DepthEstimationConfig],
        detection_config: Optional[ObjectDetectionConfig],
        scene_config: Optional[SceneUnderstandingConfig],
        camera_config: Optional[CameraEstimationConfig],
        rendering_config: Optional[RenderingConfig],
        temporal_config: Optional[TemporalConsistencyConfig]
    ) -> None:
        """Initialize all processing components."""
        try:
            # Configure components based on quality level
            depth_config = depth_config or self._get_default_depth_config()
            detection_config = detection_config or self._get_default_detection_config()
            scene_config = scene_config or self._get_default_scene_config()
            camera_config = camera_config or self._get_default_camera_config()
            rendering_config = rendering_config or self._get_default_rendering_config()
            temporal_config = temporal_config or self._get_default_temporal_config()
            
            # Initialize components
            self.depth_estimator = DepthEstimator(depth_config)
            self.object_detector = ObjectDetector(detection_config)
            self.scene_analyzer = SceneAnalyzer(scene_config)
            self.camera_estimator = CameraEstimator(camera_config)
            self.rendering_engine = RenderingEngine(rendering_config)
            self.temporal_stabilizer = TemporalStabilizer(temporal_config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise RuntimeError(f"Initialization error: {str(e)}")

    def _get_default_depth_config(self) -> DepthEstimationConfig:
        """Get default depth estimation configuration based on quality level."""
        from .depth_estimation import DepthModelType
        
        if self.config.quality_level == QualityLevel.HIGH:
            return DepthEstimationConfig(
                model_type=DepthModelType.MARIGOLD,
                input_size=(518, 518),
                batch_size=1
            )
        elif self.config.quality_level == QualityLevel.MEDIUM:
            return DepthEstimationConfig(
                model_type=DepthModelType.DEPTH_PRO,
                input_size=(384, 384),
                batch_size=2
            )
        else:  # LOW quality
            return DepthEstimationConfig(
                model_type=DepthModelType.MARIGOLD,
                input_size=(256, 256),
                batch_size=4
            )

    def _get_default_detection_config(self) -> ObjectDetectionConfig:
        """Get default object detection configuration."""
        from .object_detection import DetectionModelType
        
        if self.config.quality_level == QualityLevel.HIGH:
            return ObjectDetectionConfig(
                model_type=DetectionModelType.YOLOV9_L,
                confidence_threshold=0.5,
                input_size=(640, 640)
            )
        else:
            return ObjectDetectionConfig(
                model_type=DetectionModelType.YOLOV9_S,
                confidence_threshold=0.6,
                input_size=(416, 416)
            )

    def _get_default_scene_config(self) -> SceneUnderstandingConfig:
        """Get default scene understanding configuration."""
        return SceneUnderstandingConfig(
            min_plane_points=1000 if self.config.quality_level == QualityLevel.HIGH else 500,
            enable_texture_analysis=self.config.quality_level != QualityLevel.LOW,
            enable_lighting_analysis=True
        )

    def _get_default_camera_config(self) -> CameraEstimationConfig:
        """Get default camera estimation configuration."""
        from .camera_estimation import EstimationMethod
        
        return CameraEstimationConfig(
            estimation_method=EstimationMethod.VANISHING_POINTS,
            use_temporal_smoothing=True,
            temporal_window_size=5
        )

    def _get_default_rendering_config(self) -> RenderingConfig:
        """Get default rendering configuration."""
        from .rendering_engine import RenderingMode
        
        return RenderingConfig(
            rendering_mode=RenderingMode.PBR if self.config.quality_level == QualityLevel.HIGH else RenderingMode.BASIC,
            output_resolution=self.config.max_video_resolution,
            enable_shadows=self.config.quality_level != QualityLevel.LOW,
            enable_reflections=self.config.quality_level == QualityLevel.HIGH
        )

    def _get_default_temporal_config(self) -> TemporalConsistencyConfig:
        """Get default temporal consistency configuration."""
        return TemporalConsistencyConfig(
            enable_scene_stabilization=True,
            enable_camera_stabilization=True,
            stabilization_window_size=10
        )

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        ad_assets: List[Union[str, Path, np.ndarray]],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Process a video file with advertisement placement.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            ad_assets: List of advertisement assets (images or paths)
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            callback: Optional progress callback function
            
        Returns:
            ProcessingResult with operation status and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("VideoProcessor not initialized")

        start_processing_time = time.time()
        frames_processed = 0
        errors = []
        
        try:
            logger.info(f"Starting video processing: {input_path} -> {output_path}")
            
            # Load advertisement assets
            ad_assets_loaded = self._load_ad_assets(ad_assets)
            
            # Initialize video reader and writer
            video_reader = VideoReader(str(input_path))
            video_info = video_reader.get_info()
            
            # Calculate frame range
            start_frame = int(start_time * video_info['fps'])
            end_frame = int(end_time * video_info['fps']) if end_time else video_info['total_frames']
            total_frames = end_frame - start_frame
            
            # Initialize video writer
            output_size = self._get_output_resolution(video_info)
            video_writer = VideoWriter(
                str(output_path),
                fps=video_info['fps'],
                resolution=output_size,
                codec=self._get_output_codec()
            )
            
            # Reset temporal stabilizer for new video
            self.temporal_stabilizer.reset()
            
            # Process frames
            with ProfilerContext("video_processing", enabled=self.config.enable_performance_monitoring):
                for frame_id, frame_result in enumerate(self._process_video_frames(
                    video_reader, ad_assets_loaded, start_frame, end_frame
                )):
                    if frame_result.success:
                        video_writer.write_frame(frame_result.processed_frame)
                        frames_processed += 1
                        
                        # Update progress
                        if callback and self.config.enable_progress_tracking:
                            progress = frames_processed / total_frames
                            callback(progress, frame_result)
                    else:
                        errors.extend(frame_result.errors or [])
                        logger.warning(f"Frame {frame_id} processing failed")
            
            # Finalize video
            video_writer.close()
            video_reader.close()
            
            # Calculate final statistics
            processing_time = time.time() - start_processing_time
            average_fps = frames_processed / processing_time if processing_time > 0 else 0
            
            # Generate metadata
            metadata = self._generate_processing_metadata(
                video_info, frames_processed, processing_time, ad_assets_loaded
            )
            
            result = ProcessingResult(
                success=frames_processed > 0,
                output_path=str(output_path),
                processing_time=processing_time,
                frames_processed=frames_processed,
                average_fps=average_fps,
                errors=errors,
                metadata=metadata
            )
            
            logger.info(f"Video processing completed: {frames_processed} frames in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_processing_time
            )

    def _load_ad_assets(self, ad_assets: List[Union[str, Path, np.ndarray]]) -> List[AdAsset]:
        """Load and prepare advertisement assets."""
        loaded_assets = []
        
        for asset in ad_assets:
            try:
                if isinstance(asset, (str, Path)):
                    # Load image from file
                    image = cv2.imread(str(asset))
                    if image is None:
                        logger.warning(f"Failed to load ad asset: {asset}")
                        continue
                elif isinstance(asset, np.ndarray):
                    image = asset
                else:
                    logger.warning(f"Unsupported ad asset type: {type(asset)}")
                    continue
                
                # Create AdAsset object
                ad_asset = AdAsset(
                    image=image,
                    opacity=0.8,  # Default opacity
                    scale=(1.0, 1.0)  # Default scale
                )
                
                loaded_assets.append(ad_asset)
                
            except Exception as e:
                logger.error(f"Failed to load ad asset {asset}: {str(e)}")
        
        logger.info(f"Loaded {len(loaded_assets)} advertisement assets")
        return loaded_assets

    def _process_video_frames(
        self,
        video_reader: VideoReader,
        ad_assets: List[AdAsset],
        start_frame: int,
        end_frame: int
    ) -> Generator[FrameResult, None, None]:
        """Process video frames generator."""
        
        for frame_id in range(start_frame, end_frame):
            frame_start_time = time.time()
            
            try:
                # Read frame
                frame = video_reader.read_frame(frame_id)
                if frame is None:
                    yield FrameResult(
                        frame_id=frame_id,
                        timestamp=frame_id / video_reader.fps,
                        original_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                        processed_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                        success=False,
                        errors=[f"Failed to read frame {frame_id}"]
                    )
                    continue
                
                # Process single frame
                frame_result = self.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=frame_id / video_reader.fps,
                    ad_assets=ad_assets
                )
                
                frame_result.processing_time = time.time() - frame_start_time
                yield frame_result
                
            except Exception as e:
                logger.error(f"Frame {frame_id} processing error: {str(e)}")
                yield FrameResult(
                    frame_id=frame_id,
                    timestamp=frame_id / video_reader.fps,
                    original_frame=frame if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8),
                    processed_frame=frame if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8),
                    success=False,
                    errors=[str(e)],
                    processing_time=time.time() - frame_start_time
                )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        ad_assets: List[AdAsset]
    ) -> FrameResult:
        """
        Process a single video frame with advertisement placement.
        
        Args:
            frame: Input video frame
            frame_id: Frame identifier
            timestamp: Frame timestamp
            ad_assets: Advertisement assets to place
            
        Returns:
            FrameResult with processed frame and metadata
        """
        try:
            # Step 1: Depth estimation
            with ProfilerContext("depth_estimation", enabled=self.config.enable_performance_monitoring):
                depth_map = self.depth_estimator.estimate(frame)
            
            # Step 2: Object detection
            with ProfilerContext("object_detection", enabled=self.config.enable_performance_monitoring):
                detections = self.object_detector.detect(frame)
            
            # Step 3: Scene understanding
            with ProfilerContext("scene_understanding", enabled=self.config.enable_performance_monitoring):
                scene_layout = self.scene_analyzer.analyze_scene(frame, depth_map)
            
            # Step 4: Camera estimation
            with ProfilerContext("camera_estimation", enabled=self.config.enable_performance_monitoring):
                camera_params = self.camera_estimator.estimate(
                    frame, depth_map, detections, scene_layout.planes
                )
            
            # Step 5: Temporal consistency
            with ProfilerContext("temporal_consistency", enabled=self.config.enable_performance_monitoring):
                temporal_info = self.temporal_stabilizer.process_frame(
                    frame_id, timestamp, detections, scene_layout, camera_params
                )
            
            # Step 6: Rendering and compositing
            with ProfilerContext("rendering", enabled=self.config.enable_performance_monitoring):
                render_result = self.rendering_engine.composite_ads(
                    background_image=frame,
                    depth_map=depth_map,
                    objects=temporal_info.detections,
                    planes=temporal_info.scene_layout.placement_candidates if temporal_info.scene_layout else [],
                    camera_params=temporal_info.camera_params,
                    ad_assets=ad_assets
                )
            
            return FrameResult(
                frame_id=frame_id,
                timestamp=timestamp,
                original_frame=frame,
                processed_frame=render_result.rendered_image,
                depth_map=depth_map,
                detections=temporal_info.detections,
                scene_layout=temporal_info.scene_layout,
                camera_params=temporal_info.camera_params,
                temporal_info=temporal_info,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return FrameResult(
                frame_id=frame_id,
                timestamp=timestamp,
                original_frame=frame,
                processed_frame=frame,  # Return original frame on failure
                success=False,
                errors=[str(e)]
            )

    def _get_output_resolution(self, video_info: Dict[str, Any]) -> Tuple[int, int]:
        """Determine output video resolution."""
        original_width = video_info.get('width', 1920)
        original_height = video_info.get('height', 1080)
        max_width, max_height = self.config.max_video_resolution
        
        # Maintain aspect ratio while respecting max resolution
        scale = min(max_width / original_width, max_height / original_height, 1.0)
        
        output_width = int(original_width * scale)
        output_height = int(original_height * scale)
        
        # Ensure even dimensions for video encoding
        output_width = output_width - (output_width % 2)
        output_height = output_height - (output_height % 2)
        
        return (output_width, output_height)

    def _get_output_codec(self) -> str:
        """Get appropriate output codec based on format."""
        codec_map = {
            'mp4': 'h264',
            'avi': 'xvid',
            'mov': 'h264',
            'mkv': 'h264'
        }
        return codec_map.get(self.config.output_format, 'h264')

    def _generate_processing_metadata(
        self,
        video_info: Dict[str, Any],
        frames_processed: int,
        processing_time: float,
        ad_assets: List[AdAsset]
    ) -> Dict[str, Any]:
        """Generate comprehensive processing metadata."""
        metadata = {
            "input_video": {
                "width": video_info.get('width'),
                "height": video_info.get('height'),
                "fps": video_info.get('fps'),
                "total_frames": video_info.get('total_frames'),
                "duration": video_info.get('duration')
            },
            "processing": {
                "frames_processed": frames_processed,
                "processing_time": processing_time,
                "average_fps": frames_processed / processing_time if processing_time > 0 else 0,
                "quality_level": self.config.quality_level.value,
                "processing_mode": self.config.processing_mode.value
            },
            "components": {
                "depth_estimator": self.depth_estimator.get_model_info(),
                "object_detector": self.object_detector.get_performance_stats(),
                "rendering_engine": {
                    "mode": self.rendering_engine.config.rendering_mode.value,
                    "resolution": self.rendering_engine.config.output_resolution
                }
            },
            "advertisement": {
                "num_assets": len(ad_assets),
                "asset_info": [
                    {
                        "size": asset.image.shape[:2],
                        "opacity": asset.opacity,
                        "scale": asset.scale
                    } for asset in ad_assets
                ]
            }
        }
        
        # Add performance metrics if available
        if self.performance_monitor:
            metadata["performance"] = self.performance_monitor.get_summary()
        
        return metadata

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "component_stats": {
                "depth_estimator": self.depth_estimator.get_model_info(),
                "object_detector": self.object_detector.get_performance_stats(),
                "temporal_stabilizer": {
                    "active_tracks": len(self.temporal_stabilizer.get_active_tracks()),
                    "frame_history_size": len(self.temporal_stabilizer.frame_history)
                }
            }
        }
        
        if self.performance_monitor:
            stats["performance_monitor"] = self.performance_monitor.get_summary()
        
        return stats

    def cleanup(self) -> None:
        """Clean up all resources and components."""
        logger.info("Cleaning up VideoProcessor resources")
        
        if hasattr(self, 'depth_estimator'):
            self.depth_estimator.cleanup()
        
        if hasattr(self, 'object_detector'):
            self.object_detector.cleanup()
        
        if hasattr(self, 'scene_analyzer'):
            self.scene_analyzer.cleanup()
        
        if hasattr(self, 'camera_estimator'):
            self.camera_estimator.cleanup()
        
        if hasattr(self, 'rendering_engine'):
            self.rendering_engine.cleanup()
        
        if hasattr(self, 'temporal_stabilizer'):
            self.temporal_stabilizer.cleanup()
        
        self.is_initialized = False
        logger.info("VideoProcessor cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

# Add FrameResult success property for compatibility
setattr(FrameResult, 'success', property(lambda self: getattr(self, '_success', True)))
def _set_success(self, value):
    self._success = value
setattr(FrameResult, 'success', property(lambda self: getattr(self, '_success', True), _set_success)) 