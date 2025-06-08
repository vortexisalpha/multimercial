"""
Core modules for the video advertisement placement system.

This package contains the main processing components:
- Depth estimation
- Object detection and tracking
- Scene understanding and analysis
- Camera parameter estimation
- 3D rendering and compositing
- Temporal consistency and stabilization
- Main video processing pipeline
"""

from .depth_estimation import DepthEstimator
from .object_detection import ObjectDetector
from .scene_understanding import SceneAnalyzer
from .camera_estimation import CameraEstimator
from .rendering_engine import RenderingEngine
from .temporal_consistency import TemporalStabilizer
from .video_processor import VideoProcessor

__all__ = [
    "DepthEstimator",
    "ObjectDetector", 
    "SceneAnalyzer",
    "CameraEstimator",
    "RenderingEngine",
    "TemporalStabilizer",
    "VideoProcessor",
] 