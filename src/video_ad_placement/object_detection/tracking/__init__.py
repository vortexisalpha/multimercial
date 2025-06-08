"""
Multi-Object Tracking Module for Video Advertisement Placement

This module provides a comprehensive tracking system based on the ByteTrack algorithm,
specifically optimized for video advertisement scenarios with advanced occlusion
handling, re-identification, and real-time performance.

Features:
- ByteTrack algorithm with custom modifications for video content
- Advanced occlusion handling and re-identification
- Temporal consistency across camera movements
- Integration with YOLOv9 detection results
- Kalman filter-based motion prediction
- Track lifecycle management and optimization
- Real-time performance with minimal latency
- Comprehensive tracking metrics and evaluation
"""

from .byte_tracker import ByteTracker, TrackingConfig
from .track_models import Track, TrackState, TrackQuality
from .motion_models import KalmanTracker, MotionModel, CameraMotionCompensator
from .association import TrackAssociation, AssociationMethod
from .reid_features import ReIDFeatureExtractor, FeatureConfig
from .track_manager import TrackManager, TrackManagerConfig
from .evaluation import TrackingEvaluator, TrackingMetrics, MOTMetrics
from .visualization import TrackingVisualizer, VisualizationConfig
from .utils import TrackingUtils, TrackInterpolator, TrackSmoother

__all__ = [
    # Core tracking classes
    'ByteTracker',
    'TrackingConfig',
    
    # Track data models
    'Track',
    'TrackState',
    'TrackQuality',
    
    # Motion models and prediction
    'KalmanTracker',
    'MotionModel',
    'CameraMotionCompensator',
    
    # Association algorithms
    'TrackAssociation',
    'AssociationMethod',
    
    # Re-identification features
    'ReIDFeatureExtractor',
    'FeatureConfig',
    
    # Track management
    'TrackManager',
    'TrackManagerConfig',
    
    # Evaluation and metrics
    'TrackingEvaluator',
    'TrackingMetrics',
    'MOTMetrics',
    
    # Visualization
    'TrackingVisualizer',
    'VisualizationConfig',
    
    # Utilities
    'TrackingUtils',
    'TrackInterpolator',
    'TrackSmoother'
]

__version__ = "1.0.0" 