"""
3D Scene Understanding Module for Video Advertisement Placement

This module provides sophisticated 3D scene analysis capabilities for detecting
and analyzing wall surfaces suitable for TV advertisement placement, including
RANSAC-based plane detection, surface normal estimation, and temporal tracking.

Features:
- RANSAC-based plane detection with geometric constraints
- Surface normal estimation using depth and RGB information
- Wall surface validation and quality assessment
- Temporal consistency for plane tracking across frames
- Integration with depth estimation and camera parameters
- Real-time performance optimization
- Comprehensive validation and testing framework
- Visualization tools for debugging and analysis
"""

from .plane_models import (
    Plane, PlaneDetectionConfig, PlaneQuality,
    PlaneType, SurfaceNormalConfig, TemporalTrackingConfig
)

from .plane_detector import PlaneDetector

from .geometry_utils import (
    GeometryUtils, CameraParameters, CameraProjection, PlaneGeometry
)

from .temporal_tracker import (
    TemporalPlaneTracker, PlaneTrack, PlaneAssociator,
    TemporalConsistencyValidator
)

# New temporal consistency system
from .temporal_consistency import (
    TemporalConsistencyManager, TVPlacement, TVGeometry, 
    LightingEnvironment, QualityMetrics, TemporalConfig,
    TVPlacementState, QualityLevel,
    ExtendedKalmanFilter, OcclusionHandler, 
    LightingConsistencyManager, QualityAssessment,
    ArtifactDetector, PredictionInterpolation
)

__version__ = "1.0.0"

__all__ = [
    # Core data models
    "Plane", "PlaneDetectionConfig", "PlaneQuality", "PlaneType",
    "SurfaceNormalConfig", "TemporalTrackingConfig",
    
    # Plane detection
    "PlaneDetector",
    
    # Geometry utilities
    "GeometryUtils", "CameraParameters", "CameraProjection", "PlaneGeometry",
    
    # Temporal tracking (original)
    "TemporalPlaneTracker", "PlaneTrack", "PlaneAssociator",
    "TemporalConsistencyValidator",
    
    # Temporal consistency system (new)
    "TemporalConsistencyManager", "TVPlacement", "TVGeometry",
    "LightingEnvironment", "QualityMetrics", "TemporalConfig",
    "TVPlacementState", "QualityLevel",
    "ExtendedKalmanFilter", "OcclusionHandler",
    "LightingConsistencyManager", "QualityAssessment",
    "ArtifactDetector", "PredictionInterpolation",
] 