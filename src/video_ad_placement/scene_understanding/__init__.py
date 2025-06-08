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
    Plane, PlaneDetectionConfig, PlaneQuality, PlaneType,
    SurfaceNormalConfig, TemporalTrackingConfig, WallQualityMetrics
)
from .plane_detector import PlaneDetector
from .surface_normal_estimator import SurfaceNormalEstimator, HybridNormalEstimator
from .temporal_tracker import TemporalPlaneTracker
from .geometry_utils import GeometryUtils, CameraProjection, PlaneGeometry

__all__ = [
    # Core plane models and configuration
    'Plane',
    'PlaneDetectionConfig',
    'PlaneQuality',
    'PlaneType',
    'SurfaceNormalConfig',
    'TemporalTrackingConfig',
    'WallQualityMetrics',
    
    # Plane detection
    'PlaneDetector',
    
    # Surface normal estimation
    'SurfaceNormalEstimator',
    'HybridNormalEstimator',
    
    # Temporal tracking
    'TemporalPlaneTracker',
    
    # Geometry utilities
    'GeometryUtils',
    'CameraProjection',
    'PlaneGeometry',
]

__version__ = "1.0.0" 