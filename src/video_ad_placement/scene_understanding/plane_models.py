"""
Data Models and Configuration for 3D Plane Detection

This module defines the core data structures and configuration classes for
plane detection, surface normal estimation, and wall surface analysis.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import time


class PlaneType(Enum):
    """Plane type classification."""
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    FURNITURE = "furniture"
    UNKNOWN = "unknown"


class PlaneQuality(Enum):
    """Plane quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class WallQualityMetrics:
    """Quality metrics for wall surface assessment."""
    
    # Geometric properties
    area: float = 0.0  # Surface area in square meters
    planarity: float = 0.0  # How flat the surface is (0-1)
    aspect_ratio: float = 0.0  # Width/height ratio
    distance_to_camera: float = 0.0  # Distance in meters
    
    # Visibility metrics
    visibility_score: float = 0.0  # Overall visibility (0-1)
    occlusion_ratio: float = 0.0  # Fraction occluded (0-1)
    viewing_angle: float = 0.0  # Angle from camera normal (degrees)
    
    # Stability metrics
    temporal_stability: float = 0.0  # Consistency across frames (0-1)
    motion_stability: float = 0.0  # Resistance to camera motion (0-1)
    detection_consistency: float = 0.0  # Detection frequency (0-1)
    
    # Advertisement suitability
    tv_placement_score: float = 0.0  # Overall TV placement score (0-1)
    size_suitability: float = 0.0  # Size appropriateness (0-1)
    orientation_suitability: float = 0.0  # Orientation appropriateness (0-1)
    lighting_quality: float = 0.0  # Lighting conditions (0-1)
    
    def compute_overall_quality(self) -> PlaneQuality:
        """Compute overall quality assessment."""
        # Weighted combination of metrics
        overall_score = (
            self.tv_placement_score * 0.3 +
            self.visibility_score * 0.25 +
            self.temporal_stability * 0.2 +
            self.size_suitability * 0.15 +
            self.planarity * 0.1
        )
        
        if overall_score >= 0.8:
            return PlaneQuality.EXCELLENT
        elif overall_score >= 0.6:
            return PlaneQuality.GOOD
        elif overall_score >= 0.4:
            return PlaneQuality.FAIR
        elif overall_score >= 0.2:
            return PlaneQuality.POOR
        else:
            return PlaneQuality.UNUSABLE
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'area': self.area,
            'planarity': self.planarity,
            'aspect_ratio': self.aspect_ratio,
            'distance_to_camera': self.distance_to_camera,
            'visibility_score': self.visibility_score,
            'occlusion_ratio': self.occlusion_ratio,
            'viewing_angle': self.viewing_angle,
            'temporal_stability': self.temporal_stability,
            'motion_stability': self.motion_stability,
            'detection_consistency': self.detection_consistency,
            'tv_placement_score': self.tv_placement_score,
            'size_suitability': self.size_suitability,
            'orientation_suitability': self.orientation_suitability,
            'lighting_quality': self.lighting_quality
        }


@dataclass
class Plane:
    """
    3D plane representation with comprehensive metadata for wall surface analysis.
    """
    
    # Core geometric properties
    normal: torch.Tensor  # 3D normal vector (unit vector)
    point: torch.Tensor   # Point on plane (3D coordinates)
    equation: torch.Tensor  # Plane equation coefficients [a, b, c, d]
    
    # Pixel-level information
    inlier_mask: torch.Tensor  # Binary mask of plane pixels
    depth_values: torch.Tensor  # Depth values for plane pixels
    rgb_values: Optional[torch.Tensor] = None  # RGB values for plane pixels
    
    # Confidence and quality
    confidence: float = 0.0  # Detection confidence (0-1)
    area: float = 0.0  # Surface area in square meters
    planarity_score: float = 0.0  # How planar the surface is (0-1)
    
    # Plane classification
    plane_type: PlaneType = PlaneType.UNKNOWN
    quality: PlaneQuality = PlaneQuality.POOR
    quality_metrics: WallQualityMetrics = field(default_factory=WallQualityMetrics)
    
    # Temporal tracking
    plane_id: int = -1  # Unique identifier for temporal tracking
    first_detected_frame: int = -1
    last_detected_frame: int = -1
    detection_count: int = 0
    temporal_stability: float = 0.0
    stability_score: float = 0.0  # Overall stability score for tracking
    track_id: int = -1  # Track ID from temporal tracker
    track_length: int = 0  # Length of temporal track
    
    # Bounding information
    bounding_box_2d: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    bounding_box_3d: Optional[torch.Tensor] = None  # 3D bounding box corners
    
    # Camera-relative properties
    distance_to_camera: float = 0.0
    viewing_angle: float = 0.0  # Angle between plane normal and camera direction
    
    # Advertisement-specific metadata
    tv_placement_score: float = 0.0
    occlusion_history: List[float] = field(default_factory=list)
    visibility_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure normal is unit vector
        if torch.norm(self.normal) > 0:
            self.normal = torch.nn.functional.normalize(self.normal, dim=0)
        
        # Compute plane equation from normal and point
        if self.equation is None or torch.allclose(self.equation, torch.zeros_like(self.equation)):
            self.equation = self._compute_plane_equation()
        
        # Initialize quality metrics if not provided
        if self.quality_metrics.area == 0.0:
            self.quality_metrics.area = self.area
    
    def _compute_plane_equation(self) -> torch.Tensor:
        """Compute plane equation coefficients [a, b, c, d] from normal and point."""
        # Plane equation: ax + by + cz + d = 0
        # where (a, b, c) is the normal vector and d = -(ax0 + by0 + cz0)
        a, b, c = self.normal
        d = -torch.dot(self.normal, self.point)
        return torch.tensor([a, b, c, d], dtype=self.normal.dtype, device=self.normal.device)
    
    def distance_to_point(self, point: torch.Tensor) -> torch.Tensor:
        """Compute distance from point to plane."""
        # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        a, b, c, d = self.equation
        numerator = torch.abs(a * point[0] + b * point[1] + c * point[2] + d)
        denominator = torch.sqrt(a**2 + b**2 + c**2)
        return numerator / denominator
    
    def project_point_to_plane(self, point: torch.Tensor) -> torch.Tensor:
        """Project 3D point onto plane."""
        # Projected point = point - distance * normal
        distance = self.distance_to_point(point)
        projected = point - distance * self.normal
        return projected
    
    def is_point_on_plane(self, point: torch.Tensor, tolerance: float = 0.01) -> bool:
        """Check if point lies on plane within tolerance."""
        distance = self.distance_to_point(point)
        return distance < tolerance
    
    def compute_area_from_mask(self, depth_map: torch.Tensor, camera_intrinsics: torch.Tensor) -> float:
        """Compute surface area from depth map and camera intrinsics."""
        if self.inlier_mask is None:
            return 0.0
        
        # Get pixel coordinates of plane points
        height, width = self.inlier_mask.shape
        y_coords, x_coords = torch.where(self.inlier_mask)
        
        if len(x_coords) == 0:
            return 0.0
        
        # Convert to 3D points
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        depths = depth_map[y_coords, x_coords]
        x_3d = (x_coords - cx) * depths / fx
        y_3d = (y_coords - cy) * depths / fy
        z_3d = depths
        
        points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)
        
        # Estimate area using convex hull or triangulation
        # Simplified approach: assume rectangular area
        x_range = x_3d.max() - x_3d.min()
        y_range = y_3d.max() - y_3d.min()
        estimated_area = x_range * y_range
        
        return estimated_area.item()
    
    def update_temporal_info(self, frame_id: int):
        """Update temporal tracking information."""
        if self.first_detected_frame == -1:
            self.first_detected_frame = frame_id
        
        self.last_detected_frame = frame_id
        self.detection_count += 1
        
        # Update temporal stability
        if self.detection_count > 1:
            frame_span = self.last_detected_frame - self.first_detected_frame + 1
            self.temporal_stability = self.detection_count / frame_span
    
    def is_wall_suitable_for_tv(self, min_area: float = 2.0, max_distance: float = 5.0) -> bool:
        """Check if wall is suitable for TV placement."""
        return (
            self.plane_type == PlaneType.WALL and
            self.area >= min_area and
            self.distance_to_camera <= max_distance and
            self.quality in [PlaneQuality.EXCELLENT, PlaneQuality.GOOD] and
            self.tv_placement_score >= 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plane to dictionary representation."""
        return {
            'plane_id': self.plane_id,
            'normal': self.normal.tolist() if isinstance(self.normal, torch.Tensor) else self.normal,
            'point': self.point.tolist() if isinstance(self.point, torch.Tensor) else self.point,
            'equation': self.equation.tolist() if isinstance(self.equation, torch.Tensor) else self.equation,
            'confidence': self.confidence,
            'area': self.area,
            'planarity_score': self.planarity_score,
            'plane_type': self.plane_type.value,
            'quality': self.quality.value,
            'quality_metrics': self.quality_metrics.to_dict(),
            'distance_to_camera': self.distance_to_camera,
            'viewing_angle': self.viewing_angle,
            'tv_placement_score': self.tv_placement_score,
            'temporal_stability': self.temporal_stability,
            'detection_count': self.detection_count
        }


@dataclass 
class PlaneDetectionConfig:
    """Configuration for plane detection algorithms."""
    
    # RANSAC parameters
    ransac_iterations: int = 1000
    ransac_threshold: float = 0.01  # Distance threshold in meters
    min_inliers: int = 100  # Minimum number of inlier points
    max_inliers_ratio: float = 0.8  # Maximum inliers for plane validity
    
    # Multi-scale detection
    enable_multiscale: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    merge_threshold: float = 0.1  # Threshold for merging similar planes
    
    # Geometric constraints
    min_plane_area: float = 0.5  # Minimum area in square meters
    max_plane_area: float = 50.0  # Maximum area in square meters
    min_aspect_ratio: float = 0.1  # Minimum width/height ratio
    max_aspect_ratio: float = 10.0  # Maximum width/height ratio
    
    # Normal vector constraints
    wall_normal_tolerance: float = 0.3  # Cosine similarity threshold for walls
    floor_normal_tolerance: float = 0.2  # Cosine similarity threshold for floors
    ceiling_normal_tolerance: float = 0.2  # Cosine similarity threshold for ceilings
    
    # Quality thresholds
    min_confidence: float = 0.3
    min_planarity: float = 0.7
    min_visibility: float = 0.5
    
    # Performance optimization
    enable_multithreading: bool = True
    max_threads: int = 4
    batch_size: int = 1000  # Points per RANSAC batch
    early_termination: bool = True
    early_termination_threshold: float = 0.95
    
    # Memory management
    max_planes_per_frame: int = 20
    max_points_per_plane: int = 10000
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.ransac_iterations <= 0:
            raise ValueError("RANSAC iterations must be positive")
        
        if self.ransac_threshold <= 0:
            raise ValueError("RANSAC threshold must be positive")
        
        if self.min_inliers <= 0:
            raise ValueError("Minimum inliers must be positive")
        
        if not 0 < self.max_inliers_ratio <= 1:
            raise ValueError("Max inliers ratio must be in (0, 1]")
        
        if self.min_plane_area >= self.max_plane_area:
            raise ValueError("Min plane area must be less than max plane area")
        
        return True


@dataclass
class SurfaceNormalConfig:
    """Configuration for surface normal estimation."""
    
    # Estimation methods
    use_geometric_normals: bool = True
    use_learned_normals: bool = True
    geometric_weight: float = 0.7
    learned_weight: float = 0.3
    
    # Geometric normal estimation
    neighborhood_size: int = 5  # Kernel size for normal estimation
    depth_smoothing: bool = True
    smoothing_sigma: float = 1.0
    
    # Learned normal estimation
    model_path: Optional[str] = None
    model_type: str = "surface_normal_net"  # Type of learned model
    model_device: str = "cuda"
    model_precision: str = "fp16"  # fp16, fp32
    
    # Normal refinement
    enable_refinement: bool = True
    refinement_iterations: int = 3
    consistency_threshold: float = 0.1
    
    # Quality filtering
    confidence_threshold: float = 0.5
    gradient_threshold: float = 0.1  # For detecting depth discontinuities
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not (self.use_geometric_normals or self.use_learned_normals):
            raise ValueError("At least one normal estimation method must be enabled")
        
        if abs(self.geometric_weight + self.learned_weight - 1.0) > 1e-6:
            raise ValueError("Geometric and learned weights must sum to 1.0")
        
        return True


@dataclass
class TemporalTrackingConfig:
    """Configuration for temporal plane tracking."""
    
    # Tracking parameters
    max_distance_threshold: float = 0.2  # Maximum distance for plane association
    max_normal_angle: float = 15.0  # Maximum angle between normals (degrees)
    max_temporal_gap: int = 5  # Maximum frames without detection
    
    # Association parameters (from tests)
    association_threshold: float = 1.0  # Maximum cost for track association
    max_missing_frames: int = 5  # Maximum frames without detection before termination
    smoothing_alpha: float = 0.7  # Alpha for exponential smoothing
    
    # Track lifecycle
    min_track_length: int = 3  # Minimum frames for confirmed track
    max_track_age: int = 30  # Maximum frames to keep lost track
    
    # Stability metrics
    stability_window: int = 10  # Frames for stability computation
    min_stability_score: float = 0.6  # Minimum stability for quality tracks
    
    # Memory management
    max_active_tracks: int = 50
    max_tracks: int = 50  # Maximum total tracks (alias for compatibility)
    cleanup_interval: int = 30  # Frames between cleanup
    
    # Association weights
    center_weight: float = 0.4
    normal_weight: float = 0.3
    area_weight: float = 0.2
    type_penalty: float = 0.5
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.max_distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive")
        
        if not 0 <= self.max_normal_angle <= 180:
            raise ValueError("Normal angle must be in [0, 180] degrees")
        
        return True 