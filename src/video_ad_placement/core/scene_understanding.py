"""
Scene Understanding Module

Implements advanced scene analysis including plane detection, wall identification,
and 3D scene reconstruction for intelligent advertisement placement.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import RANSAC, DBSCAN
from sklearn.linear_model import LinearRegression
import scipy.spatial.distance as distance

from ..utils.logging_utils import get_logger
from ..utils.geometry_utils import (
    fit_plane_ransac, 
    compute_plane_normal,
    project_points_to_plane,
    compute_homography
)
from ..utils.point_cloud_utils import (
    depth_to_pointcloud,
    filter_pointcloud,
    segment_pointcloud
)

logger = get_logger(__name__)


class PlaneType(Enum):
    """Types of detected planes in the scene."""
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    TABLE = "table"
    UNKNOWN = "unknown"


class SurfaceOrientation(Enum):
    """Surface orientation categories."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    OBLIQUE = "oblique"


@dataclass
class Plane:
    """Detected plane in 3D space."""
    coefficients: Tuple[float, float, float, float]  # ax + by + cz + d = 0
    points: np.ndarray  # 3D points belonging to this plane
    normal: np.ndarray  # Normal vector
    center: np.ndarray  # Plane center point
    area: float  # Plane area
    plane_type: PlaneType
    orientation: SurfaceOrientation
    confidence: float
    bbox_2d: Optional[Tuple[int, int, int, int]] = None  # 2D bounding box in image
    texture_quality: Optional[float] = None  # Quality score for ad placement


@dataclass
class SceneLayout:
    """3D scene layout analysis result."""
    planes: List[Plane]
    room_dimensions: Dict[str, float]  # width, height, depth
    dominant_walls: List[Plane]
    floor_plane: Optional[Plane]
    ceiling_plane: Optional[Plane]
    placement_candidates: List[Plane]  # Best planes for ad placement
    scene_complexity: float  # Complexity score
    lighting_analysis: Dict[str, Any]


class SceneUnderstandingConfig:
    """Configuration for scene understanding parameters."""
    
    def __init__(
        self,
        min_plane_points: int = 1000,
        plane_distance_threshold: float = 0.02,
        plane_ransac_iterations: int = 1000,
        min_wall_area: float = 0.5,  # m²
        wall_angle_tolerance: float = 15.0,  # degrees
        floor_ceiling_angle_tolerance: float = 15.0,  # degrees
        enable_texture_analysis: bool = True,
        enable_lighting_analysis: bool = True,
        clustering_eps: float = 0.1,
        clustering_min_samples: int = 50,
        **kwargs
    ):
        self.min_plane_points = min_plane_points
        self.plane_distance_threshold = plane_distance_threshold
        self.plane_ransac_iterations = plane_ransac_iterations
        self.min_wall_area = min_wall_area
        self.wall_angle_tolerance = wall_angle_tolerance
        self.floor_ceiling_angle_tolerance = floor_ceiling_angle_tolerance
        self.enable_texture_analysis = enable_texture_analysis
        self.enable_lighting_analysis = enable_lighting_analysis
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
        self.kwargs = kwargs


class SceneAnalyzer:
    """
    Advanced scene understanding system for 3D scene analysis and plane detection.
    
    This class provides comprehensive scene analysis including plane detection,
    wall identification, and surface quality assessment for advertisement placement.
    
    Attributes:
        config: Configuration object for scene analysis parameters
        is_initialized: Initialization status
        point_cloud_cache: Cached point cloud data
        scene_history: Historical scene analysis for temporal consistency
    """

    def __init__(self, config: SceneUnderstandingConfig):
        """
        Initialize the scene analyzer with specified configuration.
        
        Args:
            config: SceneUnderstandingConfig with analysis parameters
        """
        self.config = config
        self.is_initialized = True
        self.point_cloud_cache: Optional[o3d.geometry.PointCloud] = None
        self.scene_history: List[SceneLayout] = []
        
        logger.info("SceneAnalyzer initialized successfully")

    def analyze_scene(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        camera_intrinsics: Optional[np.ndarray] = None
    ) -> SceneLayout:
        """
        Perform comprehensive scene analysis including plane detection and layout estimation.
        
        Args:
            rgb_image: Input RGB image
            depth_map: Corresponding depth map
            camera_intrinsics: Camera intrinsic parameters (optional)
            
        Returns:
            SceneLayout object with complete scene analysis
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If analysis fails
        """
        try:
            logger.info("Starting comprehensive scene analysis")
            
            # Generate 3D point cloud
            point_cloud = self._generate_point_cloud(rgb_image, depth_map, camera_intrinsics)
            
            # Detect planes in the scene
            planes = self._detect_planes(point_cloud, rgb_image)
            
            # Classify plane types
            classified_planes = self._classify_planes(planes)
            
            # Analyze room layout
            room_dimensions = self._estimate_room_dimensions(classified_planes)
            
            # Identify dominant structures
            dominant_walls = self._identify_dominant_walls(classified_planes)
            floor_plane = self._identify_floor_plane(classified_planes)
            ceiling_plane = self._identify_ceiling_plane(classified_planes)
            
            # Evaluate placement candidates
            placement_candidates = self._evaluate_placement_candidates(classified_planes, rgb_image)
            
            # Compute scene complexity
            scene_complexity = self._compute_scene_complexity(classified_planes, point_cloud)
            
            # Lighting analysis
            lighting_analysis = {}
            if self.config.enable_lighting_analysis:
                lighting_analysis = self._analyze_lighting(rgb_image, classified_planes)
            
            scene_layout = SceneLayout(
                planes=classified_planes,
                room_dimensions=room_dimensions,
                dominant_walls=dominant_walls,
                floor_plane=floor_plane,
                ceiling_plane=ceiling_plane,
                placement_candidates=placement_candidates,
                scene_complexity=scene_complexity,
                lighting_analysis=lighting_analysis
            )
            
            # Cache results for temporal consistency
            self.scene_history.append(scene_layout)
            if len(self.scene_history) > 10:  # Keep last 10 frames
                self.scene_history.pop(0)
            
            logger.info(f"Scene analysis completed: {len(classified_planes)} planes detected")
            return scene_layout
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {str(e)}")
            raise RuntimeError(f"Scene analysis error: {str(e)}")

    def _generate_point_cloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        camera_intrinsics: Optional[np.ndarray]
    ) -> o3d.geometry.PointCloud:
        """Generate 3D point cloud from RGB-D data."""
        height, width = depth_map.shape
        
        # Use default intrinsics if not provided
        if camera_intrinsics is None:
            fx = fy = width * 0.7  # Rough estimate
            cx, cy = width / 2, height / 2
            camera_intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
        # Convert depth map to point cloud using Open3D
        rgb_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            camera_intrinsics[0, 0], camera_intrinsics[1, 1],
            camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        )
        
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )
        
        # Filter and clean point cloud
        point_cloud = self._filter_point_cloud(point_cloud)
        
        # Cache for reuse
        self.point_cloud_cache = point_cloud
        
        return point_cloud

    def _filter_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Filter and clean the point cloud."""
        # Remove statistical outliers
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Downsample for efficiency
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)
        
        return point_cloud

    def _detect_planes(self, point_cloud: o3d.geometry.PointCloud, rgb_image: np.ndarray) -> List[Plane]:
        """Detect planes in the point cloud using RANSAC."""
        planes = []
        remaining_cloud = point_cloud
        
        # Iteratively detect planes
        for i in range(10):  # Maximum 10 planes
            if len(np.asarray(remaining_cloud.points)) < self.config.min_plane_points:
                break
            
            try:
                # Fit plane using RANSAC
                plane_model, inliers = remaining_cloud.segment_plane(
                    distance_threshold=self.config.plane_distance_threshold,
                    ransac_n=3,
                    num_iterations=self.config.plane_ransac_iterations
                )
                
                if len(inliers) < self.config.min_plane_points:
                    break
                
                # Extract plane points
                plane_cloud = remaining_cloud.select_by_index(inliers)
                plane_points = np.asarray(plane_cloud.points)
                
                # Compute plane properties
                normal = np.array(plane_model[:3])
                center = np.mean(plane_points, axis=0)
                area = self._estimate_plane_area(plane_points)
                
                # Determine orientation
                orientation = self._classify_orientation(normal)
                
                # Create plane object
                plane = Plane(
                    coefficients=tuple(plane_model),
                    points=plane_points,
                    normal=normal,
                    center=center,
                    area=area,
                    plane_type=PlaneType.UNKNOWN,  # Will be classified later
                    orientation=orientation,
                    confidence=len(inliers) / len(np.asarray(point_cloud.points))
                )
                
                planes.append(plane)
                
                # Remove detected plane points
                remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
                
            except Exception as e:
                logger.warning(f"Plane detection iteration {i} failed: {str(e)}")
                break
        
        logger.info(f"Detected {len(planes)} planes")
        return planes

    def _estimate_plane_area(self, points: np.ndarray) -> float:
        """Estimate the area of a plane from its points."""
        if len(points) < 3:
            return 0.0
        
        # Use convex hull to estimate area
        try:
            # Project to 2D for area calculation
            pca = np.linalg.svd(points - np.mean(points, axis=0))[2]
            points_2d = points @ pca[:2].T
            
            # Compute convex hull area
            hull = cv2.convexHull(points_2d.astype(np.float32))
            area = cv2.contourArea(hull)
            
            return max(area, 0.0)
            
        except Exception:
            # Fallback: rough estimate based on bounding box
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            dimensions = max_coords - min_coords
            return float(dimensions[0] * dimensions[1])

    def _classify_orientation(self, normal: np.ndarray) -> SurfaceOrientation:
        """Classify surface orientation based on normal vector."""
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Vertical surfaces (walls)
        if abs(normal[1]) < 0.3:  # Y component small
            return SurfaceOrientation.VERTICAL
        
        # Horizontal surfaces (floor/ceiling)
        elif abs(normal[1]) > 0.8:  # Y component large
            return SurfaceOrientation.HORIZONTAL
        
        # Oblique surfaces
        else:
            return SurfaceOrientation.OBLIQUE

    def _classify_planes(self, planes: List[Plane]) -> List[Plane]:
        """Classify detected planes into semantic categories."""
        classified_planes = []
        
        for plane in planes:
            plane_type = self._determine_plane_type(plane, planes)
            plane.plane_type = plane_type
            classified_planes.append(plane)
        
        return classified_planes

    def _determine_plane_type(self, plane: Plane, all_planes: List[Plane]) -> PlaneType:
        """Determine the semantic type of a plane."""
        normal = plane.normal
        center_y = plane.center[1]
        area = plane.area
        
        # Floor detection (horizontal, lowest, large area)
        if (plane.orientation == SurfaceOrientation.HORIZONTAL and 
            normal[1] > 0.8 and  # Normal points up
            area > self.config.min_wall_area):
            
            # Check if it's the lowest horizontal plane
            horizontal_planes = [p for p in all_planes if p.orientation == SurfaceOrientation.HORIZONTAL]
            if horizontal_planes:
                lowest_y = min(p.center[1] for p in horizontal_planes)
                if abs(center_y - lowest_y) < 0.1:
                    return PlaneType.FLOOR
        
        # Ceiling detection (horizontal, highest, large area)
        elif (plane.orientation == SurfaceOrientation.HORIZONTAL and 
              normal[1] < -0.8 and  # Normal points down
              area > self.config.min_wall_area):
            return PlaneType.CEILING
        
        # Wall detection (vertical, large area)
        elif (plane.orientation == SurfaceOrientation.VERTICAL and 
              area > self.config.min_wall_area):
            return PlaneType.WALL
        
        # Table/surface detection (horizontal, medium height)
        elif (plane.orientation == SurfaceOrientation.HORIZONTAL and
              0.5 < center_y < 1.5 and  # Typical table height
              area > 0.1):
            return PlaneType.TABLE
        
        return PlaneType.UNKNOWN

    def _estimate_room_dimensions(self, planes: List[Plane]) -> Dict[str, float]:
        """Estimate room dimensions from detected planes."""
        walls = [p for p in planes if p.plane_type == PlaneType.WALL]
        floor = next((p for p in planes if p.plane_type == PlaneType.FLOOR), None)
        ceiling = next((p for p in planes if p.plane_type == PlaneType.CEILING), None)
        
        dimensions = {"width": 0.0, "height": 0.0, "depth": 0.0}
        
        try:
            if floor and ceiling:
                # Room height
                dimensions["height"] = abs(ceiling.center[1] - floor.center[1])
            
            if len(walls) >= 2:
                # Find perpendicular walls for width and depth
                wall_normals = [w.normal for w in walls]
                
                # Simple approach: use the two largest walls
                walls_by_area = sorted(walls, key=lambda x: x.area, reverse=True)
                
                if len(walls_by_area) >= 2:
                    wall1, wall2 = walls_by_area[:2]
                    
                    # Estimate dimensions from wall separation
                    wall1_points = wall1.points
                    wall2_points = wall2.points
                    
                    # Compute approximate room dimensions
                    if len(wall1_points) > 0 and len(wall2_points) > 0:
                        # Distance between parallel walls
                        center_dist = np.linalg.norm(wall1.center - wall2.center)
                        dimensions["width"] = max(dimensions["width"], center_dist)
                        
                        # Estimate depth from wall extents
                        wall_extent = np.max(wall1_points, axis=0) - np.min(wall1_points, axis=0)
                        dimensions["depth"] = max(dimensions["depth"], np.linalg.norm(wall_extent[[0, 2]]))
        
        except Exception as e:
            logger.warning(f"Room dimension estimation failed: {str(e)}")
        
        return dimensions

    def _identify_dominant_walls(self, planes: List[Plane]) -> List[Plane]:
        """Identify the dominant walls in the scene."""
        walls = [p for p in planes if p.plane_type == PlaneType.WALL]
        
        # Sort by area and confidence
        walls_scored = []
        for wall in walls:
            score = wall.area * wall.confidence
            walls_scored.append((wall, score))
        
        walls_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 walls
        return [wall for wall, score in walls_scored[:3]]

    def _identify_floor_plane(self, planes: List[Plane]) -> Optional[Plane]:
        """Identify the main floor plane."""
        floor_planes = [p for p in planes if p.plane_type == PlaneType.FLOOR]
        
        if not floor_planes:
            return None
        
        # Return the largest floor plane
        return max(floor_planes, key=lambda x: x.area)

    def _identify_ceiling_plane(self, planes: List[Plane]) -> Optional[Plane]:
        """Identify the main ceiling plane."""
        ceiling_planes = [p for p in planes if p.plane_type == PlaneType.CEILING]
        
        if not ceiling_planes:
            return None
        
        # Return the largest ceiling plane
        return max(ceiling_planes, key=lambda x: x.area)

    def _evaluate_placement_candidates(self, planes: List[Plane], rgb_image: np.ndarray) -> List[Plane]:
        """Evaluate planes for advertisement placement suitability."""
        candidates = []
        
        for plane in planes:
            if plane.plane_type in [PlaneType.WALL, PlaneType.TABLE]:
                # Compute placement score
                placement_score = self._compute_placement_score(plane, rgb_image)
                
                if placement_score > 0.5:  # Threshold for placement suitability
                    plane.texture_quality = placement_score
                    candidates.append(plane)
        
        # Sort by placement score
        candidates.sort(key=lambda x: x.texture_quality or 0, reverse=True)
        
        return candidates

    def _compute_placement_score(self, plane: Plane, rgb_image: np.ndarray) -> float:
        """Compute placement suitability score for a plane."""
        score = 0.0
        
        try:
            # Base score from area and confidence
            area_score = min(plane.area / 2.0, 1.0)  # Normalize by 2 m²
            confidence_score = plane.confidence
            
            # Texture analysis score
            texture_score = 0.0
            if self.config.enable_texture_analysis:
                texture_score = self._analyze_texture_quality(plane, rgb_image)
            
            # Orientation preference (vertical walls preferred)
            orientation_score = 1.0 if plane.orientation == SurfaceOrientation.VERTICAL else 0.7
            
            # Combine scores
            score = (area_score * 0.3 + 
                    confidence_score * 0.2 + 
                    texture_score * 0.3 + 
                    orientation_score * 0.2)
            
        except Exception as e:
            logger.warning(f"Placement score computation failed: {str(e)}")
        
        return min(max(score, 0.0), 1.0)

    def _analyze_texture_quality(self, plane: Plane, rgb_image: np.ndarray) -> float:
        """Analyze texture quality for advertisement placement."""
        # This is a simplified implementation
        # In practice, you'd project the plane to image coordinates and analyze the texture
        
        try:
            # For now, return a placeholder score based on plane properties
            # Real implementation would involve:
            # 1. Project 3D plane to 2D image coordinates
            # 2. Extract texture features (gradients, variance, etc.)
            # 3. Assess uniformity and contrast
            
            base_score = 0.7  # Placeholder
            
            # Prefer larger, more confident planes
            size_bonus = min(plane.area / 5.0, 0.2)
            confidence_bonus = plane.confidence * 0.1
            
            return min(base_score + size_bonus + confidence_bonus, 1.0)
            
        except Exception:
            return 0.5

    def _compute_scene_complexity(self, planes: List[Plane], point_cloud: o3d.geometry.PointCloud) -> float:
        """Compute scene complexity score."""
        try:
            # Factors contributing to complexity:
            # 1. Number of planes
            # 2. Variation in plane orientations
            # 3. Point cloud density and variation
            
            plane_count_score = min(len(planes) / 10.0, 1.0)
            
            # Orientation variation
            orientations = [p.orientation for p in planes]
            unique_orientations = len(set(orientations))
            orientation_score = min(unique_orientations / 3.0, 1.0)
            
            # Point cloud complexity (simplified)
            point_count = len(np.asarray(point_cloud.points))
            density_score = min(point_count / 50000.0, 1.0)
            
            complexity = (plane_count_score * 0.4 + 
                         orientation_score * 0.3 + 
                         density_score * 0.3)
            
            return min(max(complexity, 0.0), 1.0)
            
        except Exception:
            return 0.5

    def _analyze_lighting(self, rgb_image: np.ndarray, planes: List[Plane]) -> Dict[str, Any]:
        """Analyze lighting conditions in the scene."""
        try:
            # Convert to grayscale for lighting analysis
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Compute lighting statistics
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Analyze shadows and highlights
            dark_pixels = np.sum(gray < 50) / gray.size
            bright_pixels = np.sum(gray > 200) / gray.size
            
            lighting_analysis = {
                "mean_brightness": float(mean_brightness),
                "brightness_variation": float(brightness_std),
                "shadow_ratio": float(dark_pixels),
                "highlight_ratio": float(bright_pixels),
                "lighting_uniformity": 1.0 - min(brightness_std / 128.0, 1.0),
                "overall_quality": "good" if 0.3 < mean_brightness/255.0 < 0.8 else "poor"
            }
            
            return lighting_analysis
            
        except Exception as e:
            logger.warning(f"Lighting analysis failed: {str(e)}")
            return {"error": str(e)}

    def get_scene_history(self) -> List[SceneLayout]:
        """Get historical scene analysis results."""
        return self.scene_history.copy()

    def detect_planes(
        self,
        depth_map: np.ndarray,
        rgb_image: np.ndarray,
        camera_intrinsics: Optional[np.ndarray] = None
    ) -> List[Plane]:
        """
        Detect planes in the scene (simplified interface).
        
        Args:
            depth_map: Input depth map
            rgb_image: Corresponding RGB image
            camera_intrinsics: Camera parameters
            
        Returns:
            List of detected planes
        """
        try:
            point_cloud = self._generate_point_cloud(rgb_image, depth_map, camera_intrinsics)
            planes = self._detect_planes(point_cloud, rgb_image)
            classified_planes = self._classify_planes(planes)
            
            return classified_planes
            
        except Exception as e:
            logger.error(f"Plane detection failed: {str(e)}")
            return []

    def cleanup(self) -> None:
        """Clean up resources."""
        self.point_cloud_cache = None
        self.scene_history.clear()
        logger.info("SceneAnalyzer cleanup completed") 