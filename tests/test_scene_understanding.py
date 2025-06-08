"""
Comprehensive Unit Tests for 3D Scene Understanding System

This module provides thorough testing of all components including plane detection,
surface normal estimation, temporal tracking, and quality assessment for
wall surface analysis in TV advertisement placement scenarios.
"""

import pytest
import torch
import numpy as np
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from video_ad_placement.scene_understanding.plane_detector import PlaneDetector
from video_ad_placement.scene_understanding.surface_normal_estimator import SurfaceNormalEstimator
from video_ad_placement.scene_understanding.temporal_tracker import TemporalPlaneTracker, PlaneTrack
from video_ad_placement.scene_understanding.plane_models import (
    PlaneDetectionConfig, SurfaceNormalConfig, TemporalTrackingConfig,
    Plane, PlaneType, PlaneQuality
)
from video_ad_placement.scene_understanding.geometry_utils import (
    GeometryUtils, CameraParameters, CameraProjection, PlaneGeometry
)

# Set up logging for tests
logging.basicConfig(level=logging.INFO)


class TestGeometryUtils:
    """Test geometric utility functions."""
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        # Test 3D vector
        vector = torch.tensor([3.0, 4.0, 0.0])
        normalized = GeometryUtils.normalize_vector(vector)
        
        assert torch.allclose(torch.norm(normalized), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(normalized, torch.tensor([0.6, 0.8, 0.0]), atol=1e-6)
        
        # Test batch of vectors
        vectors = torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
        normalized_batch = GeometryUtils.normalize_vector(vectors)
        
        assert normalized_batch.shape == (2, 3)
        assert torch.allclose(torch.norm(normalized_batch, dim=1), torch.ones(2), atol=1e-6)
    
    def test_angle_between_vectors(self):
        """Test angle computation between vectors."""
        # Parallel vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([2.0, 0.0, 0.0])
        angle = GeometryUtils.compute_angle_between_vectors(v1, v2)
        assert torch.allclose(angle, torch.tensor(0.0), atol=1e-6)
        
        # Perpendicular vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        angle = GeometryUtils.compute_angle_between_vectors(v1, v2)
        assert torch.allclose(angle, torch.tensor(np.pi/2), atol=1e-6)
        
        # Opposite vectors
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0, 0.0])
        angle = GeometryUtils.compute_angle_between_vectors(v1, v2)
        assert torch.allclose(angle, torch.tensor(np.pi), atol=1e-6)
    
    def test_point_to_plane_distance(self):
        """Test point to plane distance calculation."""
        # Plane at z=0 with normal pointing up
        plane_normal = torch.tensor([0.0, 0.0, 1.0])
        plane_point = torch.tensor([0.0, 0.0, 0.0])
        
        # Points above and below plane
        points = torch.tensor([
            [0.0, 0.0, 1.0],   # 1 unit above
            [0.0, 0.0, -2.0],  # 2 units below
            [1.0, 1.0, 0.0]    # On plane
        ])
        
        distances = GeometryUtils.point_to_plane_distance(points, plane_normal, plane_point)
        expected = torch.tensor([1.0, -2.0, 0.0])
        
        assert torch.allclose(distances, expected, atol=1e-6)
    
    def test_project_points_to_plane(self):
        """Test point projection onto plane."""
        # Plane at z=0
        plane_normal = torch.tensor([0.0, 0.0, 1.0])
        plane_point = torch.tensor([0.0, 0.0, 0.0])
        
        points = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, -2.0]
        ])
        
        projected = GeometryUtils.project_points_to_plane(points, plane_normal, plane_point)
        expected = torch.tensor([
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 0.0]
        ])
        
        assert torch.allclose(projected, expected, atol=1e-6)
    
    def test_compute_plane_from_points(self):
        """Test plane fitting from points."""
        # Create points on XY plane (z=0)
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        normal, center = GeometryUtils.compute_plane_from_points(points)
        
        # Normal should be close to [0, 0, 1] or [0, 0, -1]
        assert torch.allclose(torch.abs(normal[2]), torch.tensor(1.0), atol=1e-3)
        
        # Center should be at (0.5, 0.5, 0)
        expected_center = torch.tensor([0.5, 0.5, 0.0])
        assert torch.allclose(center, expected_center, atol=1e-3)


class TestCameraProjection:
    """Test camera projection utilities."""
    
    def setup_method(self):
        """Set up test camera parameters."""
        self.camera_params = CameraParameters(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0
        )
        self.projection = CameraProjection(self.camera_params)
    
    def test_project_3d_to_2d(self):
        """Test 3D to 2D projection."""
        # Point at (0, 0, 1) should project to principal point
        points_3d = torch.tensor([[0.0, 0.0, 1.0]])
        points_2d = self.projection.project_points_3d_to_2d(points_3d)
        
        expected = torch.tensor([[320.0, 240.0]])
        assert torch.allclose(points_2d, expected, atol=1e-3)
        
        # Point at (1, 0, 1) should project to (cx + fx, cy)
        points_3d = torch.tensor([[1.0, 0.0, 1.0]])
        points_2d = self.projection.project_points_3d_to_2d(points_3d)
        
        expected = torch.tensor([[845.0, 240.0]])
        assert torch.allclose(points_2d, expected, atol=1e-3)
    
    def test_unproject_2d_to_3d(self):
        """Test 2D to 3D unprojection."""
        # Principal point at depth 1 should unproject to (0, 0, 1)
        points_2d = torch.tensor([[320.0, 240.0]])
        depths = torch.tensor([1.0])
        points_3d = self.projection.unproject_2d_to_3d(points_2d, depths)
        
        expected = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.allclose(points_3d, expected, atol=1e-6)
    
    def test_depth_map_to_point_cloud(self):
        """Test depth map to point cloud conversion."""
        # Create simple 2x2 depth map
        depth_map = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        
        points_3d = self.projection.depth_map_to_point_cloud(depth_map)
        
        # Should have 4 points
        assert points_3d.shape[0] == 4
        assert points_3d.shape[1] == 3


class TestPlaneGeometry:
    """Test plane-specific geometry functions."""
    
    def test_fit_plane_ransac(self):
        """Test RANSAC plane fitting."""
        # Create points on XY plane with some noise
        n_points = 100
        x = torch.randn(n_points)
        y = torch.randn(n_points)
        z = torch.zeros(n_points) + torch.randn(n_points) * 0.01  # Small noise
        
        points = torch.stack([x, y, z], dim=1)
        
        normal, center, inliers = PlaneGeometry.fit_plane_ransac(
            points, num_iterations=500, distance_threshold=0.05
        )
        
        # Normal should be close to [0, 0, 1]
        assert torch.abs(normal[2]) > 0.8
        
        # Most points should be inliers
        assert torch.sum(inliers) > n_points * 0.8
    
    def test_compute_plane_area_from_mask(self):
        """Test plane area computation from mask."""
        # Create simple depth map and mask
        depth_map = torch.ones(10, 10) * 2.0  # 2 meters depth
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[2:8, 2:8] = True  # 6x6 pixel square
        
        camera_params = CameraParameters(fx=100, fy=100, cx=5, cy=5)
        
        area = PlaneGeometry.compute_plane_area_from_mask(mask, depth_map, camera_params)
        
        # Area should be positive
        assert area > 0.0


class TestPlaneDetectionConfig:
    """Test plane detection configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PlaneDetectionConfig()
        
        assert config.ransac_iterations > 0
        assert config.ransac_threshold > 0
        assert config.min_inliers > 0
        assert 0 < config.max_inliers_ratio <= 1
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = PlaneDetectionConfig()
        
        # Should not raise exception
        config.validate()
        
        # Test invalid configuration
        config.ransac_iterations = -1
        with pytest.raises(ValueError):
            config.validate()


class TestSurfaceNormalEstimator:
    """Test surface normal estimation."""
    
    def test_geometric_normal_estimation(self):
        """Test geometric surface normal estimation."""
        config = SurfaceNormalConfig(
            use_geometric_normals=True,
            use_learned_normals=False
        )
        estimator = SurfaceNormalEstimator(config)
        
        # Create tilted plane depth map
        height, width = 20, 20
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        
        # Tilted plane: z = 1 + 0.1*x
        depth_map = 1.0 + 0.1 * x_coords / width
        rgb_frame = torch.zeros(height, width, 3)
        
        normals = estimator.estimate_surface_normals(depth_map, rgb_frame)
        
        assert normals.shape == (height, width, 3)
        
        # Check that normals are approximately normalized
        norms = torch.norm(normals, dim=2)
        valid_mask = norms > 0.5
        if torch.any(valid_mask):
            valid_norms = norms[valid_mask]
            assert torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-1)


class TestPlaneDetector:
    """Test plane detection system."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = PlaneDetectionConfig(
            ransac_iterations=100,  # Reduced for testing
            ransac_threshold=0.05,
            min_inliers=10,
            min_plane_area=0.1
        )
        self.detector = PlaneDetector(self.config)
    
    def test_plane_detection_simple_scene(self):
        """Test plane detection on simple synthetic scene."""
        # Create simple room-like scene
        height, width = 60, 80
        depth_map = torch.zeros(height, width)
        rgb_frame = torch.zeros(height, width, 3)
        
        # Floor
        depth_map[40:, :] = 3.0
        rgb_frame[40:, :] = torch.tensor([100, 100, 100])
        
        # Wall
        depth_map[10:40, 20:60] = 4.0
        rgb_frame[10:40, 20:60] = torch.tensor([150, 120, 100])
        
        # Camera intrinsics
        camera_intrinsics = torch.tensor([
            [100.0, 0.0, width/2],
            [0.0, 100.0, height/2],
            [0.0, 0.0, 1.0]
        ])
        
        # Detect planes
        planes = self.detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
        
        # Should detect at least one plane
        assert len(planes) > 0
        
        # Check plane properties
        for plane in planes:
            assert plane.confidence > 0
            assert plane.area > 0
            assert torch.norm(plane.normal) > 0.9  # Should be approximately normalized
            assert plane.plane_type in PlaneType
            assert plane.quality in PlaneQuality
    
    def test_detect_empty_scene(self):
        """Test detection on empty scene."""
        height, width = 30, 40
        depth_map = torch.zeros(height, width)
        rgb_frame = torch.zeros(height, width, 3)
        
        camera_intrinsics = torch.eye(3)
        
        planes = self.detector.detect_planes(depth_map, rgb_frame.float(), camera_intrinsics)
        
        # Should handle empty scene gracefully
        assert isinstance(planes, list)
    
    def test_wall_surface_validation(self):
        """Test wall surface validation logic."""
        # Create a good wall plane
        normal = torch.tensor([0.0, 0.0, 1.0])  # Facing camera
        point = torch.tensor([0.0, 0.0, 2.0])
        equation = torch.tensor([0.0, 0.0, 1.0, -2.0])
        inlier_mask = torch.ones(20, 20, dtype=torch.bool)
        
        plane = Plane(
            normal=normal,
            point=point,
            equation=equation,
            inlier_mask=inlier_mask,
            depth_values=torch.ones(400) * 2.0,
            rgb_values=torch.ones(400, 3) * 128,
            confidence=0.8,
            area=5.0,  # Good size for TV
            stability_score=0.9,
            plane_type=PlaneType.WALL,  # Explicitly set as WALL
            quality=PlaneQuality.GOOD,  # Set quality to GOOD
            tv_placement_score=0.8  # Set a good TV placement score
        )
        
        # Should be suitable for TV
        assert plane.is_wall_suitable_for_tv()
        
        # Test small plane
        plane.area = 0.5  # Too small
        assert not plane.is_wall_suitable_for_tv()


class TestTemporalTracking:
    """Test temporal tracking system."""
    
    def setup_method(self):
        """Set up temporal tracking test."""
        self.config = TemporalTrackingConfig(
            association_threshold=1.0,
            max_missing_frames=3,
            smoothing_alpha=0.7
        )
        self.tracker = TemporalPlaneTracker(self.config)
    
    def test_track_creation(self):
        """Test creation of new tracks."""
        # Create test plane
        plane = self._create_test_plane([0, 0, 2], [0, 0, 1])
        
        # Update tracker
        tracks = self.tracker.update([plane])
        
        assert len(tracks) == 1
        assert tracks[0].total_detections == 1
        assert tracks[0].track_id == 0
    
    def test_track_association(self):
        """Test plane association across frames."""
        # Create two similar planes
        plane1 = self._create_test_plane([0, 0, 2], [0, 0, 1])
        plane2 = self._create_test_plane([0.05, 0, 2.05], [0, 0, 1])  # Slightly moved
        
        # First frame
        tracks1 = self.tracker.update([plane1])
        assert len(tracks1) == 1
        
        # Second frame
        tracks2 = self.tracker.update([plane2])
        assert len(tracks2) == 1
        assert tracks2[0].total_detections == 2  # Should associate with existing track
    
    def test_track_termination(self):
        """Test track termination after missing frames."""
        plane = self._create_test_plane([0, 0, 2], [0, 0, 1])
        
        # Create track
        self.tracker.update([plane])
        
        # Miss several frames
        for _ in range(self.config.max_missing_frames + 1):
            tracks = self.tracker.update([])
        
        # Track should be terminated
        assert len(tracks) == 0
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing of plane parameters."""
        # Create track with multiple detections
        planes = [
            self._create_test_plane([0, 0, 2.0], [0, 0, 1]),
            self._create_test_plane([0, 0, 2.1], [0, 0, 1]),
            self._create_test_plane([0, 0, 1.9], [0, 0, 1])
        ]
        
        for plane in planes:
            self.tracker.update([plane])
        
        # Get smoothed planes
        smoothed = self.tracker.get_smoothed_planes(min_stability=0.1)
        
        assert len(smoothed) == 1
        # Smoothed position should be between extremes
        smoothed_z = smoothed[0].point[2].item()
        assert 1.9 <= smoothed_z <= 2.1
    
    def _create_test_plane(self, point, normal):
        """Helper to create test plane."""
        point = torch.tensor(point, dtype=torch.float32)
        normal = torch.tensor(normal, dtype=torch.float32)
        normal = normal / torch.norm(normal)
        
        equation = torch.cat([normal, -torch.dot(normal, point).unsqueeze(0)])
        inlier_mask = torch.ones(10, 10, dtype=torch.bool)
        depth_values = torch.ones(100) * 2.0  # Add depth values
        rgb_values = torch.ones(100, 3) * 128  # Add RGB values
        
        return Plane(
            normal=normal,
            point=point,
            equation=equation,
            inlier_mask=inlier_mask,
            depth_values=depth_values,
            rgb_values=rgb_values,
            confidence=0.8,
            area=2.0,
            stability_score=0.7
        )


class TestIntegration:
    """Integration tests for complete system."""
    
    def test_full_pipeline(self):
        """Test complete detection and tracking pipeline."""
        # Create detector and tracker
        detection_config = PlaneDetectionConfig(
            ransac_iterations=100,
            min_inliers=5
        )
        tracking_config = TemporalTrackingConfig()
        
        detector = PlaneDetector(detection_config)
        tracker = TemporalPlaneTracker(tracking_config)
        
        # Process multiple frames
        for frame_id in range(5):
            # Create synthetic scene
            depth_map, rgb_frame = self._create_synthetic_scene()
            camera_intrinsics = torch.eye(3)
            
            # Detect planes
            planes = detector.detect_planes(depth_map, rgb_frame, camera_intrinsics)
            
            # Update tracker
            tracks = tracker.update(planes)
            
            # Verify results
            assert isinstance(planes, list)
            assert isinstance(tracks, list)
    
    def test_wall_detection_and_quality_assessment(self):
        """Test wall detection and quality assessment."""
        config = PlaneDetectionConfig(
            ransac_iterations=500,  # More iterations for better detection
            min_plane_area=0.5,     # Lower area requirement
            min_inliers=20,         # Lower inlier requirement
            wall_normal_tolerance=0.5  # More lenient normal tolerance
        )
        detector = PlaneDetector(config)
        
        # Create scene with clear wall
        depth_map, rgb_frame = self._create_wall_scene()
        camera_intrinsics = torch.tensor([
            [100.0, 0.0, 30.0],
            [0.0, 100.0, 20.0],
            [0.0, 0.0, 1.0]
        ])
        
        planes = detector.detect_planes(depth_map, rgb_frame, camera_intrinsics)
        
        # Should detect some planes in the scene
        assert len(planes) > 0, "No planes detected in synthetic scene"
        
        # Check quality assessment works for all detected planes
        for plane in planes:
            assert plane.tv_placement_score >= 0.0
            assert plane.tv_placement_score <= 1.0
            assert plane.confidence >= 0.0
            assert plane.confidence <= 1.0
            assert plane.area >= 0.0
            assert hasattr(plane, 'plane_type')
            assert hasattr(plane, 'quality')
    
    def _create_synthetic_scene(self):
        """Create synthetic scene for testing."""
        height, width = 40, 60
        depth_map = torch.ones(height, width) * 3.0
        rgb_frame = torch.ones(height, width, 3) * 128
        
        # Add wall region
        depth_map[10:30, 20:40] = 4.0
        rgb_frame[10:30, 20:40] = torch.tensor([150, 120, 100])
        
        return depth_map, rgb_frame.float()
    
    def _create_wall_scene(self):
        """Create scene with prominent wall."""
        height, width = 40, 60
        depth_map = torch.ones(height, width) * 5.0  # Background depth
        rgb_frame = torch.ones(height, width, 3) * 50  # Background color
        
        # Large wall - create a clear planar wall surface
        wall_depth = 3.0
        depth_map[5:35, 10:50] = wall_depth  # Wall at consistent depth
        rgb_frame[5:35, 10:50] = torch.tensor([140, 120, 110])
        
        # Add some floor area for contrast
        depth_map[35:, :] = 4.0
        rgb_frame[35:, :] = torch.tensor([100, 100, 100])
        
        return depth_map, rgb_frame.float()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        config = PlaneDetectionConfig()
        detector = PlaneDetector(config)
        
        # Empty depth map
        depth_map = torch.zeros(0, 0)
        rgb_frame = torch.zeros(0, 0, 3)
        camera_intrinsics = torch.eye(3)
        
        planes = detector.detect_planes(depth_map, rgb_frame, camera_intrinsics)
        assert isinstance(planes, list)
        assert len(planes) == 0
    
    def test_configuration_validation(self):
        """Test configuration validation errors."""
        # Invalid RANSAC iterations
        with pytest.raises(ValueError):
            config = PlaneDetectionConfig(ransac_iterations=-1)
            config.validate()
        
        # Invalid threshold
        with pytest.raises(ValueError):
            config = PlaneDetectionConfig(ransac_threshold=-0.1)
            config.validate()
    
    def test_memory_management(self):
        """Test memory management in long sequences."""
        config = TemporalTrackingConfig(max_tracks=5)
        tracker = TemporalPlaneTracker(config)
        
        # Create many tracks
        for i in range(10):
            plane = Plane(
                normal=torch.tensor([0.0, 0.0, 1.0]),
                point=torch.tensor([float(i), 0.0, 2.0]),
                equation=torch.tensor([0.0, 0.0, 1.0, -2.0]),
                inlier_mask=torch.ones(5, 5, dtype=torch.bool),
                depth_values=torch.ones(25) * 2.0,
                rgb_values=torch.ones(25, 3) * 128,
                confidence=0.5,
                area=1.0,
                stability_score=0.5
            )
            tracker.update([plane])
        
        # Should limit number of tracks
        assert len(tracker.tracks) <= config.max_tracks


if __name__ == "__main__":
    # Run specific test methods for debugging
    test_geometry = TestGeometryUtils()
    test_geometry.test_normalize_vector()
    test_geometry.test_angle_between_vectors()
    
    test_camera = TestCameraProjection()
    test_camera.setup_method()
    test_camera.test_project_3d_to_2d()
    
    print("Basic tests passed!")
    
    # Run more comprehensive tests
    pytest.main([__file__, "-v"]) 