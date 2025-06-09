"""
Comprehensive Tests for Temporal Consistency System

Tests for TV placement tracking, Kalman filtering, occlusion handling,
lighting consistency, quality assessment, and prediction/interpolation.
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

from video_ad_placement.scene_understanding.temporal_consistency import (
    TemporalConsistencyManager, TVPlacement, TVGeometry, 
    LightingEnvironment, QualityMetrics, TemporalConfig,
    TVPlacementState, QualityLevel,
    ExtendedKalmanFilter, OcclusionHandler, 
    LightingConsistencyManager, QualityAssessment,
    ArtifactDetector, PredictionInterpolation
)
from video_ad_placement.scene_understanding.geometry_utils import CameraParameters

# Set up logging for tests
logging.basicConfig(level=logging.INFO)


class TestTemporalConfig:
    """Test temporal consistency configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TemporalConfig()
        
        assert config.process_noise_position > 0
        assert config.measurement_noise_position > 0
        assert config.prediction_horizon > 0
        assert 0 <= config.min_visibility <= 1
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = TemporalConfig()
        
        # Should not raise exception
        assert config.validate()
        
        # Test invalid configurations
        config.process_noise_position = -0.1
        with pytest.raises(ValueError):
            config.validate()
        
        config.process_noise_position = 0.01  # Reset
        config.min_visibility = 1.5
        with pytest.raises(ValueError):
            config.validate()


class TestTVGeometry:
    """Test TV geometry representation."""
    
    def test_tv_geometry_creation(self):
        """Test TV geometry creation."""
        position = torch.tensor([0.0, 0.0, 2.0])
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        geometry = TVGeometry(
            position=position,
            orientation=orientation,
            width=1.2,
            height=0.8
        )
        
        assert torch.allclose(geometry.position, position)
        assert torch.allclose(geometry.orientation, orientation)
        assert geometry.width == 1.2
        assert geometry.height == 0.8
    
    def test_get_corners_3d(self):
        """Test 3D corner computation."""
        position = torch.tensor([0.0, 0.0, 2.0])
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        geometry = TVGeometry(
            position=position,
            orientation=orientation,
            width=1.0,
            height=1.0
        )
        
        corners = geometry.get_corners_3d()
        
        assert corners.shape == (4, 3)
        
        # Check corners are at expected positions (centered at origin, width=1, height=1)
        expected_corners = torch.tensor([
            [-0.5, -0.5, 2.0],  # Bottom-left
            [0.5, -0.5, 2.0],   # Bottom-right
            [0.5, 0.5, 2.0],    # Top-right
            [-0.5, 0.5, 2.0]    # Top-left
        ])
        
        assert torch.allclose(corners, expected_corners, atol=1e-6)


class TestLightingEnvironment:
    """Test lighting environment representation."""
    
    def test_lighting_creation(self):
        """Test lighting environment creation."""
        lighting = LightingEnvironment()
        
        # Check default values
        assert lighting.ambient_color.shape == (3,)
        assert lighting.light_direction.shape == (3,)
        assert lighting.light_color.shape == (3,)
        assert 0 <= lighting.ambient_intensity <= 1
        assert 0 <= lighting.light_intensity <= 1


class TestTVPlacement:
    """Test TV placement representation."""
    
    def test_tv_placement_creation(self):
        """Test TV placement creation."""
        geometry = TVGeometry(
            position=torch.tensor([0.0, 0.0, 2.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            width=1.2,
            height=0.8
        )
        
        lighting = LightingEnvironment()
        occlusion_mask = torch.ones((480, 640))
        
        placement = TVPlacement(
            geometry=geometry,
            confidence=0.8,
            visibility=0.9,
            occlusion_mask=occlusion_mask,
            lighting=lighting
        )
        
        assert placement.confidence == 0.8
        assert placement.visibility == 0.9
        assert placement.occlusion_mask.shape == (480, 640)
        assert placement.track_state == TVPlacementState.INITIALIZING


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter implementation."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.ekf = ExtendedKalmanFilter(self.config)
    
    def test_ekf_initialization(self):
        """Test EKF initialization."""
        initial_pose = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        
        self.ekf.initialize(initial_pose)
        
        assert self.ekf.is_initialized
        assert torch.allclose(self.ekf.state[:7], initial_pose)
    
    def test_ekf_prediction(self):
        """Test EKF prediction step."""
        initial_pose = torch.tensor([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        self.ekf.initialize(initial_pose)
        
        # Add some velocity
        self.ekf.state[7:10] = torch.tensor([0.1, 0.0, 0.0])  # Linear velocity
        
        initial_position = self.ekf.state[:3].clone()
        
        # Predict forward
        dt = 0.1
        self.ekf.predict(dt)
        
        # Position should have changed due to velocity
        new_position = self.ekf.state[:3]
        expected_position = initial_position + torch.tensor([0.1, 0.0, 0.0]) * dt
        
        assert torch.allclose(new_position, expected_position, atol=1e-6)
    
    def test_ekf_update(self):
        """Test EKF update step."""
        initial_pose = torch.tensor([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        self.ekf.initialize(initial_pose)
        
        # Simulate measurement
        measurement = torch.tensor([0.1, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        
        self.ekf.update(measurement)
        
        # State should move towards measurement
        updated_pose = self.ekf.get_pose()
        assert not torch.allclose(updated_pose, initial_pose)


class TestOcclusionHandler:
    """Test occlusion detection and handling."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.handler = OcclusionHandler(self.config)
    
    def test_occlusion_mask_generation(self):
        """Test occlusion mask generation."""
        # Create test TV placement
        geometry = TVGeometry(
            position=torch.tensor([0.0, 0.0, 2.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            width=1.0,
            height=1.0
        )
        
        tv_placement = TVPlacement(
            geometry=geometry,
            confidence=0.8,
            visibility=1.0,
            occlusion_mask=torch.ones((480, 640)),
            lighting=LightingEnvironment()
        )
        
        # Create test depth map
        depth_map = torch.ones((480, 640)) * 3.0  # Background at 3m
        depth_map[200:300, 300:400] = 1.5  # Occluding object at 1.5m
        
        # Create camera parameters
        camera_params = CameraParameters(fx=500, fy=500, cx=320, cy=240)
        
        # Generate occlusion mask
        occlusion_mask = self.handler.generate_occlusion_mask(
            [], tv_placement, depth_map, camera_params
        )
        
        assert occlusion_mask.shape == depth_map.shape
        assert 0 <= torch.min(occlusion_mask) <= torch.max(occlusion_mask) <= 1


class TestLightingConsistencyManager:
    """Test lighting consistency management."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.manager = LightingConsistencyManager(self.config)
    
    def test_lighting_smoothing(self):
        """Test lighting sequence smoothing."""
        # Create sequence of lighting environments
        lighting_sequence = []
        confidence_scores = []
        
        for i in range(5):
            lighting = LightingEnvironment()
            lighting.ambient_intensity = 0.3 + i * 0.1  # Gradually change
            lighting.light_intensity = 0.7 - i * 0.05
            
            lighting_sequence.append(lighting)
            confidence_scores.append(0.8)
        
        # Apply smoothing
        smoothed_sequence = self.manager.smooth_lighting_transitions(
            lighting_sequence, confidence_scores
        )
        
        assert len(smoothed_sequence) == len(lighting_sequence)
        
        # Check that smoothing reduces abrupt changes
        for i in range(1, len(smoothed_sequence)):
            intensity_diff = abs(
                smoothed_sequence[i].ambient_intensity - 
                smoothed_sequence[i-1].ambient_intensity
            )
            original_diff = abs(
                lighting_sequence[i].ambient_intensity - 
                lighting_sequence[i-1].ambient_intensity
            )
            
            # Smoothed difference should be smaller (or equal for first frame)
            assert intensity_diff <= original_diff + 1e-6


class TestQualityAssessment:
    """Test quality assessment system."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.assessor = QualityAssessment(self.config)
    
    def test_quality_metrics_computation(self):
        """Test quality metrics computation."""
        # Create dummy rendered sequence
        rendered_sequence = [
            torch.rand(480, 640, 3) for _ in range(10)
        ]
        
        # Create TV placement sequence
        tv_placements = []
        for i in range(10):
            geometry = TVGeometry(
                position=torch.tensor([0.0, 0.0, 2.0 + i * 0.01]),  # Small movement
                orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                width=1.0,
                height=1.0
            )
            
            placement = TVPlacement(
                geometry=geometry,
                confidence=0.8,
                visibility=0.9,
                occlusion_mask=torch.ones((480, 640)),
                lighting=LightingEnvironment(),
                track_state=TVPlacementState.TRACKING
            )
            
            tv_placements.append(placement)
        
        # Assess quality
        metrics = self.assessor.assess_quality(
            rendered_sequence, tv_placements, []
        )
        
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.temporal_stability <= 1
        assert 0 <= metrics.motion_smoothness <= 1
        assert 0 <= metrics.tracking_reliability <= 1
        assert 0 <= metrics.overall_quality <= 1


class TestArtifactDetector:
    """Test artifact detection system."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.detector = ArtifactDetector(self.config)
    
    def test_artifact_detection(self):
        """Test artifact detection in frames."""
        # Create clean frame
        clean_frame = torch.ones(100, 100, 3) * 0.5
        
        # Create noisy frame
        noisy_frame = clean_frame + torch.randn_like(clean_frame) * 0.1
        
        # Detect artifacts
        clean_score = self.detector.detect_artifacts(clean_frame)
        noisy_score = self.detector.detect_artifacts(noisy_frame)
        
        # Noisy frame should have higher artifact score
        assert noisy_score >= clean_score
        assert 0 <= clean_score <= 1
        assert 0 <= noisy_score <= 1


class TestPredictionInterpolation:
    """Test prediction and interpolation system."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig()
        self.predictor = PredictionInterpolation(self.config)
    
    def test_tv_placement_prediction(self):
        """Test TV placement prediction."""
        # Create movement sequence
        tv_history = []
        for i in range(3):
            geometry = TVGeometry(
                position=torch.tensor([i * 0.1, 0.0, 2.0]),  # Moving in X
                orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                width=1.0,
                height=1.0
            )
            
            placement = TVPlacement(
                geometry=geometry,
                confidence=0.8,
                visibility=0.9,
                occlusion_mask=torch.ones((480, 640)),
                lighting=LightingEnvironment()
            )
            
            tv_history.append(placement)
        
        # Predict next placement
        predicted = self.predictor.predict_tv_placement(tv_history, frames_ahead=1)
        
        assert predicted.is_interpolated
        assert predicted.prediction_confidence > 0
        
        # Position should continue the motion
        expected_x = 3 * 0.1  # Next position in sequence
        assert abs(predicted.geometry.position[0].item() - expected_x) < 0.01
    
    def test_tv_placement_interpolation(self):
        """Test TV placement interpolation."""
        # Create start and end placements
        start_geometry = TVGeometry(
            position=torch.tensor([0.0, 0.0, 2.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            width=1.0,
            height=1.0
        )
        
        end_geometry = TVGeometry(
            position=torch.tensor([1.0, 0.0, 2.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            width=1.0,
            height=1.0
        )
        
        start_placement = TVPlacement(
            geometry=start_geometry,
            confidence=0.8,
            visibility=0.9,
            occlusion_mask=torch.ones((480, 640)),
            lighting=LightingEnvironment()
        )
        
        end_placement = TVPlacement(
            geometry=end_geometry,
            confidence=0.8,
            visibility=0.9,
            occlusion_mask=torch.ones((480, 640)),
            lighting=LightingEnvironment()
        )
        
        # Interpolate between placements
        interpolated = self.predictor.interpolate_tv_placements(
            start_placement, end_placement, num_intermediate=2
        )
        
        assert len(interpolated) == 2
        
        # Check interpolated positions
        for i, placement in enumerate(interpolated):
            assert placement.is_interpolated
            
            # Position should be interpolated
            expected_x = (i + 1) / 3  # 1/3 and 2/3 of the way
            actual_x = placement.geometry.position[0].item()
            assert abs(actual_x - expected_x) < 0.01


class TestTemporalConsistencyManager:
    """Test main temporal consistency manager."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.config = TemporalConfig(enable_multithreading=False)  # Disable for testing
        self.manager = TemporalConsistencyManager(self.config)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.current_frame_id == 0
        assert len(self.manager.tv_placement_history) == 0
        assert self.manager.last_valid_placement is None
    
    def test_track_tv_placement(self):
        """Test TV placement tracking."""
        # Create mock detection
        class MockDetection:
            def __init__(self, position, confidence=0.8):
                self.position = position
                self.confidence = confidence
                self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
                self.width = 1.0
                self.height = 0.8
        
        detection = MockDetection(torch.tensor([0.0, 0.0, 2.0]))
        
        # Create test data
        depth_map = torch.ones((480, 640)) * 2.5
        rgb_frame = torch.ones((480, 640, 3)) * 0.5
        
        # Track placement
        placements = self.manager.track_tv_placement(
            [detection], [], depth_map, rgb_frame
        )
        
        assert len(placements) > 0
        assert self.manager.current_frame_id == 1
        assert len(self.manager.tv_placement_history) > 0
    
    def test_quality_assessment(self):
        """Test quality assessment."""
        # Create dummy rendered sequence
        rendered_sequence = [
            torch.rand(100, 100, 3) for _ in range(5)
        ]
        
        # Assess quality
        metrics = self.manager.assess_quality(rendered_sequence)
        
        assert isinstance(metrics, QualityMetrics)
        assert hasattr(metrics, 'overall_quality')
        assert hasattr(metrics, 'quality_level')
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.manager.get_performance_stats()
        
        assert 'avg_processing_time' in stats
        assert 'frames_processed' in stats
        assert 'memory_usage_mb' in stats
        assert isinstance(stats['frames_processed'], int)
    
    def test_manager_reset(self):
        """Test manager reset."""
        # Add some history
        self.manager.current_frame_id = 5
        
        # Reset
        self.manager.reset()
        
        assert self.manager.current_frame_id == 0
        assert len(self.manager.tv_placement_history) == 0
        assert self.manager.last_valid_placement is None


class TestIntegration:
    """Integration tests for temporal consistency system."""
    
    def test_full_pipeline(self):
        """Test complete temporal consistency pipeline."""
        config = TemporalConfig(enable_multithreading=False)
        manager = TemporalConsistencyManager(config)
        
        # Simulate video sequence
        for frame_id in range(10):
            # Create mock detection with slight movement
            class MockDetection:
                def __init__(self, frame_id):
                    self.position = torch.tensor([frame_id * 0.01, 0.0, 2.0])
                    self.confidence = 0.8
                    self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
                    self.width = 1.0
                    self.height = 0.8
            
            detection = MockDetection(frame_id)
            
            # Create test data
            depth_map = torch.ones((240, 320)) * 2.5
            rgb_frame = torch.ones((240, 320, 3)) * 0.5
            
            # Track placement
            placements = manager.track_tv_placement(
                [detection], [], depth_map, rgb_frame
            )
            
            assert len(placements) > 0
        
        # Check that history was maintained
        assert len(manager.tv_placement_history) > 0
        assert manager.current_frame_id == 10
        
        # Test prediction
        predicted = manager.predict_placement(frames_ahead=1)
        assert predicted is not None
        assert predicted.is_interpolated
    
    def test_occlusion_handling(self):
        """Test occlusion handling in pipeline."""
        config = TemporalConfig()
        manager = TemporalConsistencyManager(config)
        
        # Create test TV placement
        geometry = TVGeometry(
            position=torch.tensor([0.0, 0.0, 2.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            width=1.0,
            height=1.0
        )
        
        tv_placement = TVPlacement(
            geometry=geometry,
            confidence=0.8,
            visibility=1.0,
            occlusion_mask=torch.ones((240, 320)),
            lighting=LightingEnvironment()
        )
        
        # Create depth map with occlusion
        depth_map = torch.ones((240, 320)) * 3.0
        depth_map[100:150, 150:200] = 1.5  # Occluding object
        
        camera_params = CameraParameters(fx=300, fy=300, cx=160, cy=120)
        
        # Generate occlusion mask
        occlusion_mask = manager.generate_occlusion_masks(
            [], tv_placement, depth_map, camera_params
        )
        
        assert occlusion_mask.shape == depth_map.shape
        assert torch.min(occlusion_mask) >= 0
        assert torch.max(occlusion_mask) <= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        config = TemporalConfig()
        manager = TemporalConsistencyManager(config)
        
        # Empty detections
        depth_map = torch.ones((240, 320)) * 2.5
        rgb_frame = torch.ones((240, 320, 3)) * 0.5
        
        placements = manager.track_tv_placement([], [], depth_map, rgb_frame)
        
        # Should handle gracefully
        assert isinstance(placements, list)
    
    def test_invalid_configuration(self):
        """Test invalid configuration handling."""
        # Invalid noise values
        with pytest.raises(ValueError):
            config = TemporalConfig(process_noise_position=-0.1)
            config.validate()
        
        # Invalid visibility range
        with pytest.raises(ValueError):
            config = TemporalConfig(min_visibility=1.5)
            config.validate()
    
    def test_prediction_without_history(self):
        """Test prediction without sufficient history."""
        config = TemporalConfig()
        predictor = PredictionInterpolation(config)
        
        # Should raise error for empty history
        with pytest.raises(ValueError):
            predictor.predict_tv_placement([])
    
    def test_quality_assessment_edge_cases(self):
        """Test quality assessment with edge cases."""
        config = TemporalConfig()
        assessor = QualityAssessment(config)
        
        # Empty sequences
        metrics = assessor.assess_quality([], [], [])
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_quality >= 0
        
        # Single frame
        single_frame = [torch.rand(100, 100, 3)]
        metrics = assessor.assess_quality(single_frame, [], [])
        assert isinstance(metrics, QualityMetrics)


if __name__ == "__main__":
    # Run specific test methods for debugging
    test_config = TestTemporalConfig()
    test_config.test_default_config()
    test_config.test_config_validation()
    
    test_geometry = TestTVGeometry()
    test_geometry.test_tv_geometry_creation()
    test_geometry.test_get_corners_3d()
    
    print("Basic temporal consistency tests passed!")
    
    # Run more comprehensive tests
    pytest.main([__file__, "-v"]) 