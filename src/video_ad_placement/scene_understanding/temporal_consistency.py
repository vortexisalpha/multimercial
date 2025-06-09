"""
Advanced Temporal Consistency System for TV Advertisement Placement

This module implements sophisticated temporal tracking and consistency management
for TV placement in video sequences, including Kalman filter-based tracking,
dynamic occlusion handling, lighting consistency, and quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque, defaultdict
import threading
from abc import ABC, abstractmethod
import queue

from .plane_models import Plane, PlaneType
from .geometry_utils import GeometryUtils, CameraParameters
from .temporal_tracker import PlaneTrack

logger = logging.getLogger(__name__)


class TVPlacementState(Enum):
    """TV placement tracking states."""
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    OCCLUDED = "occluded"
    LOST = "lost"
    RECOVERING = "recovering"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class TVGeometry:
    """TV placement geometry information."""
    
    # 6-DOF pose (position + orientation)
    position: torch.Tensor  # 3D position (x, y, z)
    orientation: torch.Tensor  # Quaternion (w, x, y, z) or rotation matrix (3, 3)
    
    # TV dimensions
    width: float  # TV width in meters
    height: float  # TV height in meters
    depth: float = 0.05  # TV depth in meters
    
    # Placement confidence
    pose_confidence: float = 0.0
    scale_confidence: float = 0.0
    
    # Validation flags
    is_valid: bool = True
    validation_score: float = 0.0
    
    def get_corners_3d(self) -> torch.Tensor:
        """Get 3D corners of TV placement."""
        # TV corners in local coordinate system
        corners_local = torch.tensor([
            [-self.width/2, -self.height/2, 0],
            [self.width/2, -self.height/2, 0],
            [self.width/2, self.height/2, 0],
            [-self.width/2, self.height/2, 0]
        ], dtype=torch.float32)
        
        # Transform to world coordinates
        if self.orientation.shape == (4,):  # Quaternion
            rotation_matrix = self._quaternion_to_rotation_matrix(self.orientation)
        else:  # Rotation matrix
            rotation_matrix = self.orientation
        
        corners_world = torch.mm(corners_local, rotation_matrix.T) + self.position.unsqueeze(0)
        
        return corners_world
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        
        # Rotation matrix from quaternion
        R = torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R


@dataclass 
class LightingEnvironment:
    """Lighting environment for TV placement."""
    
    # Global illumination
    ambient_color: torch.Tensor = field(default_factory=lambda: torch.tensor([0.3, 0.3, 0.3]))
    ambient_intensity: float = 0.3
    
    # Directional lighting
    light_direction: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, -1.0, 0.5]))
    light_color: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 1.0, 1.0]))
    light_intensity: float = 0.7
    
    # Shadow parameters
    shadow_intensity: float = 0.4
    shadow_softness: float = 0.1
    
    # Environment-specific
    scene_brightness: float = 0.5
    contrast_ratio: float = 1.0
    color_temperature: float = 6500.0  # Kelvin
    
    # Temporal consistency
    confidence: float = 0.0
    stability_score: float = 0.0


@dataclass
class TVPlacement:
    """Complete TV placement information with temporal data."""
    
    # Core placement data
    geometry: TVGeometry
    confidence: float
    visibility: float
    occlusion_mask: torch.Tensor
    lighting: LightingEnvironment
    
    # Temporal tracking
    placement_id: int = -1
    track_state: TVPlacementState = TVPlacementState.INITIALIZING
    frame_id: int = -1
    
    # Quality metrics
    temporal_stability: float = 0.0
    spatial_consistency: float = 0.0
    lighting_consistency: float = 0.0
    
    # Associated plane information
    wall_plane: Optional[Plane] = None
    wall_confidence: float = 0.0
    
    # Validation and debugging
    is_interpolated: bool = False
    prediction_confidence: float = 0.0
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Comprehensive quality assessment metrics."""
    
    # Temporal consistency
    temporal_stability: float = 0.0
    motion_smoothness: float = 0.0
    tracking_reliability: float = 0.0
    
    # Occlusion handling
    occlusion_accuracy: float = 0.0
    occlusion_completeness: float = 0.0
    occlusion_consistency: float = 0.0
    
    # Lighting consistency
    lighting_consistency: float = 0.0
    shadow_consistency: float = 0.0
    color_consistency: float = 0.0
    
    # Perceptual quality
    visual_stability: float = 0.0
    artifact_score: float = 0.0
    realism_score: float = 0.0
    
    # Overall assessment
    overall_quality: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR
    
    # Detailed metrics
    metrics_detail: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def compute_overall_quality(self):
        """Compute overall quality from component metrics."""
        # Weighted combination of quality metrics
        weights = {
            'temporal_stability': 0.25,
            'occlusion_accuracy': 0.20,
            'lighting_consistency': 0.20,
            'visual_stability': 0.15,
            'motion_smoothness': 0.10,
            'realism_score': 0.10
        }
        
        self.overall_quality = sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )
        
        # Determine quality level
        if self.overall_quality >= 0.9:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.overall_quality >= 0.75:
            self.quality_level = QualityLevel.GOOD
        elif self.overall_quality >= 0.6:
            self.quality_level = QualityLevel.FAIR
        elif self.overall_quality >= 0.4:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.UNACCEPTABLE


@dataclass
class TemporalConfig:
    """Configuration for temporal consistency system."""
    
    # Kalman filter parameters
    process_noise_position: float = 0.01
    process_noise_orientation: float = 0.005
    measurement_noise_position: float = 0.02
    measurement_noise_orientation: float = 0.01
    
    # Tracking parameters
    max_tracking_distance: float = 0.5  # Max distance for association
    max_orientation_angle: float = 30.0  # Max angle difference (degrees)
    prediction_horizon: int = 5  # Frames to predict ahead
    
    # Occlusion handling
    occlusion_threshold: float = 0.1  # Depth threshold for occlusion
    min_visibility: float = 0.3  # Minimum visibility to maintain tracking
    occlusion_buffer_frames: int = 10  # Frames to buffer during occlusion
    
    # Lighting consistency
    lighting_smoothing_alpha: float = 0.8
    color_change_threshold: float = 0.1
    lighting_adaptation_frames: int = 30
    
    # Quality assessment
    quality_window_size: int = 30  # Frames for quality assessment
    artifact_detection_threshold: float = 0.1
    stability_threshold: float = 0.8
    
    # Performance optimization
    enable_multithreading: bool = True
    max_threads: int = 4
    memory_limit_mb: int = 512
    processing_queue_size: int = 100
    
    # Debugging and logging
    enable_debug_logging: bool = False
    save_debug_data: bool = False
    debug_output_path: str = "./debug/temporal_consistency/"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.process_noise_position <= 0:
            raise ValueError("Process noise for position must be positive")
        
        if not 0 <= self.min_visibility <= 1:
            raise ValueError("Minimum visibility must be in [0, 1]")
        
        if self.prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive")
        
        return True


class ExtendedKalmanFilter:
    """Extended Kalman Filter for 6-DOF TV tracking."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize EKF with configuration."""
        self.config = config
        
        # State vector: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # Position (3) + Quaternion (4) + Linear velocity (3) + Angular velocity (3)
        self.state_dim = 13
        self.measurement_dim = 7  # Position (3) + Quaternion (4)
        
        # Initialize state and covariance
        self.state = torch.zeros(self.state_dim, dtype=torch.float32)
        self.covariance = torch.eye(self.state_dim, dtype=torch.float32) * 0.1
        
        # Process and measurement noise
        self.process_noise = self._create_process_noise_matrix()
        self.measurement_noise = self._create_measurement_noise_matrix()
        
        self.is_initialized = False
        self.last_update_time = 0.0
        
    def initialize(self, initial_pose: torch.Tensor, initial_covariance: Optional[torch.Tensor] = None):
        """Initialize filter with initial pose estimate."""
        # Set initial position and orientation
        self.state[:3] = initial_pose[:3]  # Position
        self.state[3:7] = initial_pose[3:7] if initial_pose.shape[0] >= 7 else torch.tensor([1, 0, 0, 0])  # Quaternion
        
        # Initialize velocities to zero
        self.state[7:] = 0.0
        
        # Set initial covariance
        if initial_covariance is not None:
            self.covariance = initial_covariance
        else:
            self.covariance = torch.eye(self.state_dim) * 0.1
            
        self.is_initialized = True
        self.last_update_time = time.time()
        
    def predict(self, dt: float):
        """Predict next state using motion model."""
        if not self.is_initialized:
            return
        
        # State transition function f(x)
        predicted_state = self._state_transition(self.state, dt)
        
        # Jacobian of state transition
        F = self._compute_state_jacobian(self.state, dt)
        
        # Predict covariance
        predicted_covariance = torch.mm(torch.mm(F, self.covariance), F.T) + self.process_noise * dt
        
        self.state = predicted_state
        self.covariance = predicted_covariance
    
    def update(self, measurement: torch.Tensor, measurement_covariance: Optional[torch.Tensor] = None):
        """Update state with new measurement."""
        if not self.is_initialized:
            self.initialize(measurement)
            return
        
        # Measurement function h(x)
        predicted_measurement = self._measurement_function(self.state)
        
        # Measurement residual
        residual = measurement - predicted_measurement
        
        # Handle quaternion residual (ensure shortest rotation)
        if len(residual) >= 7:
            quat_residual = residual[3:7]
            # Normalize quaternion residual
            residual[3:7] = self._normalize_quaternion_residual(quat_residual)
        
        # Jacobian of measurement function
        H = self._compute_measurement_jacobian(self.state)
        
        # Innovation covariance
        R = measurement_covariance if measurement_covariance is not None else self.measurement_noise
        S = torch.mm(torch.mm(H, self.covariance), H.T) + R
        
        # Kalman gain
        K = torch.mm(torch.mm(self.covariance, H.T), torch.inverse(S))
        
        # Update state and covariance
        self.state = self.state + torch.mv(K, residual)
        I_KH = torch.eye(self.state_dim) - torch.mm(K, H)
        self.covariance = torch.mm(I_KH, self.covariance)
        
        # Normalize quaternion
        self.state[3:7] = self._normalize_quaternion(self.state[3:7])
    
    def get_pose(self) -> torch.Tensor:
        """Get current pose estimate."""
        return self.state[:7]  # Position + quaternion
    
    def get_velocity(self) -> torch.Tensor:
        """Get current velocity estimate."""
        return self.state[7:]  # Linear and angular velocity
    
    def get_uncertainty(self) -> torch.Tensor:
        """Get current pose uncertainty."""
        return torch.diag(self.covariance[:7])
    
    def _state_transition(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """State transition function for 6-DOF motion."""
        new_state = state.clone()
        
        # Update position: p = p + v * dt
        new_state[:3] = state[:3] + state[7:10] * dt
        
        # Update orientation using quaternion integration
        angular_vel = state[10:13]
        if torch.norm(angular_vel) > 1e-6:
            angle = torch.norm(angular_vel) * dt
            axis = angular_vel / torch.norm(angular_vel)
            
            # Convert axis-angle to quaternion
            sin_half = torch.sin(angle / 2)
            cos_half = torch.cos(angle / 2)
            delta_q = torch.cat([cos_half.unsqueeze(0), axis * sin_half])
            
            # Quaternion multiplication
            new_state[3:7] = self._quaternion_multiply(state[3:7], delta_q)
            new_state[3:7] = self._normalize_quaternion(new_state[3:7])
        
        # Velocities remain the same (constant velocity model)
        # In practice, you might add acceleration or damping
        
        return new_state
    
    def _measurement_function(self, state: torch.Tensor) -> torch.Tensor:
        """Measurement function (direct observation of pose)."""
        return state[:7]  # Directly observe position and orientation
    
    def _compute_state_jacobian(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute Jacobian of state transition function."""
        F = torch.eye(self.state_dim, dtype=torch.float32)
        
        # Position derivatives
        F[:3, 7:10] = torch.eye(3) * dt  # dp/dv
        
        # Quaternion derivatives (simplified)
        # In practice, this would be more complex
        angular_vel = state[10:13]
        if torch.norm(angular_vel) > 1e-6:
            # Approximate Jacobian for quaternion integration
            F[3:7, 10:13] = torch.eye(4, 3) * dt * 0.5
        
        return F
    
    def _compute_measurement_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of measurement function."""
        H = torch.zeros(self.measurement_dim, self.state_dim, dtype=torch.float32)
        H[:7, :7] = torch.eye(7)  # Direct observation
        return H
    
    def _create_process_noise_matrix(self) -> torch.Tensor:
        """Create process noise covariance matrix."""
        Q = torch.zeros(self.state_dim, self.state_dim, dtype=torch.float32)
        
        # Position noise
        Q[:3, :3] = torch.eye(3) * self.config.process_noise_position
        
        # Orientation noise
        Q[3:7, 3:7] = torch.eye(4) * self.config.process_noise_orientation
        
        # Velocity noise (higher for acceleration)
        Q[7:10, 7:10] = torch.eye(3) * self.config.process_noise_position * 2
        Q[10:13, 10:13] = torch.eye(3) * self.config.process_noise_orientation * 2
        
        return Q
    
    def _create_measurement_noise_matrix(self) -> torch.Tensor:
        """Create measurement noise covariance matrix."""
        R = torch.zeros(self.measurement_dim, self.measurement_dim, dtype=torch.float32)
        
        # Position measurement noise
        R[:3, :3] = torch.eye(3) * self.config.measurement_noise_position
        
        # Orientation measurement noise
        R[3:7, 3:7] = torch.eye(4) * self.config.measurement_noise_orientation
        
        return R
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return torch.tensor([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _normalize_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion to unit length."""
        return q / torch.norm(q)
    
    def _normalize_quaternion_residual(self, q_residual: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion residual for shortest rotation."""
        # Ensure shortest rotation path
        if q_residual[0] < 0:  # If w component is negative
            q_residual = -q_residual
        return q_residual


class OcclusionHandler:
    """Handles dynamic occlusion detection and mask generation."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize occlusion handler."""
        self.config = config
        self.occlusion_history = deque(maxlen=config.occlusion_buffer_frames)
        
    def generate_occlusion_mask(self, 
                               objects: List[Any],  # Object tracks
                               tv_placement: TVPlacement,
                               depth_map: torch.Tensor,
                               camera_params: CameraParameters) -> torch.Tensor:
        """
        Generate dynamic occlusion mask for TV placement.
        
        Args:
            objects: List of tracked objects
            tv_placement: Current TV placement
            depth_map: Scene depth map
            camera_params: Camera parameters
            
        Returns:
            Occlusion mask (H, W) with 1.0 = visible, 0.0 = occluded
        """
        height, width = depth_map.shape
        occlusion_mask = torch.ones((height, width), dtype=torch.float32)
        
        # Get TV corners in image space
        tv_corners_3d = tv_placement.geometry.get_corners_3d()
        tv_corners_2d = self._project_to_image(tv_corners_3d, camera_params)
        
        # Create TV region mask
        tv_mask = self._create_polygon_mask(tv_corners_2d, height, width)
        
        # Get TV depth from placement
        tv_depth = tv_placement.geometry.position[2].item()
        
        # Check occlusion from depth map
        depth_occlusion = self._compute_depth_occlusion(
            depth_map, tv_mask, tv_depth
        )
        
        # Check occlusion from tracked objects
        object_occlusion = self._compute_object_occlusion(
            objects, tv_corners_3d, tv_corners_2d, height, width
        )
        
        # Combine occlusion sources
        combined_occlusion = torch.maximum(depth_occlusion, object_occlusion)
        
        # Apply temporal smoothing
        smoothed_occlusion = self._apply_temporal_smoothing(combined_occlusion)
        
        # Final occlusion mask (1.0 = visible, 0.0 = occluded)
        occlusion_mask = tv_mask * (1.0 - smoothed_occlusion)
        
        # Store in history for temporal smoothing
        self.occlusion_history.append(occlusion_mask.clone())
        
        return occlusion_mask
    
    def _project_to_image(self, points_3d: torch.Tensor, 
                         camera_params: CameraParameters) -> torch.Tensor:
        """Project 3D points to image coordinates."""
        # Simplified projection (assuming no distortion)
        fx, fy = camera_params.fx, camera_params.fy
        cx, cy = camera_params.cx, camera_params.cy
        
        # Project to image plane
        x_img = (points_3d[:, 0] * fx / points_3d[:, 2]) + cx
        y_img = (points_3d[:, 1] * fy / points_3d[:, 2]) + cy
        
        return torch.stack([x_img, y_img], dim=1)
    
    def _create_polygon_mask(self, corners_2d: torch.Tensor, 
                           height: int, width: int) -> torch.Tensor:
        """Create binary mask from polygon corners."""
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        # Convert to numpy for cv2
        corners_np = corners_2d.cpu().numpy().astype(np.int32)
        mask_np = mask.cpu().numpy()
        
        # Fill polygon
        cv2.fillPoly(mask_np, [corners_np], 1.0)
        
        return torch.from_numpy(mask_np)
    
    def _compute_depth_occlusion(self, depth_map: torch.Tensor, 
                                tv_mask: torch.Tensor, tv_depth: float) -> torch.Tensor:
        """Compute occlusion based on depth comparison."""
        # Objects closer than TV cause occlusion
        depth_diff = tv_depth - depth_map
        
        # Only consider areas near the TV
        relevant_area = tv_mask > 0.5
        
        # Occlusion where depth is significantly closer
        occlusion = (depth_diff > self.config.occlusion_threshold) & relevant_area
        
        return occlusion.float()
    
    def _compute_object_occlusion(self, objects: List[Any], 
                                 tv_corners_3d: torch.Tensor,
                                 tv_corners_2d: torch.Tensor,
                                 height: int, width: int) -> torch.Tensor:
        """Compute occlusion from tracked objects."""
        occlusion_mask = torch.zeros((height, width), dtype=torch.float32)
        
        # Get TV bounding box in image
        tv_bbox = self._get_bounding_box(tv_corners_2d)
        tv_depth = torch.mean(tv_corners_3d[:, 2]).item()
        
        for obj in objects:
            # Skip if object is behind TV
            if hasattr(obj, 'position') and obj.position[2] > tv_depth + 0.1:
                continue
            
            # Get object bounding box
            if hasattr(obj, 'bbox_2d'):
                obj_bbox = obj.bbox_2d
                
                # Check if object overlaps with TV
                if self._boxes_overlap(tv_bbox, obj_bbox):
                    # Create occlusion mask for this object
                    obj_mask = self._create_bbox_mask(obj_bbox, height, width)
                    occlusion_mask = torch.maximum(occlusion_mask, obj_mask)
        
        return occlusion_mask
    
    def _get_bounding_box(self, corners_2d: torch.Tensor) -> torch.Tensor:
        """Get axis-aligned bounding box from corners."""
        min_coords = torch.min(corners_2d, dim=0)[0]
        max_coords = torch.max(corners_2d, dim=0)[0]
        
        return torch.tensor([min_coords[0], min_coords[1], max_coords[0], max_coords[1]])
    
    def _boxes_overlap(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
    
    def _create_bbox_mask(self, bbox: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Create mask from bounding box."""
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        x_min, y_min, x_max, y_max = bbox.int()
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        mask[y_min:y_max, x_min:x_max] = 1.0
        
        return mask
    
    def _apply_temporal_smoothing(self, current_occlusion: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to occlusion mask."""
        if len(self.occlusion_history) == 0:
            return current_occlusion
        
        # Exponential moving average
        alpha = 0.7
        smoothed = current_occlusion.clone()
        
        for i, hist_mask in enumerate(reversed(self.occlusion_history)):
            weight = alpha ** (i + 1)
            smoothed = smoothed * (1 - weight) + hist_mask * weight
        
        return smoothed


class LightingConsistencyManager:
    """Manages temporal lighting consistency for TV placement."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize lighting consistency manager."""
        self.config = config
        self.lighting_history = deque(maxlen=config.lighting_adaptation_frames)
        self.reference_lighting = None
        
    def smooth_lighting_transitions(self, 
                                   lighting_sequence: List[LightingEnvironment],
                                   confidence_scores: List[float]) -> List[LightingEnvironment]:
        """
        Apply temporal smoothing to lighting sequence.
        
        Args:
            lighting_sequence: Sequence of lighting environments
            confidence_scores: Confidence scores for each lighting estimate
            
        Returns:
            Smoothed lighting sequence
        """
        if not lighting_sequence:
            return []
        
        smoothed_sequence = []
        
        for i, (lighting, confidence) in enumerate(zip(lighting_sequence, confidence_scores)):
            if i == 0:
                # First frame - initialize reference
                self.reference_lighting = lighting
                smoothed_lighting = lighting
            else:
                # Apply temporal smoothing
                smoothed_lighting = self._apply_lighting_smoothing(
                    lighting, confidence, smoothed_sequence[-1] if smoothed_sequence else lighting
                )
            
            # Update lighting history
            self.lighting_history.append(smoothed_lighting)
            
            # Validate lighting consistency
            consistency_score = self._assess_lighting_consistency(smoothed_lighting)
            smoothed_lighting.stability_score = consistency_score
            
            smoothed_sequence.append(smoothed_lighting)
        
        return smoothed_sequence
    
    def _apply_lighting_smoothing(self, current_lighting: LightingEnvironment,
                                 confidence: float,
                                 previous_lighting: LightingEnvironment) -> LightingEnvironment:
        """Apply smoothing between current and previous lighting."""
        alpha = self.config.lighting_smoothing_alpha * confidence
        
        smoothed = LightingEnvironment()
        
        # Smooth color components
        smoothed.ambient_color = self._smooth_vector(
            current_lighting.ambient_color, previous_lighting.ambient_color, alpha
        )
        smoothed.light_color = self._smooth_vector(
            current_lighting.light_color, previous_lighting.light_color, alpha
        )
        smoothed.light_direction = self._smooth_vector(
            current_lighting.light_direction, previous_lighting.light_direction, alpha
        )
        
        # Smooth scalar values
        smoothed.ambient_intensity = self._smooth_scalar(
            current_lighting.ambient_intensity, previous_lighting.ambient_intensity, alpha
        )
        smoothed.light_intensity = self._smooth_scalar(
            current_lighting.light_intensity, previous_lighting.light_intensity, alpha
        )
        smoothed.shadow_intensity = self._smooth_scalar(
            current_lighting.shadow_intensity, previous_lighting.shadow_intensity, alpha
        )
        smoothed.scene_brightness = self._smooth_scalar(
            current_lighting.scene_brightness, previous_lighting.scene_brightness, alpha
        )
        smoothed.color_temperature = self._smooth_scalar(
            current_lighting.color_temperature, previous_lighting.color_temperature, alpha
        )
        
        # Maintain confidence and other metadata
        smoothed.confidence = confidence
        
        return smoothed
    
    def _smooth_vector(self, current: torch.Tensor, previous: torch.Tensor, alpha: float) -> torch.Tensor:
        """Smooth vector values."""
        return alpha * current + (1 - alpha) * previous
    
    def _smooth_scalar(self, current: float, previous: float, alpha: float) -> float:
        """Smooth scalar values."""
        return alpha * current + (1 - alpha) * previous
    
    def _assess_lighting_consistency(self, lighting: LightingEnvironment) -> float:
        """Assess lighting consistency over time."""
        if len(self.lighting_history) < 2:
            return 1.0
        
        # Compare with recent history
        recent_lighting = list(self.lighting_history)[-5:]  # Last 5 frames
        
        # Compute color consistency
        color_consistency = self._compute_color_consistency(lighting, recent_lighting)
        
        # Compute intensity consistency
        intensity_consistency = self._compute_intensity_consistency(lighting, recent_lighting)
        
        # Compute direction consistency
        direction_consistency = self._compute_direction_consistency(lighting, recent_lighting)
        
        # Overall consistency
        overall_consistency = (
            color_consistency * 0.4 +
            intensity_consistency * 0.3 +
            direction_consistency * 0.3
        )
        
        return overall_consistency
    
    def _compute_color_consistency(self, current: LightingEnvironment, 
                                  history: List[LightingEnvironment]) -> float:
        """Compute color consistency score."""
        color_diffs = []
        
        for hist_lighting in history:
            # Ambient color difference
            ambient_diff = torch.norm(current.ambient_color - hist_lighting.ambient_color).item()
            
            # Light color difference
            light_diff = torch.norm(current.light_color - hist_lighting.light_color).item()
            
            color_diffs.append(ambient_diff + light_diff)
        
        # Convert to consistency score (lower difference = higher consistency)
        avg_diff = np.mean(color_diffs) if color_diffs else 0.0
        consistency = max(0.0, 1.0 - avg_diff / self.config.color_change_threshold)
        
        return consistency
    
    def _compute_intensity_consistency(self, current: LightingEnvironment,
                                     history: List[LightingEnvironment]) -> float:
        """Compute intensity consistency score."""
        intensity_diffs = []
        
        for hist_lighting in history:
            ambient_diff = abs(current.ambient_intensity - hist_lighting.ambient_intensity)
            light_diff = abs(current.light_intensity - hist_lighting.light_intensity)
            
            intensity_diffs.append(ambient_diff + light_diff)
        
        avg_diff = np.mean(intensity_diffs) if intensity_diffs else 0.0
        consistency = max(0.0, 1.0 - avg_diff / 0.5)  # Threshold of 0.5 for intensity changes
        
        return consistency
    
    def _compute_direction_consistency(self, current: LightingEnvironment,
                                     history: List[LightingEnvironment]) -> float:
        """Compute light direction consistency score."""
        direction_diffs = []
        
        for hist_lighting in history:
            # Angle between light directions
            cos_angle = torch.dot(
                F.normalize(current.light_direction, dim=0),
                F.normalize(hist_lighting.light_direction, dim=0)
            ).item()
            
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            direction_diffs.append(angle)
        
        avg_angle = np.mean(direction_diffs) if direction_diffs else 0.0
        consistency = max(0.0, 1.0 - avg_angle / (np.pi / 4))  # Threshold of 45 degrees
        
        return consistency 


class QualityAssessment:
    """Comprehensive quality assessment for temporal consistency."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize quality assessment system."""
        self.config = config
        self.metrics_history = deque(maxlen=config.quality_window_size)
        self.artifact_detector = ArtifactDetector(config)
        
    def assess_quality(self, rendered_sequence: List[torch.Tensor],
                      tv_placements: List[TVPlacement],
                      depth_sequences: List[torch.Tensor]) -> QualityMetrics:
        """
        Assess overall quality of temporal consistency.
        
        Args:
            rendered_sequence: Sequence of rendered frames
            tv_placements: Corresponding TV placements
            depth_sequences: Depth map sequences
            
        Returns:
            Comprehensive quality metrics
        """
        metrics = QualityMetrics()
        
        if len(rendered_sequence) < 2:
            return metrics
        
        # Temporal stability assessment
        metrics.temporal_stability = self._assess_temporal_stability(tv_placements)
        
        # Motion smoothness
        metrics.motion_smoothness = self._assess_motion_smoothness(tv_placements)
        
        # Tracking reliability
        metrics.tracking_reliability = self._assess_tracking_reliability(tv_placements)
        
        # Occlusion handling quality
        occlusion_metrics = self._assess_occlusion_quality(tv_placements, depth_sequences)
        metrics.occlusion_accuracy = occlusion_metrics['accuracy']
        metrics.occlusion_completeness = occlusion_metrics['completeness']
        metrics.occlusion_consistency = occlusion_metrics['consistency']
        
        # Lighting consistency
        lighting_metrics = self._assess_lighting_quality(tv_placements)
        metrics.lighting_consistency = lighting_metrics['consistency']
        metrics.shadow_consistency = lighting_metrics['shadow_consistency']
        metrics.color_consistency = lighting_metrics['color_consistency']
        
        # Visual quality assessment
        visual_metrics = self._assess_visual_quality(rendered_sequence)
        metrics.visual_stability = visual_metrics['stability']
        metrics.artifact_score = visual_metrics['artifacts']
        metrics.realism_score = visual_metrics['realism']
        
        # Compute overall quality
        metrics.compute_overall_quality()
        
        # Store metrics for temporal analysis
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _assess_temporal_stability(self, tv_placements: List[TVPlacement]) -> float:
        """Assess temporal stability of TV placements."""
        if len(tv_placements) < 2:
            return 1.0
        
        position_variations = []
        orientation_variations = []
        
        for i in range(1, len(tv_placements)):
            prev_pos = tv_placements[i-1].geometry.position
            curr_pos = tv_placements[i].geometry.position
            
            # Position variation
            pos_diff = torch.norm(curr_pos - prev_pos).item()
            position_variations.append(pos_diff)
            
            # Orientation variation (simplified)
            prev_orient = tv_placements[i-1].geometry.orientation
            curr_orient = tv_placements[i].geometry.orientation
            
            if prev_orient.shape == curr_orient.shape:
                if prev_orient.shape == (4,):  # Quaternion
                    orient_diff = 1 - abs(torch.dot(prev_orient, curr_orient).item())
                else:  # Rotation matrix
                    orient_diff = torch.norm(prev_orient - curr_orient).item()
                
                orientation_variations.append(orient_diff)
        
        # Stability is inversely related to variation
        pos_stability = max(0.0, 1.0 - np.mean(position_variations) / 0.1)  # 10cm threshold
        orient_stability = max(0.0, 1.0 - np.mean(orientation_variations) / 0.2)  # Arbitrary threshold
        
        return (pos_stability + orient_stability) / 2
    
    def _assess_motion_smoothness(self, tv_placements: List[TVPlacement]) -> float:
        """Assess smoothness of TV motion."""
        if len(tv_placements) < 3:
            return 1.0
        
        accelerations = []
        
        for i in range(2, len(tv_placements)):
            # Compute acceleration as second derivative of position
            p0 = tv_placements[i-2].geometry.position
            p1 = tv_placements[i-1].geometry.position
            p2 = tv_placements[i].geometry.position
            
            # Approximate acceleration
            v1 = p1 - p0
            v2 = p2 - p1
            acceleration = torch.norm(v2 - v1).item()
            
            accelerations.append(acceleration)
        
        # Smoothness is inversely related to acceleration variation
        if accelerations:
            avg_acceleration = np.mean(accelerations)
            smoothness = max(0.0, 1.0 - avg_acceleration / 0.05)  # 5cm/frame^2 threshold
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _assess_tracking_reliability(self, tv_placements: List[TVPlacement]) -> float:
        """Assess reliability of tracking system."""
        if not tv_placements:
            return 0.0
        
        # Count tracking states
        state_counts = defaultdict(int)
        confidence_scores = []
        
        for placement in tv_placements:
            state_counts[placement.track_state] += 1
            confidence_scores.append(placement.confidence)
        
        total_frames = len(tv_placements)
        tracking_ratio = state_counts[TVPlacementState.TRACKING] / total_frames
        avg_confidence = np.mean(confidence_scores)
        
        # Reliability based on tracking ratio and confidence
        reliability = (tracking_ratio * 0.7 + avg_confidence * 0.3)
        
        return reliability
    
    def _assess_occlusion_quality(self, tv_placements: List[TVPlacement], 
                                 depth_sequences: List[torch.Tensor]) -> Dict[str, float]:
        """Assess quality of occlusion handling."""
        metrics = {
            'accuracy': 0.0,
            'completeness': 0.0,
            'consistency': 0.0
        }
        
        if not tv_placements or not depth_sequences:
            return metrics
        
        occlusion_masks = [p.occlusion_mask for p in tv_placements if p.occlusion_mask is not None]
        
        if len(occlusion_masks) < 2:
            return metrics
        
        # Consistency across frames
        consistency_scores = []
        for i in range(1, len(occlusion_masks)):
            prev_mask = occlusion_masks[i-1]
            curr_mask = occlusion_masks[i]
            
            # IoU-based consistency
            intersection = torch.sum(prev_mask * curr_mask)
            union = torch.sum(torch.maximum(prev_mask, curr_mask))
            
            if union > 0:
                iou = intersection / union
                consistency_scores.append(iou.item())
        
        metrics['consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Accuracy and completeness would require ground truth
        # For now, use heuristics based on visibility scores
        visibility_scores = [p.visibility for p in tv_placements]
        metrics['accuracy'] = np.mean(visibility_scores)
        metrics['completeness'] = min(1.0, np.std(visibility_scores) + 0.5)  # Lower std = better completeness
        
        return metrics
    
    def _assess_lighting_quality(self, tv_placements: List[TVPlacement]) -> Dict[str, float]:
        """Assess lighting consistency quality."""
        metrics = {
            'consistency': 0.0,
            'shadow_consistency': 0.0,
            'color_consistency': 0.0
        }
        
        if len(tv_placements) < 2:
            return metrics
        
        lighting_envs = [p.lighting for p in tv_placements]
        
        # Color consistency
        color_diffs = []
        shadow_diffs = []
        intensity_diffs = []
        
        for i in range(1, len(lighting_envs)):
            prev_lighting = lighting_envs[i-1]
            curr_lighting = lighting_envs[i]
            
            # Ambient color difference
            color_diff = torch.norm(prev_lighting.ambient_color - curr_lighting.ambient_color).item()
            color_diffs.append(color_diff)
            
            # Shadow consistency
            shadow_diff = abs(prev_lighting.shadow_intensity - curr_lighting.shadow_intensity)
            shadow_diffs.append(shadow_diff)
            
            # Light intensity
            intensity_diff = abs(prev_lighting.light_intensity - curr_lighting.light_intensity)
            intensity_diffs.append(intensity_diff)
        
        metrics['color_consistency'] = max(0.0, 1.0 - np.mean(color_diffs) / 0.2)
        metrics['shadow_consistency'] = max(0.0, 1.0 - np.mean(shadow_diffs) / 0.3)
        metrics['consistency'] = (metrics['color_consistency'] + metrics['shadow_consistency']) / 2
        
        return metrics
    
    def _assess_visual_quality(self, rendered_sequence: List[torch.Tensor]) -> Dict[str, float]:
        """Assess visual quality of rendered sequence."""
        metrics = {
            'stability': 0.0,
            'artifacts': 0.0,
            'realism': 0.0
        }
        
        if len(rendered_sequence) < 2:
            return metrics
        
        # Visual stability (frame-to-frame differences)
        frame_diffs = []
        for i in range(1, len(rendered_sequence)):
            prev_frame = rendered_sequence[i-1]
            curr_frame = rendered_sequence[i]
            
            # Compute perceptual difference
            diff = torch.mean(torch.abs(curr_frame - prev_frame)).item()
            frame_diffs.append(diff)
        
        # Stability inversely related to frame differences
        avg_diff = np.mean(frame_diffs)
        metrics['stability'] = max(0.0, 1.0 - avg_diff / 0.1)  # Threshold of 0.1
        
        # Artifact detection
        artifact_scores = []
        for frame in rendered_sequence:
            artifact_score = self.artifact_detector.detect_artifacts(frame)
            artifact_scores.append(artifact_score)
        
        metrics['artifacts'] = 1.0 - np.mean(artifact_scores)  # Lower artifacts = higher score
        
        # Realism assessment (simplified)
        # In practice, this might use a trained neural network
        metrics['realism'] = self._assess_realism(rendered_sequence)
        
        return metrics
    
    def _assess_realism(self, rendered_sequence: List[torch.Tensor]) -> float:
        """Assess realism of rendered sequence."""
        # Simplified realism assessment
        # In practice, this would use more sophisticated metrics
        
        realism_scores = []
        
        for frame in rendered_sequence:
            # Check for common unrealistic artifacts
            
            # 1. Color saturation check
            mean_saturation = torch.mean(torch.std(frame, dim=-1))
            saturation_score = torch.clamp(mean_saturation / 0.3, 0, 1).item()
            
            # 2. Brightness distribution
            brightness = torch.mean(frame)
            brightness_score = 1.0 - abs(brightness.item() - 0.5) * 2
            
            # 3. Edge sharpness (simplified)
            # Compute edge gradients separately to avoid dimension mismatch
            edge_grad_x = torch.abs(frame[:, 1:] - frame[:, :-1])  # Shape: (H, W-1, C)
            edge_grad_y = torch.abs(frame[1:, :] - frame[:-1, :])  # Shape: (H-1, W, C)
            
            # Compute mean edge strength separately
            edge_strength_x = torch.mean(edge_grad_x)
            edge_strength_y = torch.mean(edge_grad_y)
            
            # Average edge strength
            edge_strength = (edge_strength_x + edge_strength_y) / 2
            edge_score = torch.clamp(edge_strength / 0.1, 0, 1).item()
            
            # Combined realism score
            frame_realism = (saturation_score + brightness_score + edge_score) / 3
            realism_scores.append(frame_realism)
        
        return np.mean(realism_scores)


class ArtifactDetector:
    """Detects visual artifacts in rendered frames."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize artifact detector."""
        self.config = config
        
    def detect_artifacts(self, frame: torch.Tensor) -> float:
        """
        Detect artifacts in a single frame.
        
        Args:
            frame: Input frame tensor (H, W, C)
            
        Returns:
            Artifact score (0.0 = no artifacts, 1.0 = severe artifacts)
        """
        artifact_scores = []
        
        # 1. Detect temporal flicker
        flicker_score = self._detect_flicker(frame)
        artifact_scores.append(flicker_score)
        
        # 2. Detect geometry inconsistencies
        geometry_score = self._detect_geometry_artifacts(frame)
        artifact_scores.append(geometry_score)
        
        # 3. Detect lighting artifacts
        lighting_score = self._detect_lighting_artifacts(frame)
        artifact_scores.append(lighting_score)
        
        # 4. Detect occlusion artifacts
        occlusion_score = self._detect_occlusion_artifacts(frame)
        artifact_scores.append(occlusion_score)
        
        # Combine artifact scores
        overall_score = np.mean(artifact_scores)
        
        return overall_score
    
    def _detect_flicker(self, frame: torch.Tensor) -> float:
        """Detect temporal flicker artifacts."""
        # This would typically compare with previous frames
        # For now, detect high-frequency noise as potential flicker
        
        # Simplified approach: compute gradients to detect high-frequency content
        if len(frame.shape) == 3:
            # Convert to grayscale for simpler processing
            gray = torch.mean(frame, dim=-1)
        else:
            gray = frame
        
        # Compute gradients in x and y directions
        grad_x = torch.abs(gray[:, 1:] - gray[:, :-1])
        grad_y = torch.abs(gray[1:, :] - gray[:-1, :])
        
        # Compute local variance (high-frequency content indicator)
        variance_x = torch.var(grad_x)
        variance_y = torch.var(grad_y)
        
        # Average variance as flicker indicator
        avg_variance = (variance_x + variance_y) / 2
        
        # Normalize flicker score
        flicker_score = torch.clamp(avg_variance / self.config.artifact_detection_threshold, 0, 1).item()
        
        return flicker_score
    
    def _detect_geometry_artifacts(self, frame: torch.Tensor) -> float:
        """Detect geometry-related artifacts."""
        # Look for geometric inconsistencies like impossible perspectives
        
        # Simplified: check for extreme gradients that might indicate geometric errors
        grad_x = torch.abs(frame[:, 1:] - frame[:, :-1])
        grad_y = torch.abs(frame[1:, :] - frame[:-1, :])
        
        max_grad_x = torch.max(grad_x)
        max_grad_y = torch.max(grad_y)
        
        max_gradient = torch.max(max_grad_x, max_grad_y)
        
        # Extreme gradients might indicate geometry artifacts
        geometry_score = torch.clamp(max_gradient / 0.5, 0, 1).item()
        
        return geometry_score
    
    def _detect_lighting_artifacts(self, frame: torch.Tensor) -> float:
        """Detect lighting-related artifacts."""
        # Check for lighting inconsistencies
        
        # 1. Check for overexposure/underexposure
        overexposed = torch.mean((frame > 0.95).float())
        underexposed = torch.mean((frame < 0.05).float())
        
        exposure_score = (overexposed + underexposed).item()
        
        # 2. Check for unrealistic lighting gradients
        brightness = torch.mean(frame, dim=-1)
        
        # Compute gradients separately to avoid dimension mismatch
        grad_x = torch.abs(brightness[:, 1:] - brightness[:, :-1])  # Shape: (H, W-1)
        grad_y = torch.abs(brightness[1:, :] - brightness[:-1, :])  # Shape: (H-1, W)
        
        # Compute extreme gradient statistics separately
        extreme_grad_x = torch.mean((grad_x > 0.3).float()).item()
        extreme_grad_y = torch.mean((grad_y > 0.3).float()).item()
        
        # Average the gradient statistics
        extreme_gradients = (extreme_grad_x + extreme_grad_y) / 2
        
        lighting_score = (exposure_score + extreme_gradients) / 2
        
        return lighting_score
    
    def _detect_occlusion_artifacts(self, frame: torch.Tensor) -> float:
        """Detect occlusion-related artifacts."""
        # Look for abrupt changes that might indicate occlusion artifacts
        
        # Detect sharp boundaries that might be unnatural
        edges = self._compute_edges(frame)
        
        # Very sharp edges might indicate occlusion artifacts
        sharp_edges = torch.mean((edges > 0.4).float()).item()
        
        return sharp_edges
    
    def _compute_edges(self, frame: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel operators."""
        if len(frame.shape) == 3:
            # Convert to grayscale
            gray = torch.mean(frame, dim=-1)
        else:
            gray = frame
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Pad for convolution - use constant padding (mode='constant' is always supported)
        padded = F.pad(gray, (1, 1, 1, 1), mode='constant', value=0)
        
        # Apply Sobel filters
        grad_x = F.conv2d(padded.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))
        grad_y = F.conv2d(padded.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))
        
        # Compute magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        return magnitude


class PredictionInterpolation:
    """Handles prediction and interpolation for missing or occluded frames."""
    
    def __init__(self, config: TemporalConfig):
        """Initialize prediction and interpolation system."""
        self.config = config
        
    def predict_tv_placement(self, 
                           tv_history: List[TVPlacement],
                           frames_ahead: int = 1) -> TVPlacement:
        """
        Predict TV placement for future frames.
        
        Args:
            tv_history: Historical TV placements
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted TV placement
        """
        if not tv_history:
            raise ValueError("Need at least one historical placement for prediction")
        
        if len(tv_history) == 1:
            # No motion history, return same placement
            predicted = tv_history[-1]
            predicted.is_interpolated = True
            predicted.prediction_confidence = 0.5
            return predicted
        
        # Use linear prediction based on recent motion
        recent_placements = tv_history[-min(3, len(tv_history)):]
        
        # Predict position
        predicted_position = self._predict_position(recent_placements, frames_ahead)
        
        # Predict orientation
        predicted_orientation = self._predict_orientation(recent_placements, frames_ahead)
        
        # Create predicted geometry
        predicted_geometry = TVGeometry(
            position=predicted_position,
            orientation=predicted_orientation,
            width=recent_placements[-1].geometry.width,
            height=recent_placements[-1].geometry.height,
            depth=recent_placements[-1].geometry.depth
        )
        
        # Create predicted placement
        predicted_placement = TVPlacement(
            geometry=predicted_geometry,
            confidence=max(0.3, recent_placements[-1].confidence - 0.1 * frames_ahead),
            visibility=recent_placements[-1].visibility,
            occlusion_mask=recent_placements[-1].occlusion_mask,
            lighting=recent_placements[-1].lighting,
            is_interpolated=True,
            prediction_confidence=max(0.1, 1.0 - 0.2 * frames_ahead)
        )
        
        return predicted_placement
    
    def interpolate_tv_placements(self,
                                start_placement: TVPlacement,
                                end_placement: TVPlacement,
                                num_intermediate: int) -> List[TVPlacement]:
        """
        Interpolate TV placements between two keyframes.
        
        Args:
            start_placement: Starting placement
            end_placement: Ending placement
            num_intermediate: Number of intermediate frames to generate
            
        Returns:
            List of interpolated placements
        """
        if num_intermediate <= 0:
            return []
        
        interpolated_placements = []
        
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            
            # Interpolate position
            interp_position = (1 - alpha) * start_placement.geometry.position + \
                             alpha * end_placement.geometry.position
            
            # Interpolate orientation (SLERP for quaternions)
            interp_orientation = self._slerp_orientation(
                start_placement.geometry.orientation,
                end_placement.geometry.orientation,
                alpha
            )
            
            # Interpolate other properties
            interp_confidence = (1 - alpha) * start_placement.confidence + \
                               alpha * end_placement.confidence
            
            interp_visibility = (1 - alpha) * start_placement.visibility + \
                               alpha * end_placement.visibility
            
            # Interpolate lighting
            interp_lighting = self._interpolate_lighting(
                start_placement.lighting,
                end_placement.lighting,
                alpha
            )
            
            # Create interpolated geometry
            interp_geometry = TVGeometry(
                position=interp_position,
                orientation=interp_orientation,
                width=start_placement.geometry.width,
                height=start_placement.geometry.height,
                depth=start_placement.geometry.depth
            )
            
            # Create interpolated placement
            interp_placement = TVPlacement(
                geometry=interp_geometry,
                confidence=interp_confidence,
                visibility=interp_visibility,
                occlusion_mask=start_placement.occlusion_mask,  # Use start mask
                lighting=interp_lighting,
                is_interpolated=True,
                prediction_confidence=0.8  # High confidence for interpolation
            )
            
            interpolated_placements.append(interp_placement)
        
        return interpolated_placements
    
    def _predict_position(self, placements: List[TVPlacement], frames_ahead: int) -> torch.Tensor:
        """Predict position using linear extrapolation."""
        positions = [p.geometry.position for p in placements]
        
        if len(positions) >= 2:
            # Linear velocity
            velocity = positions[-1] - positions[-2]
            predicted_position = positions[-1] + velocity * frames_ahead
        else:
            predicted_position = positions[-1]
        
        return predicted_position
    
    def _predict_orientation(self, placements: List[TVPlacement], frames_ahead: int) -> torch.Tensor:
        """Predict orientation using angular velocity."""
        orientations = [p.geometry.orientation for p in placements]
        
        if len(orientations) >= 2:
            # Simplified angular prediction
            # In practice, this would use proper quaternion/rotation matrix math
            current_orient = orientations[-1]
            prev_orient = orientations[-2]
            
            if current_orient.shape == (4,):  # Quaternion
                # Simple SLERP-based prediction
                predicted_orient = self._slerp_orientation(prev_orient, current_orient, 1.0 + frames_ahead)
            else:  # Rotation matrix
                # Linear interpolation (not ideal for rotation matrices)
                delta = current_orient - prev_orient
                predicted_orient = current_orient + delta * frames_ahead
        else:
            predicted_orient = orientations[-1]
        
        return predicted_orient
    
    def _slerp_orientation(self, q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation for quaternions."""
        if q1.shape != (4,) or q2.shape != (4,):
            # Fallback for rotation matrices
            return (1 - t) * q1 + t * q2
        
        # Ensure shortest path
        dot = torch.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Clamp for numerical stability
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # SLERP formula
        if dot > 0.9995:
            # Linear interpolation for very close quaternions
            result = (1 - t) * q1 + t * q2
        else:
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            result = w1 * q1 + w2 * q2
        
        # Normalize
        return result / torch.norm(result)
    
    def _interpolate_lighting(self, 
                            lighting1: LightingEnvironment,
                            lighting2: LightingEnvironment,
                            alpha: float) -> LightingEnvironment:
        """Interpolate between two lighting environments."""
        interp_lighting = LightingEnvironment()
        
        # Interpolate color components
        interp_lighting.ambient_color = (1 - alpha) * lighting1.ambient_color + alpha * lighting2.ambient_color
        interp_lighting.light_color = (1 - alpha) * lighting1.light_color + alpha * lighting2.light_color
        interp_lighting.light_direction = (1 - alpha) * lighting1.light_direction + alpha * lighting2.light_direction
        
        # Interpolate scalar values
        interp_lighting.ambient_intensity = (1 - alpha) * lighting1.ambient_intensity + alpha * lighting2.ambient_intensity
        interp_lighting.light_intensity = (1 - alpha) * lighting1.light_intensity + alpha * lighting2.light_intensity
        interp_lighting.shadow_intensity = (1 - alpha) * lighting1.shadow_intensity + alpha * lighting2.shadow_intensity
        interp_lighting.scene_brightness = (1 - alpha) * lighting1.scene_brightness + alpha * lighting2.scene_brightness
        interp_lighting.color_temperature = (1 - alpha) * lighting1.color_temperature + alpha * lighting2.color_temperature
        
        # Interpolate confidence
        interp_lighting.confidence = (1 - alpha) * lighting1.confidence + alpha * lighting2.confidence
        
        return interp_lighting


class TemporalConsistencyManager:
    """Main temporal consistency management system for TV placement."""
    
    def __init__(self, config: TemporalConfig):
        """
        Initialize temporal consistency manager.
        
        Args:
            config: Temporal consistency configuration
        """
        self.config = config
        config.validate()
        
        # Core components
        self.kalman_filter = ExtendedKalmanFilter(config)
        self.occlusion_handler = OcclusionHandler(config)
        self.lighting_manager = LightingConsistencyManager(config)
        self.quality_assessor = QualityAssessment(config)
        self.predictor = PredictionInterpolation(config)
        
        # State management
        self.tv_placement_history = deque(maxlen=100)  # Keep last 100 placements
        self.current_frame_id = 0
        self.last_valid_placement = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=50)
        self.memory_usage = deque(maxlen=50)
        
        # Threading for real-time processing
        if config.enable_multithreading:
            self.processing_queue = queue.Queue(maxsize=config.processing_queue_size)
            self.result_queue = queue.Queue()
            self._start_processing_threads()
        
        logger.info("TemporalConsistencyManager initialized")
    
    def track_tv_placement(self,
                          tv_detections: List[Any],  # Raw TV detections
                          camera_poses: List[Any],   # Camera pose sequence
                          depth_map: torch.Tensor,
                          rgb_frame: torch.Tensor,
                          object_tracks: List[Any] = None) -> List[TVPlacement]:
        """
        Track TV placement with temporal consistency.
        
        Args:
            tv_detections: Raw TV detections from detection system
            camera_poses: Camera pose information
            depth_map: Current depth map
            rgb_frame: Current RGB frame
            object_tracks: Tracked objects for occlusion
            
        Returns:
            List of temporally consistent TV placements
        """
        start_time = time.time()
        
        try:
            # Convert detections to TVPlacement objects
            current_placements = self._convert_detections_to_placements(
                tv_detections, depth_map, rgb_frame
            )
            
            # Apply Kalman filtering for temporal consistency
            filtered_placements = self._apply_kalman_filtering(current_placements)
            
            # Handle occlusions
            self._update_occlusion_masks(filtered_placements, object_tracks or [], depth_map)
            
            # Update lighting consistency
            self._update_lighting_consistency(filtered_placements, rgb_frame)
            
            # Quality assessment
            self._assess_placement_quality(filtered_placements)
            
            # Handle prediction/interpolation if needed
            final_placements = self._handle_missing_detections(filtered_placements)
            
            # Update history
            self.tv_placement_history.extend(final_placements)
            self.current_frame_id += 1
            
            # Performance monitoring
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.debug(f"Processed frame {self.current_frame_id} in {processing_time:.3f}s")
            
            return final_placements
            
        except Exception as e:
            logger.error(f"Error in track_tv_placement: {e}")
            return []
    
    def generate_occlusion_masks(self,
                               objects: List[Any],
                               tv_placement: TVPlacement,
                               depth_map: torch.Tensor,
                               camera_params: CameraParameters) -> torch.Tensor:
        """
        Generate dynamic occlusion masks for TV placement.
        
        Args:
            objects: List of tracked objects
            tv_placement: Current TV placement
            depth_map: Scene depth map
            camera_params: Camera parameters
            
        Returns:
            Occlusion mask tensor
        """
        return self.occlusion_handler.generate_occlusion_mask(
            objects, tv_placement, depth_map, camera_params
        )
    
    def smooth_lighting_transitions(self,
                                   lighting_sequence: List[LightingEnvironment],
                                   confidence_scores: List[float]) -> List[LightingEnvironment]:
        """
        Apply temporal smoothing to lighting sequence.
        
        Args:
            lighting_sequence: Sequence of lighting environments
            confidence_scores: Confidence scores for each estimate
            
        Returns:
            Smoothed lighting sequence
        """
        return self.lighting_manager.smooth_lighting_transitions(
            lighting_sequence, confidence_scores
        )
    
    def assess_quality(self, 
                      rendered_sequence: List[torch.Tensor],
                      tv_placements: Optional[List[TVPlacement]] = None,
                      depth_sequences: Optional[List[torch.Tensor]] = None) -> QualityMetrics:
        """
        Assess quality of temporal consistency.
        
        Args:
            rendered_sequence: Sequence of rendered frames
            tv_placements: TV placement sequence (optional)
            depth_sequences: Depth sequence (optional)
            
        Returns:
            Quality assessment metrics
        """
        if tv_placements is None:
            tv_placements = list(self.tv_placement_history)
        
        return self.quality_assessor.assess_quality(
            rendered_sequence, tv_placements, depth_sequences or []
        )
    
    def predict_placement(self, frames_ahead: int = 1) -> Optional[TVPlacement]:
        """
        Predict TV placement for future frames.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted TV placement or None
        """
        if len(self.tv_placement_history) == 0:
            return None
        
        return self.predictor.predict_tv_placement(
            list(self.tv_placement_history), frames_ahead
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0.0,
            'frames_processed': self.current_frame_id,
            'placements_in_history': len(self.tv_placement_history),
            'memory_usage_mb': self._get_memory_usage()
        }
        
        return stats
    
    def reset(self):
        """Reset temporal consistency manager state."""
        self.tv_placement_history.clear()
        self.current_frame_id = 0
        self.last_valid_placement = None
        self.kalman_filter = ExtendedKalmanFilter(self.config)
        
        logger.info("TemporalConsistencyManager reset")
    
    def _convert_detections_to_placements(self,
                                        detections: List[Any],
                                        depth_map: torch.Tensor,
                                        rgb_frame: torch.Tensor) -> List[TVPlacement]:
        """Convert raw detections to TVPlacement objects."""
        placements = []
        
        for detection in detections:
            # Extract geometry from detection
            geometry = self._extract_geometry_from_detection(detection, depth_map)
            
            # Extract lighting from scene
            lighting = self._extract_lighting_from_scene(rgb_frame, geometry)
            
            # Create initial occlusion mask
            initial_mask = torch.ones((depth_map.shape[0], depth_map.shape[1]), dtype=torch.float32)
            
            # Create placement
            placement = TVPlacement(
                geometry=geometry,
                confidence=getattr(detection, 'confidence', 0.8),
                visibility=1.0,  # Will be updated by occlusion handler
                occlusion_mask=initial_mask,
                lighting=lighting,
                frame_id=self.current_frame_id
            )
            
            placements.append(placement)
        
        return placements
    
    def _extract_geometry_from_detection(self, detection: Any, depth_map: torch.Tensor) -> TVGeometry:
        """Extract geometry information from detection."""
        # This would depend on your detection format
        # For now, create a simple geometry
        
        position = getattr(detection, 'position', torch.tensor([0.0, 0.0, 2.0]))
        orientation = getattr(detection, 'orientation', torch.tensor([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion
        width = getattr(detection, 'width', 1.2)  # Typical TV width
        height = getattr(detection, 'height', 0.8)  # Typical TV height
        
        return TVGeometry(
            position=position,
            orientation=orientation,
            width=width,
            height=height
        )
    
    def _extract_lighting_from_scene(self, rgb_frame: torch.Tensor, geometry: TVGeometry) -> LightingEnvironment:
        """Extract lighting information from scene."""
        # Simplified lighting extraction
        avg_brightness = torch.mean(rgb_frame).item()
        
        lighting = LightingEnvironment(
            scene_brightness=avg_brightness,
            confidence=0.7
        )
        
        return lighting
    
    def _apply_kalman_filtering(self, placements: List[TVPlacement]) -> List[TVPlacement]:
        """Apply Kalman filtering for temporal consistency."""
        filtered_placements = []
        
        for placement in placements:
            # Convert placement to measurement vector
            measurement = torch.cat([
                placement.geometry.position,
                placement.geometry.orientation[:4] if placement.geometry.orientation.shape[0] >= 4 
                else torch.tensor([1.0, 0.0, 0.0, 0.0])
            ])
            
            # Update Kalman filter
            dt = 1.0 / 30.0  # Assume 30 FPS
            self.kalman_filter.predict(dt)
            self.kalman_filter.update(measurement)
            
            # Get filtered pose
            filtered_pose = self.kalman_filter.get_pose()
            
            # Update placement with filtered values
            placement.geometry.position = filtered_pose[:3]
            placement.geometry.orientation = filtered_pose[3:7]
            
            # Update confidence based on uncertainty
            uncertainty = self.kalman_filter.get_uncertainty()
            avg_uncertainty = torch.mean(uncertainty).item()
            placement.confidence *= max(0.3, 1.0 - avg_uncertainty)
            
            filtered_placements.append(placement)
        
        return filtered_placements
    
    def _update_occlusion_masks(self, placements: List[TVPlacement], 
                               objects: List[Any], depth_map: torch.Tensor):
        """Update occlusion masks for placements."""
        # Simplified camera parameters (would come from actual camera)
        camera_params = CameraParameters(fx=500, fy=500, cx=320, cy=240)
        
        for placement in placements:
            occlusion_mask = self.occlusion_handler.generate_occlusion_mask(
                objects, placement, depth_map, camera_params
            )
            
            placement.occlusion_mask = occlusion_mask
            placement.visibility = torch.mean(occlusion_mask).item()
    
    def _update_lighting_consistency(self, placements: List[TVPlacement], rgb_frame: torch.Tensor):
        """Update lighting consistency for placements."""
        if not placements:
            return
        
        # Extract lighting for each placement
        lighting_envs = [p.lighting for p in placements]
        confidence_scores = [p.confidence for p in placements]
        
        # Apply smoothing
        smoothed_lighting = self.lighting_manager.smooth_lighting_transitions(
            lighting_envs, confidence_scores
        )
        
        # Update placements with smoothed lighting
        for placement, lighting in zip(placements, smoothed_lighting):
            placement.lighting = lighting
            placement.lighting_consistency = lighting.stability_score
    
    def _assess_placement_quality(self, placements: List[TVPlacement]):
        """Assess quality of current placements."""
        for placement in placements:
            # Simple quality assessment based on confidence and visibility
            placement.temporal_stability = min(placement.confidence, placement.visibility)
            placement.spatial_consistency = placement.confidence  # Simplified
    
    def _handle_missing_detections(self, placements: List[TVPlacement]) -> List[TVPlacement]:
        """Handle missing detections through prediction/interpolation."""
        if not placements and self.last_valid_placement is not None:
            # No detections - predict from last valid placement
            predicted_placement = self.predictor.predict_tv_placement(
                [self.last_valid_placement], frames_ahead=1
            )
            placements = [predicted_placement]
        
        # Update last valid placement
        if placements:
            self.last_valid_placement = placements[-1]
        
        return placements
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _start_processing_threads(self):
        """Start background processing threads."""
        for _ in range(self.config.max_threads):
            thread = threading.Thread(target=self._processing_worker, daemon=True)
            thread.start()
    
    def _processing_worker(self):
        """Worker thread for background processing."""
        while True:
            try:
                task = self.processing_queue.get(timeout=1.0)
                if task is None:
                    break
                
                # Process task
                result = self._process_background_task(task)
                self.result_queue.put(result)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _process_background_task(self, task: Dict[str, Any]) -> Any:
        """Process background task."""
        task_type = task.get('type')
        
        if task_type == 'quality_assessment':
            return self.quality_assessor.assess_quality(
                task['rendered_sequence'],
                task.get('tv_placements'),
                task.get('depth_sequences')
            )
        elif task_type == 'prediction':
            return self.predictor.predict_tv_placement(
                task['tv_history'],
                task.get('frames_ahead', 1)
            )
        
        return None 