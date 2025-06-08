"""
Motion Models for Multi-Object Tracking

This module provides various motion models for tracking objects, including
Kalman filters for position and velocity estimation, and camera motion
compensation for handling camera movements.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False


class MotionModel(Enum):
    """Motion model types."""
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    ADAPTIVE = "adaptive"
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"


@dataclass
class MotionParameters:
    """Parameters for motion models."""
    
    # Process noise
    position_std: float = 1.0
    velocity_std: float = 10.0
    acceleration_std: float = 1.0
    
    # Measurement noise
    measurement_std: float = 1.0
    
    # Model-specific parameters
    max_velocity: float = 100.0
    max_acceleration: float = 50.0
    
    # Adaptive parameters
    adaptation_rate: float = 0.1
    velocity_threshold: float = 5.0


class BaseKalmanTracker(ABC):
    """Base class for Kalman filter-based trackers."""
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float],
                 parameters: MotionParameters):
        self.parameters = parameters
        self.initialized = False
        self.last_bbox = initial_bbox
        self.velocity_history = []
        
    @abstractmethod
    def predict(self) -> Tuple[float, float, float, float]:
        """Predict next position."""
        pass
    
    @abstractmethod
    def update(self, bbox: Tuple[float, float, float, float]):
        """Update with new measurement."""
        pass
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        pass
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        pass


class ConstantVelocityKalman(BaseKalmanTracker):
    """Kalman filter with constant velocity motion model."""
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float],
                 parameters: MotionParameters):
        super().__init__(initial_bbox, parameters)
        
        if not FILTERPY_AVAILABLE:
            # Fallback to simple tracking
            self.use_simple_tracker = True
            self.velocity = (0.0, 0.0)
            self.position = self._bbox_to_center(initial_bbox)
            return
        
        self.use_simple_tracker = False
        
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity)
        dt = 1.0  # Time step
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],   # x
            [0, 1, 0, 0, 0, dt, 0, 0],   # y
            [0, 0, 1, 0, 0, 0, dt, 0],   # w
            [0, 0, 0, 1, 0, 0, 0, dt],   # h
            [0, 0, 0, 0, 1, 0, 0, 0],    # vx
            [0, 0, 0, 0, 0, 1, 0, 0],    # vy
            [0, 0, 0, 0, 0, 0, 1, 0],    # vw
            [0, 0, 0, 0, 0, 0, 0, 1],    # vh
        ])
        
        # Measurement matrix (observe position and size)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.kf.Q = Q_discrete_white_noise(
            dim=2, dt=dt, var=parameters.velocity_std**2, block_size=4
        )
        
        # Measurement noise covariance
        self.kf.R *= parameters.measurement_std**2
        
        # Initial state covariance
        self.kf.P *= 1000.0
        
        # Initialize state
        x, y, w, h = self._bbox_to_xywh(initial_bbox)
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(-1, 1)
        
        self.initialized = True
    
    def predict(self) -> Tuple[float, float, float, float]:
        """Predict next position."""
        if self.use_simple_tracker:
            # Simple linear prediction
            new_x = self.position[0] + self.velocity[0]
            new_y = self.position[1] + self.velocity[1]
            self.position = (new_x, new_y)
            
            # Convert back to bbox (assuming constant size)
            x1, y1, x2, y2 = self.last_bbox
            w, h = x2 - x1, y2 - y1
            return (new_x - w/2, new_y - h/2, new_x + w/2, new_y + h/2)
        
        self.kf.predict()
        x, y, w, h = self.kf.x[:4].flatten()
        return self._xywh_to_bbox(x, y, w, h)
    
    def update(self, bbox: Tuple[float, float, float, float]):
        """Update with new measurement."""
        if self.use_simple_tracker:
            # Update velocity and position
            new_center = self._bbox_to_center(bbox)
            self.velocity = (
                new_center[0] - self.position[0],
                new_center[1] - self.position[1]
            )
            self.position = new_center
            self.last_bbox = bbox
            return
        
        x, y, w, h = self._bbox_to_xywh(bbox)
        measurement = np.array([x, y, w, h]).reshape(-1, 1)
        self.kf.update(measurement)
        self.last_bbox = bbox
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.use_simple_tracker:
            return np.array([*self.position, *self.velocity])
        return self.kf.x.flatten()
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        if self.use_simple_tracker:
            return np.eye(4)
        return self.kf.P
    
    def _bbox_to_xywh(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert bbox to center coordinates and size."""
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h
    
    def _xywh_to_bbox(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Convert center coordinates and size to bbox."""
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2
    
    def _bbox_to_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Extract center from bbox."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class AdaptiveKalmanTracker(BaseKalmanTracker):
    """Adaptive Kalman filter that adjusts parameters based on motion patterns."""
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float],
                 parameters: MotionParameters):
        super().__init__(initial_bbox, parameters)
        
        # Use constant velocity as base
        self.base_tracker = ConstantVelocityKalman(initial_bbox, parameters)
        
        # Adaptation state
        self.motion_pattern = "unknown"
        self.adaptation_counter = 0
        self.velocity_variance_history = []
        
    def predict(self) -> Tuple[float, float, float, float]:
        """Predict with adaptive parameters."""
        # Analyze motion pattern
        self._analyze_motion_pattern()
        
        # Adapt parameters if needed
        self._adapt_parameters()
        
        return self.base_tracker.predict()
    
    def update(self, bbox: Tuple[float, float, float, float]):
        """Update with adaptive learning."""
        self.base_tracker.update(bbox)
        
        # Update velocity history for adaptation
        current_center = self._bbox_to_center(bbox)
        if hasattr(self, 'last_center'):
            velocity = (
                current_center[0] - self.last_center[0],
                current_center[1] - self.last_center[1]
            )
            self.velocity_history.append(velocity)
            
            # Keep limited history
            if len(self.velocity_history) > 20:
                self.velocity_history.pop(0)
        
        self.last_center = current_center
        self.adaptation_counter += 1
    
    def _analyze_motion_pattern(self):
        """Analyze motion pattern from velocity history."""
        if len(self.velocity_history) < 5:
            return
        
        velocities = np.array(self.velocity_history[-10:])  # Last 10 frames
        
        # Compute velocity statistics
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        vel_mean = np.mean(vel_magnitudes)
        vel_std = np.std(vel_magnitudes)
        
        self.velocity_variance_history.append(vel_std)
        
        # Classify motion pattern
        if vel_mean < 2.0:
            self.motion_pattern = "stationary"
        elif vel_std < 1.0:
            self.motion_pattern = "linear"
        elif vel_std < 3.0:
            self.motion_pattern = "smooth"
        else:
            self.motion_pattern = "erratic"
    
    def _adapt_parameters(self):
        """Adapt Kalman filter parameters based on motion pattern."""
        if not self.base_tracker.use_simple_tracker and self.adaptation_counter % 10 == 0:
            # Adjust process noise based on motion pattern
            if self.motion_pattern == "stationary":
                # Reduce process noise for stationary objects
                self.base_tracker.kf.Q *= 0.5
            elif self.motion_pattern == "erratic":
                # Increase process noise for erratic motion
                self.base_tracker.kf.Q *= 1.5
            
            # Clamp values to reasonable ranges
            self.base_tracker.kf.Q = np.clip(self.base_tracker.kf.Q, 0.1, 100.0)
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.base_tracker.get_state()
    
    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.base_tracker.get_covariance()
    
    def _bbox_to_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Extract center from bbox."""
        return self.base_tracker._bbox_to_center(bbox)


class CameraMotionCompensator:
    """Compensate for camera motion in tracking."""
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.homography_buffer = []
        self.camera_velocity = (0.0, 0.0)
        
        # Feature detector for motion estimation
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def update_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Update with new frame and estimate camera motion.
        
        Args:
            frame: Current frame (grayscale)
            
        Returns:
            Homography matrix for camera motion compensation
        """
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) < 2:
            return None
        
        # Estimate motion between consecutive frames
        prev_frame = self.frame_buffer[-2]
        curr_frame = self.frame_buffer[-1]
        
        homography = self._estimate_homography(prev_frame, curr_frame)
        
        if homography is not None:
            self.homography_buffer.append(homography)
            
            if len(self.homography_buffer) > self.buffer_size:
                self.homography_buffer.pop(0)
            
            # Estimate camera velocity
            self._estimate_camera_velocity(homography)
        
        return homography
    
    def compensate_bbox(self, bbox: Tuple[float, float, float, float],
                       homography: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """
        Compensate bounding box for camera motion.
        
        Args:
            bbox: Original bounding box
            homography: Camera motion homography
            
        Returns:
            Compensated bounding box
        """
        if homography is None and self.homography_buffer:
            homography = self.homography_buffer[-1]
        
        if homography is None:
            return bbox
        
        try:
            # Convert bbox to points
            x1, y1, x2, y2 = bbox
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform points using homography
            transformed_corners = cv2.perspectiveTransform(corners, homography)
            transformed_corners = transformed_corners.reshape(-1, 2)
            
            # Get new bounding box
            min_x = np.min(transformed_corners[:, 0])
            min_y = np.min(transformed_corners[:, 1])
            max_x = np.max(transformed_corners[:, 0])
            max_y = np.max(transformed_corners[:, 1])
            
            return (min_x, min_y, max_x, max_y)
            
        except Exception:
            # Fallback to original bbox if transformation fails
            return bbox
    
    def get_camera_velocity(self) -> Tuple[float, float]:
        """Get estimated camera velocity."""
        return self.camera_velocity
    
    def _estimate_homography(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate homography between consecutive frames."""
        try:
            # Detect features
            kp1, des1 = self.feature_detector.detectAndCompute(prev_frame, None)
            kp2, des2 = self.feature_detector.detectAndCompute(curr_frame, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return None
            
            # Match features
            matches = self.matcher.match(des1, des2)
            
            if len(matches) < 10:
                return None
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography with RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                ransacReprojThreshold=3.0,
                maxIters=1000
            )
            
            # Validate homography
            if homography is not None and self._validate_homography(homography):
                return homography
            
        except Exception:
            pass
        
        return None
    
    def _validate_homography(self, homography: np.ndarray) -> bool:
        """Validate homography matrix."""
        if homography is None or homography.shape != (3, 3):
            return False
        
        # Check if homography is reasonable (not too distorted)
        try:
            # Test corners transformation
            corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], 
                              dtype=np.float32).reshape(-1, 1, 2)
            
            transformed = cv2.perspectiveTransform(corners, homography)
            
            # Check if transformation preserves approximate rectangle shape
            # (this is a simple validation - more sophisticated checks could be added)
            original_area = 100 * 100
            transformed_corners = transformed.reshape(-1, 2)
            transformed_area = cv2.contourArea(transformed_corners)
            
            # Area should not change dramatically
            area_ratio = transformed_area / original_area
            if area_ratio < 0.5 or area_ratio > 2.0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _estimate_camera_velocity(self, homography: np.ndarray):
        """Estimate camera velocity from homography."""
        try:
            # Extract translation from homography
            # This is a simplified approach - more sophisticated methods exist
            center_point = np.array([[320, 240]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_center = cv2.perspectiveTransform(center_point, homography)
            
            velocity_x = transformed_center[0, 0, 0] - center_point[0, 0, 0]
            velocity_y = transformed_center[0, 0, 1] - center_point[0, 0, 1]
            
            # Smooth velocity estimation
            alpha = 0.3  # Smoothing factor
            self.camera_velocity = (
                alpha * velocity_x + (1 - alpha) * self.camera_velocity[0],
                alpha * velocity_y + (1 - alpha) * self.camera_velocity[1]
            )
            
        except Exception:
            pass


class KalmanTracker:
    """Main Kalman tracker factory and manager."""
    
    def __init__(self, motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY,
                 parameters: Optional[MotionParameters] = None):
        self.motion_model = motion_model
        self.parameters = parameters or MotionParameters()
        self.camera_compensator = CameraMotionCompensator()
        
    def create_tracker(self, initial_bbox: Tuple[float, float, float, float]) -> BaseKalmanTracker:
        """Create appropriate tracker based on motion model."""
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            return ConstantVelocityKalman(initial_bbox, self.parameters)
        elif self.motion_model == MotionModel.ADAPTIVE:
            return AdaptiveKalmanTracker(initial_bbox, self.parameters)
        else:
            # Default to constant velocity
            return ConstantVelocityKalman(initial_bbox, self.parameters)
    
    def update_camera_motion(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Update camera motion estimation."""
        return self.camera_compensator.update_frame(frame)
    
    def compensate_camera_motion(self, bbox: Tuple[float, float, float, float],
                                homography: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """Compensate bounding box for camera motion."""
        return self.camera_compensator.compensate_bbox(bbox, homography)
    
    def get_camera_velocity(self) -> Tuple[float, float]:
        """Get estimated camera velocity."""
        return self.camera_compensator.get_camera_velocity() 