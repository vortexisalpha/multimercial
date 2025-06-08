"""
Temporal Consistency Module

Implements Kalman filtering and advanced stabilization techniques for
maintaining temporal consistency in video advertisement placement.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any, Sequence
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..utils.logging_utils import get_logger
from ..utils.kalman_utils import (
    create_constant_velocity_filter,
    create_constant_acceleration_filter,
    predict_state,
    update_state
)

logger = get_logger(__name__)


class FilterType(Enum):
    """Types of Kalman filters for different tracking scenarios."""
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    UNSCENTED = "unscented"
    EXTENDED = "extended"


class SmoothingMethod(Enum):
    """Methods for temporal smoothing."""
    KALMAN = "kalman"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"


@dataclass
class TrackState:
    """State information for tracked objects/features."""
    track_id: int
    position: np.ndarray  # 2D or 3D position
    velocity: np.ndarray  # Velocity vector
    acceleration: Optional[np.ndarray] = None  # Acceleration vector
    covariance: Optional[np.ndarray] = None  # State covariance
    age: int = 0  # Track age in frames
    hits: int = 0  # Number of successful detections
    time_since_update: int = 0  # Frames since last update
    confidence: float = 1.0  # Track confidence
    
    @property
    def is_active(self) -> bool:
        """Check if track is currently active."""
        return self.time_since_update < 5  # Active if updated within 5 frames


@dataclass
class TemporalFrame:
    """Frame information for temporal consistency."""
    frame_id: int
    timestamp: float
    detections: List[Any]
    tracks: List[TrackState]
    scene_layout: Optional[Any] = None
    camera_params: Optional[Any] = None
    stabilization_transform: Optional[np.ndarray] = None


class TemporalConsistencyConfig:
    """Configuration for temporal consistency and stabilization."""
    
    def __init__(
        self,
        filter_type: FilterType = FilterType.CONSTANT_VELOCITY,
        smoothing_method: SmoothingMethod = SmoothingMethod.KALMAN,
        max_track_age: int = 30,
        min_hits: int = 3,
        max_time_since_update: int = 5,
        track_association_threshold: float = 0.3,
        enable_scene_stabilization: bool = True,
        enable_camera_stabilization: bool = True,
        stabilization_window_size: int = 10,
        smoothing_alpha: float = 0.7,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.1,
        **kwargs
    ):
        self.filter_type = filter_type
        self.smoothing_method = smoothing_method
        self.max_track_age = max_track_age
        self.min_hits = min_hits
        self.max_time_since_update = max_time_since_update
        self.track_association_threshold = track_association_threshold
        self.enable_scene_stabilization = enable_scene_stabilization
        self.enable_camera_stabilization = enable_camera_stabilization
        self.stabilization_window_size = stabilization_window_size
        self.smoothing_alpha = smoothing_alpha
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.kwargs = kwargs


class TemporalStabilizer:
    """
    Advanced temporal consistency and stabilization system.
    
    This class provides comprehensive temporal filtering, object tracking,
    and scene stabilization for smooth video advertisement placement.
    
    Attributes:
        config: Configuration for temporal processing
        active_tracks: Currently active object tracks
        frame_history: Historical frame information
        kalman_filters: Kalman filters for different entities
        is_initialized: Initialization status
    """

    def __init__(self, config: TemporalConsistencyConfig):
        """
        Initialize the temporal stabilizer with specified configuration.
        
        Args:
            config: TemporalConsistencyConfig with processing parameters
        """
        self.config = config
        self.active_tracks: Dict[int, TrackState] = {}
        self.frame_history: List[TemporalFrame] = []
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        self.next_track_id = 0
        self.is_initialized = True
        
        # Stabilization data
        self.camera_stabilizer = CameraStabilizer(config)
        self.scene_stabilizer = SceneStabilizer(config)
        
        logger.info("TemporalStabilizer initialized successfully")

    def process_frame(
        self,
        frame_id: int,
        timestamp: float,
        detections: List[Any],
        scene_layout: Optional[Any] = None,
        camera_params: Optional[Any] = None
    ) -> TemporalFrame:
        """
        Process a video frame for temporal consistency.
        
        Args:
            frame_id: Frame identifier
            timestamp: Frame timestamp
            detections: Object detections in this frame
            scene_layout: Scene analysis results
            camera_params: Camera parameters
            
        Returns:
            TemporalFrame with stabilized and tracked information
        """
        try:
            logger.debug(f"Processing frame {frame_id} with {len(detections)} detections")
            
            # Update existing tracks with predictions
            self._predict_tracks()
            
            # Associate detections with existing tracks
            track_assignments = self._associate_detections_to_tracks(detections)
            
            # Update tracks with new detections
            updated_tracks = self._update_tracks(track_assignments, detections)
            
            # Create new tracks for unassigned detections
            new_tracks = self._create_new_tracks(track_assignments, detections)
            
            # Manage track lifecycle
            self._manage_track_lifecycle()
            
            # Apply scene stabilization if enabled
            stabilization_transform = None
            if self.config.enable_scene_stabilization and scene_layout:
                stabilization_transform = self.scene_stabilizer.stabilize_scene(
                    scene_layout, self.frame_history
                )
            
            # Apply camera stabilization if enabled
            if self.config.enable_camera_stabilization and camera_params:
                camera_params = self.camera_stabilizer.stabilize_camera(
                    camera_params, self.frame_history
                )
            
            # Apply temporal smoothing to detections
            smoothed_detections = self._apply_temporal_smoothing(detections)
            
            # Create temporal frame
            temporal_frame = TemporalFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=smoothed_detections,
                tracks=list(self.active_tracks.values()),
                scene_layout=scene_layout,
                camera_params=camera_params,
                stabilization_transform=stabilization_transform
            )
            
            # Update frame history
            self.frame_history.append(temporal_frame)
            if len(self.frame_history) > self.config.stabilization_window_size:
                self.frame_history.pop(0)
            
            logger.debug(f"Frame {frame_id} processed: {len(self.active_tracks)} active tracks")
            return temporal_frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            raise RuntimeError(f"Temporal processing error: {str(e)}")

    def _predict_tracks(self) -> None:
        """Predict the next state for all active tracks."""
        for track_id, track in self.active_tracks.items():
            if track_id in self.kalman_filters:
                kf = self.kalman_filters[track_id]
                kf.predict()
                
                # Update track state from filter
                track.position = kf.x[:len(track.position)]
                if len(kf.x) > len(track.position):
                    track.velocity = kf.x[len(track.position):len(track.position)*2]
                
                track.covariance = kf.P
                track.time_since_update += 1

    def _associate_detections_to_tracks(self, detections: List[Any]) -> Dict[str, List]:
        """Associate detections with existing tracks using Hungarian algorithm."""
        if not detections or not self.active_tracks:
            return {
                'matched': [],
                'unmatched_detections': list(range(len(detections))),
                'unmatched_tracks': list(self.active_tracks.keys())
            }
        
        # Compute cost matrix based on position distance
        detection_positions = self._extract_detection_positions(detections)
        track_positions = np.array([track.position for track in self.active_tracks.values()])
        track_ids = list(self.active_tracks.keys())
        
        if len(detection_positions) == 0 or len(track_positions) == 0:
            return {
                'matched': [],
                'unmatched_detections': list(range(len(detections))),
                'unmatched_tracks': track_ids
            }
        
        # Compute distance matrix
        cost_matrix = cdist(detection_positions, track_positions)
        
        # Apply Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on threshold
        matched_pairs = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]
            if cost < self.config.track_association_threshold:
                track_id = track_ids[track_idx]
                matched_pairs.append((det_idx, track_id))
                unmatched_detections.discard(det_idx)
                unmatched_tracks.discard(track_id)
        
        return {
            'matched': matched_pairs,
            'unmatched_detections': list(unmatched_detections),
            'unmatched_tracks': list(unmatched_tracks)
        }

    def _extract_detection_positions(self, detections: List[Any]) -> np.ndarray:
        """Extract 2D positions from detections."""
        positions = []
        
        for detection in detections:
            if hasattr(detection, 'bbox'):
                # Use bounding box center
                x1, y1, x2, y2 = detection.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                positions.append([center_x, center_y])
            elif hasattr(detection, 'center'):
                positions.append(detection.center[:2])  # Use first 2 dimensions
            else:
                # Default position
                positions.append([0, 0])
        
        return np.array(positions) if positions else np.empty((0, 2))

    def _update_tracks(self, assignments: Dict[str, List], detections: List[Any]) -> List[TrackState]:
        """Update existing tracks with new detections."""
        updated_tracks = []
        
        for det_idx, track_id in assignments['matched']:
            track = self.active_tracks[track_id]
            detection = detections[det_idx]
            
            # Extract measurement from detection
            measurement = self._extract_measurement_from_detection(detection)
            
            # Update Kalman filter
            if track_id in self.kalman_filters:
                kf = self.kalman_filters[track_id]
                kf.update(measurement)
                
                # Update track state
                track.position = kf.x[:len(track.position)]
                if len(kf.x) > len(track.position):
                    track.velocity = kf.x[len(track.position):len(track.position)*2]
                
                track.covariance = kf.P
            else:
                # Direct update if no filter
                track.position = measurement
            
            # Update track metadata
            track.hits += 1
            track.time_since_update = 0
            track.confidence = min(track.confidence + 0.1, 1.0)
            
            updated_tracks.append(track)
        
        return updated_tracks

    def _extract_measurement_from_detection(self, detection: Any) -> np.ndarray:
        """Extract measurement vector from detection."""
        if hasattr(detection, 'bbox'):
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return np.array([center_x, center_y])
        elif hasattr(detection, 'center'):
            return np.array(detection.center[:2])
        else:
            return np.array([0, 0])

    def _create_new_tracks(self, assignments: Dict[str, List], detections: List[Any]) -> List[TrackState]:
        """Create new tracks for unassigned detections."""
        new_tracks = []
        
        for det_idx in assignments['unmatched_detections']:
            detection = detections[det_idx]
            
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            position = self._extract_measurement_from_detection(detection)
            velocity = np.zeros_like(position)
            
            track = TrackState(
                track_id=track_id,
                position=position,
                velocity=velocity,
                age=0,
                hits=1,
                time_since_update=0,
                confidence=0.5
            )
            
            # Create Kalman filter for this track
            kf = self._create_kalman_filter(position)
            self.kalman_filters[track_id] = kf
            
            self.active_tracks[track_id] = track
            new_tracks.append(track)
        
        return new_tracks

    def _create_kalman_filter(self, initial_position: np.ndarray) -> KalmanFilter:
        """Create Kalman filter for object tracking."""
        dim_pos = len(initial_position)
        
        if self.config.filter_type == FilterType.CONSTANT_VELOCITY:
            # State: [x, y, vx, vy] or [x, y, z, vx, vy, vz]
            dim_state = dim_pos * 2
            kf = KalmanFilter(dim_x=dim_state, dim_z=dim_pos)
            
            # State transition matrix (constant velocity model)
            dt = 1.0  # Time step
            kf.F = np.eye(dim_state)
            for i in range(dim_pos):
                kf.F[i, i + dim_pos] = dt
            
            # Measurement matrix (observe position only)
            kf.H = np.zeros((dim_pos, dim_state))
            for i in range(dim_pos):
                kf.H[i, i] = 1.0
            
            # Initial state
            kf.x = np.zeros(dim_state)
            kf.x[:dim_pos] = initial_position
            
            # Process noise
            q_std = self.config.process_noise_std
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q_std**2, block_size=dim_pos)
            
            # Measurement noise
            r_std = self.config.measurement_noise_std
            kf.R = np.eye(dim_pos) * r_std**2
            
            # Initial covariance
            kf.P = np.eye(dim_state) * 1000.0
            
        elif self.config.filter_type == FilterType.CONSTANT_ACCELERATION:
            # State: [x, y, vx, vy, ax, ay] or [x, y, z, vx, vy, vz, ax, ay, az]
            dim_state = dim_pos * 3
            kf = KalmanFilter(dim_x=dim_state, dim_z=dim_pos)
            
            # State transition matrix (constant acceleration model)
            dt = 1.0
            kf.F = np.eye(dim_state)
            for i in range(dim_pos):
                kf.F[i, i + dim_pos] = dt  # position += velocity * dt
                kf.F[i, i + 2*dim_pos] = 0.5 * dt**2  # position += 0.5 * acceleration * dt^2
                kf.F[i + dim_pos, i + 2*dim_pos] = dt  # velocity += acceleration * dt
            
            # Measurement matrix
            kf.H = np.zeros((dim_pos, dim_state))
            for i in range(dim_pos):
                kf.H[i, i] = 1.0
            
            # Initial state
            kf.x = np.zeros(dim_state)
            kf.x[:dim_pos] = initial_position
            
            # Process and measurement noise (simplified)
            kf.Q = np.eye(dim_state) * 0.1
            kf.R = np.eye(dim_pos) * 1.0
            kf.P = np.eye(dim_state) * 1000.0
        
        else:
            # Default to constant velocity
            return self._create_kalman_filter(initial_position)
        
        return kf

    def _manage_track_lifecycle(self) -> None:
        """Manage the lifecycle of tracks (creation, deletion)."""
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            track.age += 1
            
            # Remove tracks that are too old or inactive
            if (track.time_since_update > self.config.max_time_since_update or
                track.age > self.config.max_track_age):
                tracks_to_remove.append(track_id)
            
            # Remove tracks with insufficient hits
            elif track.age > 10 and track.hits < self.config.min_hits:
                tracks_to_remove.append(track_id)
        
        # Remove inactive tracks
        for track_id in tracks_to_remove:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]

    def _apply_temporal_smoothing(self, detections: List[Any]) -> List[Any]:
        """Apply temporal smoothing to detections."""
        if self.config.smoothing_method == SmoothingMethod.EXPONENTIAL:
            return self._apply_exponential_smoothing(detections)
        elif self.config.smoothing_method == SmoothingMethod.GAUSSIAN:
            return self._apply_gaussian_smoothing(detections)
        else:
            # Kalman smoothing is already applied through tracking
            return detections

    def _apply_exponential_smoothing(self, detections: List[Any]) -> List[Any]:
        """Apply exponential smoothing to detection parameters."""
        if len(self.frame_history) == 0:
            return detections
        
        alpha = self.config.smoothing_alpha
        previous_detections = self.frame_history[-1].detections
        
        # Simple smoothing for demonstration
        # In practice, you'd match detections and smooth specific attributes
        smoothed_detections = []
        for i, detection in enumerate(detections):
            if i < len(previous_detections):
                # Smooth confidence scores as an example
                if hasattr(detection, 'confidence') and hasattr(previous_detections[i], 'confidence'):
                    smoothed_confidence = (alpha * detection.confidence + 
                                         (1 - alpha) * previous_detections[i].confidence)
                    detection.confidence = smoothed_confidence
            
            smoothed_detections.append(detection)
        
        return smoothed_detections

    def _apply_gaussian_smoothing(self, detections: List[Any]) -> List[Any]:
        """Apply Gaussian smoothing using historical data."""
        # Placeholder for Gaussian smoothing implementation
        return detections

    def get_active_tracks(self) -> List[TrackState]:
        """Get currently active tracks."""
        return [track for track in self.active_tracks.values() if track.is_active]

    def get_track_predictions(self, num_steps: int = 1) -> Dict[int, np.ndarray]:
        """Get predicted positions for active tracks."""
        predictions = {}
        
        for track_id, track in self.active_tracks.items():
            if track_id in self.kalman_filters and track.is_active:
                kf = self.kalman_filters[track_id]
                
                # Make multiple predictions
                predicted_positions = []
                current_state = kf.x.copy()
                current_cov = kf.P.copy()
                
                for step in range(num_steps):
                    # Predict next state
                    kf.x = current_state
                    kf.P = current_cov
                    kf.predict()
                    
                    # Extract position
                    position = kf.x[:len(track.position)]
                    predicted_positions.append(position.copy())
                    
                    current_state = kf.x.copy()
                    current_cov = kf.P.copy()
                
                predictions[track_id] = np.array(predicted_positions)
        
        return predictions

    def reset(self) -> None:
        """Reset the temporal stabilizer state."""
        self.active_tracks.clear()
        self.kalman_filters.clear()
        self.frame_history.clear()
        self.next_track_id = 0
        
        self.camera_stabilizer.reset()
        self.scene_stabilizer.reset()
        
        logger.info("TemporalStabilizer reset completed")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.reset()
        logger.info("TemporalStabilizer cleanup completed")


class CameraStabilizer:
    """Camera parameter stabilization for smooth transitions."""
    
    def __init__(self, config: TemporalConsistencyConfig):
        self.config = config
        self.camera_history: List[Any] = []
    
    def stabilize_camera(self, camera_params: Any, frame_history: List[TemporalFrame]) -> Any:
        """Stabilize camera parameters using temporal information."""
        if not self.config.enable_camera_stabilization:
            return camera_params
        
        # Add current parameters to history
        self.camera_history.append(camera_params)
        if len(self.camera_history) > self.config.stabilization_window_size:
            self.camera_history.pop(0)
        
        # Apply smoothing to intrinsic parameters
        if len(self.camera_history) > 1:
            camera_params = self._smooth_camera_intrinsics(camera_params)
        
        return camera_params
    
    def _smooth_camera_intrinsics(self, current_params: Any) -> Any:
        """Smooth camera intrinsic parameters."""
        # Simple exponential smoothing for focal length
        if hasattr(current_params, 'intrinsics') and len(self.camera_history) > 1:
            alpha = self.config.smoothing_alpha
            prev_params = self.camera_history[-2]
            
            if hasattr(prev_params, 'intrinsics'):
                # Smooth focal lengths
                current_params.intrinsics.fx = (alpha * current_params.intrinsics.fx + 
                                               (1 - alpha) * prev_params.intrinsics.fx)
                current_params.intrinsics.fy = (alpha * current_params.intrinsics.fy + 
                                               (1 - alpha) * prev_params.intrinsics.fy)
        
        return current_params
    
    def reset(self) -> None:
        """Reset camera stabilizer."""
        self.camera_history.clear()


class SceneStabilizer:
    """Scene layout stabilization for consistent plane detection."""
    
    def __init__(self, config: TemporalConsistencyConfig):
        self.config = config
        self.scene_history: List[Any] = []
    
    def stabilize_scene(self, scene_layout: Any, frame_history: List[TemporalFrame]) -> Optional[np.ndarray]:
        """Stabilize scene layout and return stabilization transform."""
        if not self.config.enable_scene_stabilization:
            return None
        
        # Add current scene to history
        self.scene_history.append(scene_layout)
        if len(self.scene_history) > self.config.stabilization_window_size:
            self.scene_history.pop(0)
        
        # Compute stabilization transform
        if len(self.scene_history) > 1:
            return self._compute_scene_stabilization_transform()
        
        return None
    
    def _compute_scene_stabilization_transform(self) -> Optional[np.ndarray]:
        """Compute transformation for scene stabilization."""
        # Placeholder for scene stabilization transform computation
        # In practice, this would align planes across frames
        return np.eye(4)  # Identity transform as placeholder
    
    def reset(self) -> None:
        """Reset scene stabilizer."""
        self.scene_history.clear() 