"""
Track Data Models for Multi-Object Tracking

This module defines the core data structures for tracking objects across video frames,
including track state management, quality assessment, and trajectory history.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import torch

from ..detection_models import Detection


class TrackState(Enum):
    """Track state enumeration."""
    NEW = "new"                    # Newly created track
    TRACKED = "tracked"            # Successfully tracked
    LOST = "lost"                 # Temporarily lost
    REMOVED = "removed"           # Permanently removed
    CONFIRMED = "confirmed"       # Confirmed after initial frames
    TENTATIVE = "tentative"       # Tentative in first few frames


class TrackQuality(Enum):
    """Track quality assessment."""
    HIGH = "high"         # High quality track with consistent detections
    MEDIUM = "medium"     # Medium quality track with some inconsistencies
    LOW = "low"          # Low quality track with frequent interruptions
    UNCERTAIN = "uncertain"  # Quality uncertain, needs more observations


@dataclass
class TrackingMetadata:
    """Additional metadata for tracks."""
    
    # Object information
    object_class: str = ""
    object_class_id: int = -1
    class_confidence: float = 0.0
    
    # Advertisement relevance
    ad_relevance_score: float = 0.0
    placement_suitability: float = 0.0
    visibility_score: float = 1.0
    
    # Motion characteristics
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    motion_pattern: str = "static"  # static, linear, curved, erratic
    
    # Scene context
    depth_estimate: Optional[float] = None
    scene_complexity: float = 0.5
    lighting_condition: str = "normal"
    
    # Track statistics
    avg_confidence: float = 0.0
    detection_rate: float = 0.0  # Ratio of frames with detections
    occlusion_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'object_class': self.object_class,
            'object_class_id': self.object_class_id,
            'class_confidence': self.class_confidence,
            'ad_relevance_score': self.ad_relevance_score,
            'placement_suitability': self.placement_suitability,
            'visibility_score': self.visibility_score,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'motion_pattern': self.motion_pattern,
            'depth_estimate': self.depth_estimate,
            'scene_complexity': self.scene_complexity,
            'lighting_condition': self.lighting_condition,
            'avg_confidence': self.avg_confidence,
            'detection_rate': self.detection_rate,
            'occlusion_count': self.occlusion_count
        }


@dataclass
class Track:
    """
    Enhanced track object for multi-object tracking with comprehensive features
    for video advertisement placement scenarios.
    """
    
    # Core track information
    track_id: int
    state: TrackState = TrackState.NEW
    quality: TrackQuality = TrackQuality.UNCERTAIN
    
    # Trajectory data
    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    # Feature representations for re-identification
    feature_history: List[torch.Tensor] = field(default_factory=list)
    avg_feature: Optional[torch.Tensor] = None
    feature_buffer_size: int = 30
    
    # Motion tracking
    kalman_filter: Optional[Any] = None  # KalmanFilter instance
    predicted_bbox: Optional[Tuple[float, float, float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Track lifecycle
    age: int = 0  # Total frames since track creation
    hits: int = 0  # Number of successful detections
    hit_streak: int = 0  # Consecutive successful detections
    time_since_update: int = 0  # Frames since last update
    
    # Track management
    max_age: int = 30  # Maximum frames without detection
    min_hits: int = 3   # Minimum hits for confirmation
    tentative_period: int = 3  # Frames to remain tentative
    
    # Re-identification
    reid_features: List[torch.Tensor] = field(default_factory=list)
    appearance_descriptor: Optional[torch.Tensor] = None
    feature_update_rate: float = 0.1  # EMA rate for feature updates
    
    # Advertisement-specific metadata
    metadata: TrackingMetadata = field(default_factory=TrackingMetadata)
    
    # Performance optimization
    last_frame_id: int = -1
    is_confirmed: bool = False
    is_active: bool = True
    
    def __post_init__(self):
        """Initialize track after creation."""
        self.timestamps.append(time.time())
        self.frame_ids.append(0)
    
    def update(self, detection: Detection, frame_id: int, kalman_filter=None):
        """
        Update track with new detection.
        
        Args:
            detection: New detection to associate with track
            frame_id: Current frame ID
            kalman_filter: Kalman filter for motion prediction
        """
        # Update basic information
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_frame_id = frame_id
        
        # Update trajectory
        self.bbox_history.append(detection.bbox)
        self.confidence_history.append(detection.confidence)
        self.frame_ids.append(frame_id)
        self.timestamps.append(time.time())
        
        # Update metadata
        self.metadata.object_class = detection.class_name
        self.metadata.object_class_id = detection.class_id
        self.metadata.class_confidence = detection.confidence
        
        # Update features if available
        if detection.features is not None:
            self._update_features(detection.features)
        
        # Update motion model
        if kalman_filter is not None:
            self.kalman_filter = kalman_filter
            self.predicted_bbox = None  # Reset prediction
        
        # Update velocity
        self._update_velocity()
        
        # Update state based on hits
        if self.hits >= self.min_hits and self.state != TrackState.CONFIRMED:
            self.state = TrackState.CONFIRMED
            self.is_confirmed = True
        elif self.state == TrackState.NEW and self.age < self.tentative_period:
            self.state = TrackState.TENTATIVE
        elif self.state == TrackState.TENTATIVE and self.hits >= 2:
            self.state = TrackState.TRACKED
        
        # Update quality assessment
        self._update_quality()
        
        # Maintain buffer sizes
        self._maintain_buffers()
    
    def predict(self, frame_id: int):
        """
        Predict track position for current frame.
        
        Args:
            frame_id: Current frame ID
        """
        self.age += 1
        self.time_since_update += 1
        
        # Update state based on time since update
        if self.time_since_update > 1:
            self.hit_streak = 0
            
        if self.state == TrackState.TRACKED and self.time_since_update > 5:
            self.state = TrackState.LOST
        
        # Predict using Kalman filter
        if self.kalman_filter is not None:
            self.predicted_bbox = self.kalman_filter.predict()
        else:
            # Simple linear prediction based on velocity
            if len(self.bbox_history) >= 2:
                last_bbox = self.bbox_history[-1]
                x1, y1, x2, y2 = last_bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                
                # Apply velocity
                new_cx = cx + self.velocity[0]
                new_cy = cy + self.velocity[1]
                
                self.predicted_bbox = (
                    new_cx - w/2, new_cy - h/2,
                    new_cx + w/2, new_cy + h/2
                )
            else:
                self.predicted_bbox = self.bbox_history[-1] if self.bbox_history else None
    
    def mark_missed(self):
        """Mark track as missed (no detection in current frame)."""
        self.time_since_update += 1
        self.hit_streak = 0
        
        if self.time_since_update > self.max_age:
            self.state = TrackState.REMOVED
            self.is_active = False
        elif self.state == TrackState.TRACKED:
            self.state = TrackState.LOST
    
    def get_current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get current bounding box (predicted or last known)."""
        if self.predicted_bbox is not None:
            return self.predicted_bbox
        elif self.bbox_history:
            return self.bbox_history[-1]
        else:
            return None
    
    def get_trajectory(self, max_length: int = None) -> List[Tuple[float, float]]:
        """
        Get track trajectory as list of center points.
        
        Args:
            max_length: Maximum number of points to return
            
        Returns:
            List of (x, y) center coordinates
        """
        centers = []
        bboxes = self.bbox_history[-max_length:] if max_length else self.bbox_history
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        return centers
    
    def compute_iou(self, bbox: Tuple[float, float, float, float]) -> float:
        """Compute IoU with given bounding box."""
        current_bbox = self.get_current_bbox()
        if current_bbox is None:
            return 0.0
        
        # Compute intersection
        x1_i = max(current_bbox[0], bbox[0])
        y1_i = max(current_bbox[1], bbox[1])
        x2_i = min(current_bbox[2], bbox[2])
        y2_i = min(current_bbox[3], bbox[3])
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        area2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_feature_similarity(self, features: torch.Tensor) -> float:
        """Compute cosine similarity with track's average features."""
        if self.avg_feature is None or features is None:
            return 0.0
        
        # Normalize features
        norm_features = torch.nn.functional.normalize(features, p=2, dim=-1)
        norm_avg = torch.nn.functional.normalize(self.avg_feature, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.dot(norm_features.flatten(), norm_avg.flatten()).item()
        return max(0.0, similarity)  # Clamp to [0, 1]
    
    def _update_features(self, features: torch.Tensor):
        """Update track features with exponential moving average."""
        if features is None:
            return
        
        self.feature_history.append(features.clone())
        
        # Maintain buffer size
        if len(self.feature_history) > self.feature_buffer_size:
            self.feature_history.pop(0)
        
        # Update average feature
        if self.avg_feature is None:
            self.avg_feature = features.clone()
        else:
            # Exponential moving average
            self.avg_feature = (
                (1 - self.feature_update_rate) * self.avg_feature +
                self.feature_update_rate * features
            )
    
    def _update_velocity(self):
        """Update velocity based on recent trajectory."""
        if len(self.bbox_history) < 2:
            return
        
        # Get centers of last two bboxes
        bbox1 = self.bbox_history[-2]
        bbox2 = self.bbox_history[-1]
        
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Compute velocity
        self.velocity = (center2[0] - center1[0], center2[1] - center1[1])
        
        # Update metadata
        self.metadata.velocity = self.velocity
        
        # Classify motion pattern
        speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed < 2.0:
            self.metadata.motion_pattern = "static"
        elif len(self.bbox_history) >= 5:
            # Analyze motion pattern over last 5 frames
            velocities = []
            for i in range(len(self.bbox_history) - 4, len(self.bbox_history)):
                if i > 0:
                    b1 = self.bbox_history[i-1]
                    b2 = self.bbox_history[i]
                    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
                    velocities.append((c2[0] - c1[0], c2[1] - c1[1]))
            
            if velocities:
                # Check consistency of velocities
                vel_std = np.std([np.sqrt(v[0]**2 + v[1]**2) for v in velocities])
                if vel_std < 1.0:
                    self.metadata.motion_pattern = "linear"
                elif vel_std < 3.0:
                    self.metadata.motion_pattern = "curved"
                else:
                    self.metadata.motion_pattern = "erratic"
    
    def _update_quality(self):
        """Update track quality based on various factors."""
        if self.age == 0:
            return
        
        # Detection rate
        detection_rate = self.hits / self.age
        self.metadata.detection_rate = detection_rate
        
        # Average confidence
        if self.confidence_history:
            self.metadata.avg_confidence = np.mean(self.confidence_history)
        
        # Determine quality
        if detection_rate > 0.8 and self.metadata.avg_confidence > 0.7:
            self.quality = TrackQuality.HIGH
        elif detection_rate > 0.6 and self.metadata.avg_confidence > 0.5:
            self.quality = TrackQuality.MEDIUM
        elif detection_rate > 0.3:
            self.quality = TrackQuality.LOW
        else:
            self.quality = TrackQuality.UNCERTAIN
    
    def _maintain_buffers(self):
        """Maintain buffer sizes to prevent memory growth."""
        max_history = 100  # Maximum trajectory length
        
        if len(self.bbox_history) > max_history:
            # Keep recent history
            self.bbox_history = self.bbox_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
            self.frame_ids = self.frame_ids[-max_history:]
            self.timestamps = self.timestamps[-max_history:]
    
    def is_confirmed_track(self) -> bool:
        """Check if track is confirmed."""
        return self.is_confirmed or self.hits >= self.min_hits
    
    def should_remove(self) -> bool:
        """Check if track should be removed."""
        return (
            self.state == TrackState.REMOVED or
            self.time_since_update > self.max_age or
            not self.is_active
        )
    
    def get_advertisement_score(self) -> float:
        """Compute advertisement placement score for this track."""
        base_score = 0.5
        
        # Quality factor
        quality_scores = {
            TrackQuality.HIGH: 1.0,
            TrackQuality.MEDIUM: 0.8,
            TrackQuality.LOW: 0.5,
            TrackQuality.UNCERTAIN: 0.2
        }
        quality_factor = quality_scores.get(self.quality, 0.2)
        
        # Stability factor (based on detection rate)
        stability_factor = min(1.0, self.metadata.detection_rate * 1.2)
        
        # Confidence factor
        confidence_factor = self.metadata.avg_confidence
        
        # Visibility factor
        visibility_factor = self.metadata.visibility_score
        
        # Motion factor (stable motion is better for ads)
        motion_factors = {
            "static": 1.0,
            "linear": 0.9,
            "curved": 0.7,
            "erratic": 0.3
        }
        motion_factor = motion_factors.get(self.metadata.motion_pattern, 0.5)
        
        # Combined score
        score = (
            base_score * 0.2 +
            quality_factor * 0.25 +
            stability_factor * 0.2 +
            confidence_factor * 0.15 +
            visibility_factor * 0.1 +
            motion_factor * 0.1
        )
        
        return min(1.0, score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary representation."""
        return {
            'track_id': self.track_id,
            'state': self.state.value,
            'quality': self.quality.value,
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'is_confirmed': self.is_confirmed,
            'is_active': self.is_active,
            'current_bbox': self.get_current_bbox(),
            'velocity': self.velocity,
            'advertisement_score': self.get_advertisement_score(),
            'metadata': self.metadata.to_dict(),
            'trajectory_length': len(self.bbox_history)
        } 