"""
ByteTracker Implementation for Video Advertisement Placement

This module implements the ByteTrack algorithm with custom modifications for
video advertisement scenarios, featuring advanced occlusion handling,
re-identification, and temporal consistency optimization.
"""

import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
import torch

from .track_models import Track, TrackState, TrackQuality, TrackingMetadata
from .motion_models import KalmanTracker, MotionModel, MotionParameters
from .association import TrackAssociation, AssociationMethod
from ..detection_models import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for ByteTracker."""
    
    # Detection thresholds
    high_threshold: float = 0.6  # High confidence detection threshold
    low_threshold: float = 0.1   # Low confidence detection threshold
    match_threshold: float = 0.8 # IoU threshold for first association
    second_match_threshold: float = 0.5  # IoU threshold for second association
    
    # Track management
    track_buffer_size: int = 30  # Maximum frames to keep lost tracks
    min_box_area: float = 10.0   # Minimum bounding box area
    aspect_ratio_thresh: float = 1.6  # Maximum aspect ratio
    
    # Frame rate and temporal settings
    frame_rate: int = 30
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    
    # Motion model settings
    motion_model: MotionModel = MotionModel.ADAPTIVE
    motion_parameters: MotionParameters = field(default_factory=MotionParameters)
    
    # Re-identification settings
    use_reid: bool = True
    reid_weight: float = 0.3
    appearance_thresh: float = 0.25
    
    # Camera motion compensation
    use_camera_motion_compensation: bool = True
    
    # Advertisement-specific settings
    prioritize_person_tracks: bool = True
    furniture_stability_bonus: float = 1.2
    
    # Performance settings
    max_tracks: int = 100
    cleanup_interval: int = 30  # Frames between cleanup
    
    # Occlusion handling
    occlusion_detection: bool = True
    occlusion_threshold: float = 0.7  # IoU threshold for occlusion detection
    max_occlusion_frames: int = 10
    
    # Temporal consistency
    temporal_smoothing: bool = True
    smoothing_alpha: float = 0.7
    interpolation_enabled: bool = True
    
    def validate(self):
        """Validate configuration parameters."""
        if self.high_threshold <= self.low_threshold:
            raise ValueError("high_threshold must be greater than low_threshold")
        
        if self.match_threshold <= 0 or self.match_threshold > 1:
            raise ValueError("match_threshold must be in (0, 1]")
        
        if self.track_buffer_size <= 0:
            raise ValueError("track_buffer_size must be positive")


class ByteTracker:
    """
    Enhanced ByteTracker for video advertisement placement scenarios.
    
    Features:
    - Two-stage association with high and low confidence detections
    - Advanced motion models with camera motion compensation
    - Re-identification features for long-term tracking
    - Occlusion handling and track interpolation
    - Advertisement-specific optimizations
    """
    
    def __init__(self, config: TrackingConfig):
        """
        Initialize ByteTracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.config.validate()
        
        # Track management
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        
        # Track ID management
        self.track_id_count = 0
        
        # Motion and association
        self.kalman_tracker = KalmanTracker(
            motion_model=config.motion_model,
            parameters=config.motion_parameters
        )
        self.associator = TrackAssociation(method=AssociationMethod.HUNGARIAN)
        
        # Re-identification (placeholder for future implementation)
        self.reid_extractor = None
        if config.use_reid:
            self._initialize_reid()
        
        # Performance tracking
        self.frame_count = 0
        self.total_tracks_created = 0
        self.processing_times = []
        
        # Camera motion
        self.last_frame = None
        self.camera_homography = None
        
        logger.info(f"ByteTracker initialized with config: {config}")
    
    def update(self, detections: List[Detection], frame_id: int,
               frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections for current frame
            frame_id: Current frame ID
            frame: Optional frame image for camera motion compensation
            
        Returns:
            List of active tracks
        """
        start_time = time.time()
        self.frame_count = frame_id
        
        try:
            # Update camera motion if frame provided
            if frame is not None and self.config.use_camera_motion_compensation:
                self._update_camera_motion(frame)
            
            # Preprocess detections
            detections = self._preprocess_detections(detections, frame_id)
            
            # Separate detections by confidence
            high_conf_dets, low_conf_dets = self._separate_detections(detections)
            
            # Predict track positions
            self._predict_tracks()
            
            # First association: high confidence detections
            matched_tracks, unmatched_tracks, unmatched_dets = self._first_association(
                self.tracked_tracks, high_conf_dets
            )
            
            # Update matched tracks
            for track_idx, det_idx in matched_tracks:
                track = self.tracked_tracks[track_idx]
                detection = high_conf_dets[det_idx]
                
                # Create Kalman filter if needed
                kalman_filter = self._get_or_create_kalman_filter(track, detection)
                track.update(detection, frame_id, kalman_filter)
            
            # Second association: unmatched tracks with low confidence detections
            if low_conf_dets:
                unmatched_tracked_tracks = [self.tracked_tracks[i] for i in unmatched_tracks]
                matched_tracks_2, unmatched_tracks_2, unmatched_low_dets = self._second_association(
                    unmatched_tracked_tracks, low_conf_dets
                )
                
                # Update tracks from second association
                for track_idx, det_idx in matched_tracks_2:
                    track = unmatched_tracked_tracks[track_idx]
                    detection = low_conf_dets[det_idx]
                    
                    kalman_filter = self._get_or_create_kalman_filter(track, detection)
                    track.update(detection, frame_id, kalman_filter)
                
                # Update unmatched tracks list
                unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_2]
                
                # Combine unmatched detections
                unmatched_dets.extend(unmatched_low_dets)
            
            # Third association: lost tracks with unmatched detections
            if self.lost_tracks and unmatched_dets:
                matched_lost, unmatched_lost, unmatched_dets = self._third_association(
                    self.lost_tracks, unmatched_dets
                )
                
                # Reactivate matched lost tracks
                for track_idx, det_idx in matched_lost:
                    track = self.lost_tracks[track_idx]
                    detection = unmatched_dets[det_idx]
                    
                    track.state = TrackState.TRACKED
                    kalman_filter = self._get_or_create_kalman_filter(track, detection)
                    track.update(detection, frame_id, kalman_filter)
                    
                    self.tracked_tracks.append(track)
                
                # Update lost tracks
                self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]
                
                # Remove matched detections
                unmatched_dets = [unmatched_dets[i] for i in range(len(unmatched_dets)) 
                                if i not in [det_idx for _, det_idx in matched_lost]]
            
            # Handle unmatched tracks
            self._handle_unmatched_tracks(unmatched_tracks, frame_id)
            
            # Create new tracks from unmatched high-confidence detections
            self._create_new_tracks(unmatched_dets, frame_id)
            
            # Update lost tracks
            self._update_lost_tracks(frame_id)
            
            # Clean up removed tracks periodically
            if frame_id % self.config.cleanup_interval == 0:
                self._cleanup_tracks()
            
            # Post-processing
            active_tracks = self._postprocess_tracks(frame_id)
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return active_tracks
            
        except Exception as e:
            logger.error(f"Tracking update failed: {e}")
            return self.tracked_tracks
    
    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        return [track for track in self.tracked_tracks if track.is_active]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (active and lost)."""
        return self.tracked_tracks + self.lost_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.get_all_tracks():
            if track.track_id == track_id:
                return track
        return None
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking performance statistics."""
        return {
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': len(self.tracked_tracks),
            'lost_tracks': len(self.lost_tracks),
            'removed_tracks': len(self.removed_tracks),
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'frames_processed': self.frame_count,
            'track_quality_distribution': self._get_quality_distribution()
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.track_id_count = 0
        self.frame_count = 0
        self.total_tracks_created = 0
        self.processing_times.clear()
        self.last_frame = None
        self.camera_homography = None
    
    def _preprocess_detections(self, detections: List[Detection], frame_id: int) -> List[Detection]:
        """Preprocess detections before tracking."""
        processed_detections = []
        
        for detection in detections:
            # Filter by minimum area
            bbox_area = detection.area()
            if bbox_area < self.config.min_box_area:
                continue
            
            # Filter by aspect ratio
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = max(width / height, height / width) if height > 0 else float('inf')
            
            if aspect_ratio > self.config.aspect_ratio_thresh:
                continue
            
            # Apply camera motion compensation
            if self.camera_homography is not None:
                compensated_bbox = self.kalman_tracker.compensate_camera_motion(
                    detection.bbox, self.camera_homography
                )
                detection.bbox = compensated_bbox
            
            # Update frame ID
            detection.frame_id = frame_id
            
            processed_detections.append(detection)
        
        return processed_detections
    
    def _separate_detections(self, detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """Separate detections by confidence threshold."""
        high_conf_dets = []
        low_conf_dets = []
        
        for detection in detections:
            if detection.confidence >= self.config.high_threshold:
                high_conf_dets.append(detection)
            elif detection.confidence >= self.config.low_threshold:
                low_conf_dets.append(detection)
        
        return high_conf_dets, low_conf_dets
    
    def _predict_tracks(self):
        """Predict positions for all active tracks."""
        for track in self.tracked_tracks:
            track.predict(self.frame_count)
    
    def _first_association(self, tracks: List[Track], detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """First association: high confidence detections with tracked tracks."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix (IoU-based)
        iou_matrix = self._compute_iou_matrix(tracks, detections)
        
        # Apply re-identification if available
        if self.config.use_reid and self.reid_extractor:
            reid_matrix = self._compute_reid_matrix(tracks, detections)
            # Combine IoU and ReID scores
            cost_matrix = (1 - self.config.reid_weight) * iou_matrix + self.config.reid_weight * reid_matrix
        else:
            cost_matrix = iou_matrix
        
        # Apply Hungarian algorithm
        matched_pairs, unmatched_tracks, unmatched_detections = self.associator.associate(
            cost_matrix, threshold=self.config.match_threshold
        )
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _second_association(self, tracks: List[Track], detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Second association: low confidence detections with unmatched tracks."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Use only IoU for second association (lower threshold)
        iou_matrix = self._compute_iou_matrix(tracks, detections)
        
        matched_pairs, unmatched_tracks, unmatched_detections = self.associator.associate(
            iou_matrix, threshold=self.config.second_match_threshold
        )
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _third_association(self, tracks: List[Track], detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Third association: lost tracks with remaining detections."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Use lower threshold for lost track association
        iou_matrix = self._compute_iou_matrix(tracks, detections)
        
        matched_pairs, unmatched_tracks, unmatched_detections = self.associator.associate(
            iou_matrix, threshold=self.config.second_match_threshold
        )
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _compute_iou_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_bbox = track.get_current_bbox()
            if track_bbox is None:
                continue
                
            for j, detection in enumerate(detections):
                iou = self._compute_bbox_iou(track_bbox, detection.bbox)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    def _compute_reid_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Compute re-identification similarity matrix."""
        if not tracks or not detections or self.reid_extractor is None:
            return np.zeros((len(tracks), len(detections)))
        
        reid_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            if track.avg_feature is None:
                continue
                
            for j, detection in enumerate(detections):
                if detection.features is None:
                    continue
                
                similarity = track.compute_feature_similarity(detection.features)
                reid_matrix[i, j] = similarity
        
        return reid_matrix
    
    def _compute_bbox_iou(self, bbox1: Tuple[float, float, float, float], 
                         bbox2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Intersection area
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _handle_unmatched_tracks(self, unmatched_track_indices: List[int], frame_id: int):
        """Handle tracks that were not matched with any detection."""
        tracks_to_remove = []
        
        for track_idx in unmatched_track_indices:
            track = self.tracked_tracks[track_idx]
            track.mark_missed()
            
            # Move to lost tracks if necessary
            if track.state == TrackState.LOST and track not in self.lost_tracks:
                self.lost_tracks.append(track)
                tracks_to_remove.append(track_idx)
            elif track.should_remove():
                self.removed_tracks.append(track)
                tracks_to_remove.append(track_idx)
        
        # Remove tracks from tracked list
        for track_idx in sorted(tracks_to_remove, reverse=True):
            self.tracked_tracks.pop(track_idx)
    
    def _create_new_tracks(self, unmatched_detections: List[Detection], frame_id: int):
        """Create new tracks from unmatched high-confidence detections."""
        for detection in unmatched_detections:
            # Only create tracks for high-confidence detections
            if detection.confidence >= self.config.new_track_thresh:
                # Check if we've reached maximum tracks
                if len(self.tracked_tracks) >= self.config.max_tracks:
                    break
                
                # Create new track
                track = Track(
                    track_id=self._get_next_track_id(),
                    state=TrackState.NEW,
                    max_age=self.config.track_buffer_size
                )
                
                # Initialize with detection
                kalman_filter = self._get_or_create_kalman_filter(track, detection)
                track.update(detection, frame_id, kalman_filter)
                
                # Apply advertisement-specific scoring
                self._update_advertisement_metadata(track, detection)
                
                self.tracked_tracks.append(track)
                self.total_tracks_created += 1
    
    def _update_lost_tracks(self, frame_id: int):
        """Update lost tracks and remove old ones."""
        tracks_to_remove = []
        
        for i, track in enumerate(self.lost_tracks):
            track.age += 1
            track.time_since_update += 1
            
            if track.should_remove():
                self.removed_tracks.append(track)
                tracks_to_remove.append(i)
        
        # Remove old lost tracks
        for track_idx in sorted(tracks_to_remove, reverse=True):
            self.lost_tracks.pop(track_idx)
    
    def _postprocess_tracks(self, frame_id: int) -> List[Track]:
        """Post-process tracks for output."""
        active_tracks = []
        
        for track in self.tracked_tracks:
            if track.is_confirmed_track() and track.is_active:
                # Apply temporal smoothing if enabled
                if self.config.temporal_smoothing:
                    self._apply_temporal_smoothing(track)
                
                active_tracks.append(track)
        
        return active_tracks
    
    def _apply_temporal_smoothing(self, track: Track):
        """Apply temporal smoothing to track trajectory."""
        if len(track.bbox_history) < 2:
            return
        
        # Simple exponential smoothing
        alpha = self.config.smoothing_alpha
        current_bbox = track.bbox_history[-1]
        prev_bbox = track.bbox_history[-2]
        
        smoothed_bbox = tuple(
            alpha * curr + (1 - alpha) * prev
            for curr, prev in zip(current_bbox, prev_bbox)
        )
        
        track.bbox_history[-1] = smoothed_bbox
    
    def _get_or_create_kalman_filter(self, track: Track, detection: Detection):
        """Get or create Kalman filter for track."""
        if track.kalman_filter is None:
            track.kalman_filter = self.kalman_tracker.create_tracker(detection.bbox)
        
        return track.kalman_filter
    
    def _update_advertisement_metadata(self, track: Track, detection: Detection):
        """Update advertisement-specific metadata for track."""
        metadata = track.metadata
        
        # Set relevance score based on object class
        if detection.class_name.lower() in ['person', 'people']:
            metadata.ad_relevance_score = 0.9
            if self.config.prioritize_person_tracks:
                track.max_age = int(self.config.track_buffer_size * 1.5)
        elif 'furniture' in detection.class_name.lower():
            metadata.ad_relevance_score = 0.7
            metadata.placement_suitability *= self.config.furniture_stability_bonus
        elif detection.class_name.lower() in ['hand', 'arm']:
            metadata.ad_relevance_score = 0.8
        else:
            metadata.ad_relevance_score = 0.5
        
        # Update placement suitability based on detection confidence
        metadata.placement_suitability = detection.confidence * metadata.ad_relevance_score
    
    def _update_camera_motion(self, frame: np.ndarray):
        """Update camera motion estimation."""
        if frame is None:
            return
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        self.camera_homography = self.kalman_tracker.update_camera_motion(gray_frame)
        self.last_frame = gray_frame
    
    def _initialize_reid(self):
        """Initialize re-identification feature extractor."""
        # Placeholder for re-identification initialization
        # In practice, this would load a pre-trained ReID model
        logger.info("Re-identification initialized (placeholder)")
    
    def _get_next_track_id(self) -> int:
        """Get next available track ID."""
        self.track_id_count += 1
        return self.track_id_count
    
    def _cleanup_tracks(self):
        """Clean up old removed tracks to prevent memory growth."""
        max_removed_tracks = 1000  # Keep at most 1000 removed tracks
        
        if len(self.removed_tracks) > max_removed_tracks:
            # Keep only the most recent removed tracks
            self.removed_tracks = self.removed_tracks[-max_removed_tracks:]
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of track qualities."""
        distribution = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNCERTAIN': 0}
        
        for track in self.get_all_tracks():
            distribution[track.quality.value.upper()] += 1
        
        return distribution 