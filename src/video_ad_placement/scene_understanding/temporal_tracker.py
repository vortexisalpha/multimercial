"""
Temporal Tracking for 3D Plane Detection

This module implements temporal tracking of detected planes across video frames
to maintain consistency and stability for wall surface analysis in advertisement
placement scenarios.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .plane_models import Plane, PlaneType, TemporalTrackingConfig
from .geometry_utils import GeometryUtils

logger = logging.getLogger(__name__)


@dataclass
class PlaneTrack:
    """Temporal track for a detected plane across frames."""
    
    track_id: int
    planes: List[Plane] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    area_history: List[float] = field(default_factory=list)
    normal_history: List[torch.Tensor] = field(default_factory=list)
    center_history: List[torch.Tensor] = field(default_factory=list)
    
    # Track statistics
    first_seen: int = 0
    last_seen: int = 0
    total_detections: int = 0
    consecutive_misses: int = 0
    
    # Quality metrics
    stability_score: float = 0.0
    temporal_consistency: float = 0.0
    average_confidence: float = 0.0
    
    def __post_init__(self):
        """Initialize track after creation."""
        if self.planes:
            self.update_statistics()
    
    def add_detection(self, plane: Plane, frame_id: int):
        """Add new plane detection to track."""
        self.planes.append(plane)
        self.confidence_history.append(plane.confidence)
        self.area_history.append(plane.area)
        self.normal_history.append(plane.normal.clone())
        self.center_history.append(plane.point.clone())
        
        self.last_seen = frame_id
        self.total_detections += 1
        self.consecutive_misses = 0
        
        # Update first seen if this is the first detection
        if self.total_detections == 1:
            self.first_seen = frame_id
        
        self.update_statistics()
    
    def mark_miss(self, frame_id: int):
        """Mark frame where plane was not detected."""
        self.consecutive_misses += 1
    
    def update_statistics(self):
        """Update track quality metrics."""
        if not self.planes:
            return
        
        # Average confidence
        self.average_confidence = np.mean(self.confidence_history)
        
        # Temporal consistency based on normal vector stability
        if len(self.normal_history) > 1:
            angles = []
            for i in range(1, len(self.normal_history)):
                angle = GeometryUtils.compute_angle_between_vectors(
                    self.normal_history[i-1], self.normal_history[i]
                )
                angles.append(angle.item())
            
            # Consistency inversely related to angle variation
            angle_std = np.std(angles) if angles else 0.0
            self.temporal_consistency = max(0.0, 1.0 - angle_std / np.pi)
        else:
            self.temporal_consistency = 1.0
        
        # Stability score based on detection frequency and consistency
        detection_ratio = self.total_detections / max(1, self.last_seen - self.first_seen + 1)
        self.stability_score = (detection_ratio * 0.6 + 
                              self.temporal_consistency * 0.3 + 
                              self.average_confidence * 0.1)
    
    def get_smoothed_plane(self, alpha: float = 0.7) -> Optional[Plane]:
        """Get temporally smoothed plane representation."""
        if not self.planes:
            return None
        
        # Use most recent plane as base
        latest_plane = self.planes[-1]
        
        if len(self.planes) == 1:
            return latest_plane
        
        # Exponentially weighted smoothing
        weights = torch.tensor([alpha ** i for i in range(len(self.planes))][::-1])
        weights = weights / weights.sum()
        
        # Smooth normal vector
        normals_stack = torch.stack(self.normal_history)
        smoothed_normal = torch.sum(weights.unsqueeze(1) * normals_stack, dim=0)
        smoothed_normal = GeometryUtils.normalize_vector(smoothed_normal)
        
        # Smooth center point
        centers_stack = torch.stack(self.center_history)
        smoothed_center = torch.sum(weights.unsqueeze(1) * centers_stack, dim=0)
        
        # Weighted confidence and area
        smoothed_confidence = torch.sum(weights * torch.tensor(self.confidence_history)).item()
        smoothed_area = torch.sum(weights * torch.tensor(self.area_history)).item()
        
        # Create smoothed plane
        smoothed_plane = Plane(
            normal=smoothed_normal,
            point=smoothed_center,
            equation=torch.cat([smoothed_normal, -torch.dot(smoothed_normal, smoothed_center).unsqueeze(0)]),
            inlier_mask=latest_plane.inlier_mask,  # Use latest mask
            depth_values=latest_plane.depth_values,  # Use latest depth values
            rgb_values=latest_plane.rgb_values,  # Use latest RGB values
            confidence=smoothed_confidence,
            area=smoothed_area,
            stability_score=self.stability_score,
            plane_type=latest_plane.plane_type,
            quality=latest_plane.quality
        )
        
        return smoothed_plane
    
    def is_active(self, current_frame: int, max_missing_frames: int = 5) -> bool:
        """Check if track is still active."""
        return self.consecutive_misses <= max_missing_frames
    
    def get_predicted_position(self, frame_id: int) -> Optional[torch.Tensor]:
        """Predict plane center position for given frame."""
        if len(self.center_history) < 2:
            return self.center_history[-1] if self.center_history else None
        
        # Simple linear prediction based on recent motion
        recent_centers = self.center_history[-3:]  # Use last 3 positions
        if len(recent_centers) >= 2:
            velocity = recent_centers[-1] - recent_centers[-2]
            frames_ahead = frame_id - self.last_seen
            predicted_center = recent_centers[-1] + velocity * frames_ahead
            return predicted_center
        
        return self.center_history[-1]


class PlaneAssociator:
    """Associates detected planes across frames for temporal tracking."""
    
    def __init__(self, config: TemporalTrackingConfig):
        """
        Initialize plane associator.
        
        Args:
            config: Temporal tracking configuration
        """
        self.config = config
        
    def associate_planes(self, tracks: List[PlaneTrack], new_planes: List[Plane], 
                        frame_id: int) -> Tuple[Dict[int, int], List[int]]:
        """
        Associate new planes with existing tracks.
        
        Args:
            tracks: Existing plane tracks
            new_planes: Newly detected planes
            frame_id: Current frame ID
            
        Returns:
            Tuple of (track_to_plane_mapping, unmatched_plane_indices)
        """
        if not tracks or not new_planes:
            return {}, list(range(len(new_planes)))
        
        # Compute association cost matrix
        cost_matrix = self._compute_cost_matrix(tracks, new_planes, frame_id)
        
        # Perform assignment using Hungarian algorithm
        assignments = self._solve_assignment(cost_matrix)
        
        # Filter assignments by cost threshold
        track_to_plane = {}
        unmatched_planes = set(range(len(new_planes)))
        
        for track_idx, plane_idx in assignments.items():
            if track_idx < len(tracks) and plane_idx < len(new_planes):
                cost = cost_matrix[track_idx, plane_idx]
                if cost < self.config.association_threshold:
                    track_to_plane[track_idx] = plane_idx
                    unmatched_planes.discard(plane_idx)
        
        return track_to_plane, list(unmatched_planes)
    
    def _compute_cost_matrix(self, tracks: List[PlaneTrack], planes: List[Plane], 
                           frame_id: int) -> torch.Tensor:
        """Compute cost matrix for plane-track association."""
        num_tracks = len(tracks)
        num_planes = len(planes)
        
        cost_matrix = torch.full((num_tracks, num_planes), float('inf'))
        
        for t_idx, track in enumerate(tracks):
            if not track.is_active(frame_id, self.config.max_missing_frames):
                continue
            
            predicted_center = track.get_predicted_position(frame_id)
            if predicted_center is None:
                continue
            
            for p_idx, plane in enumerate(planes):
                # Compute multiple distance metrics
                costs = []
                
                # 1. Center distance
                center_dist = torch.norm(plane.point - predicted_center).item()
                costs.append(center_dist * self.config.center_weight)
                
                # 2. Normal angle distance
                if track.normal_history:
                    latest_normal = track.normal_history[-1]
                    angle_dist = GeometryUtils.compute_angle_between_vectors(
                        plane.normal, latest_normal
                    ).item()
                    costs.append(angle_dist * self.config.normal_weight)
                
                # 3. Area difference
                if track.area_history:
                    latest_area = track.area_history[-1]
                    area_diff = abs(plane.area - latest_area) / max(plane.area, latest_area)
                    costs.append(area_diff * self.config.area_weight)
                
                # 4. Type consistency
                if track.planes:
                    latest_type = track.planes[-1].plane_type
                    if plane.plane_type != latest_type:
                        costs.append(self.config.type_penalty)
                
                # Combined cost
                total_cost = sum(costs)
                cost_matrix[t_idx, p_idx] = total_cost
        
        return cost_matrix
    
    def _solve_assignment(self, cost_matrix: torch.Tensor) -> Dict[int, int]:
        """Solve assignment problem using Hungarian algorithm."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Convert to numpy for scipy
            cost_np = cost_matrix.cpu().numpy()
            
            # Solve assignment
            row_indices, col_indices = linear_sum_assignment(cost_np)
            
            # Convert back to dictionary
            assignments = {int(r): int(c) for r, c in zip(row_indices, col_indices)}
            
            return assignments
            
        except ImportError:
            logger.warning("scipy not available, using greedy assignment")
            return self._greedy_assignment(cost_matrix)
    
    def _greedy_assignment(self, cost_matrix: torch.Tensor) -> Dict[int, int]:
        """Greedy assignment as fallback when scipy not available."""
        assignments = {}
        used_planes = set()
        
        # Sort by cost and assign greedily
        num_tracks, num_planes = cost_matrix.shape
        
        for _ in range(min(num_tracks, num_planes)):
            # Find minimum cost assignment
            min_cost = float('inf')
            best_track = -1
            best_plane = -1
            
            for t in range(num_tracks):
                if t in assignments:
                    continue
                for p in range(num_planes):
                    if p in used_planes:
                        continue
                    if cost_matrix[t, p] < min_cost:
                        min_cost = cost_matrix[t, p]
                        best_track = t
                        best_plane = p
            
            if best_track >= 0 and best_plane >= 0:
                assignments[best_track] = best_plane
                used_planes.add(best_plane)
            else:
                break
        
        return assignments


class TemporalPlaneTracker:
    """Main temporal tracking system for planes across video frames."""
    
    def __init__(self, config: TemporalTrackingConfig):
        """
        Initialize temporal plane tracker.
        
        Args:
            config: Temporal tracking configuration
        """
        self.config = config
        config.validate()
        
        self.tracks: List[PlaneTrack] = []
        self.associator = PlaneAssociator(config)
        self.frame_id = 0
        self.next_track_id = 0
        
        # Statistics
        self.stats = {
            'total_tracks_created': 0,
            'total_tracks_terminated': 0,
            'current_active_tracks': 0,
            'average_track_length': 0.0,
            'association_rate': 0.0
        }
        
        logger.info("TemporalPlaneTracker initialized")
    
    def update(self, detected_planes: List[Plane]) -> List[PlaneTrack]:
        """
        Update tracker with new frame detections.
        
        Args:
            detected_planes: Planes detected in current frame
            
        Returns:
            List of active tracks with updated planes
        """
        # Associate detections with existing tracks
        track_assignments, unmatched_planes = self.associator.associate_planes(
            self.tracks, detected_planes, self.frame_id
        )
        
        # Update existing tracks
        for track_idx, plane_idx in track_assignments.items():
            self.tracks[track_idx].add_detection(detected_planes[plane_idx], self.frame_id)
        
        # Mark misses for unassociated tracks
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in track_assignments:
                track.mark_miss(self.frame_id)
        
        # Create new tracks for unmatched planes
        for plane_idx in unmatched_planes:
            new_track = PlaneTrack(track_id=self.next_track_id)
            new_track.add_detection(detected_planes[plane_idx], self.frame_id)
            self.tracks.append(new_track)
            self.next_track_id += 1
            self.stats['total_tracks_created'] += 1
        
        # Remove inactive tracks
        self._cleanup_tracks()
        
        # Update statistics
        self._update_statistics(track_assignments, detected_planes)
        
        self.frame_id += 1
        
        # Return active tracks
        active_tracks = [t for t in self.tracks if t.is_active(self.frame_id, self.config.max_missing_frames)]
        return active_tracks
    
    def get_smoothed_planes(self, min_stability: float = 0.3) -> List[Plane]:
        """
        Get temporally smoothed planes from stable tracks.
        
        Args:
            min_stability: Minimum stability score for inclusion
            
        Returns:
            List of smoothed planes
        """
        smoothed_planes = []
        
        for track in self.tracks:
            if (track.is_active(self.frame_id, self.config.max_missing_frames) and
                track.stability_score >= min_stability):
                
                smoothed_plane = track.get_smoothed_plane(self.config.smoothing_alpha)
                if smoothed_plane is not None:
                    # Add temporal information
                    smoothed_plane.temporal_stability = track.stability_score
                    smoothed_plane.temporal_consistency = track.temporal_consistency
                    smoothed_plane.track_id = track.track_id
                    smoothed_plane.track_length = track.total_detections
                    
                    smoothed_planes.append(smoothed_plane)
        
        return smoothed_planes
    
    def get_track_by_id(self, track_id: int) -> Optional[PlaneTrack]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return self.stats.copy()
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.frame_id = 0
        self.next_track_id = 0
        
        # Reset statistics
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
        
        logger.info("TemporalPlaneTracker reset")
    
    def _cleanup_tracks(self):
        """Remove inactive tracks from memory."""
        before_count = len(self.tracks)
        
        self.tracks = [
            track for track in self.tracks 
            if track.is_active(self.frame_id, self.config.max_missing_frames)
        ]
        
        terminated_count = before_count - len(self.tracks)
        self.stats['total_tracks_terminated'] += terminated_count
        
        # Memory management: limit total tracks
        if len(self.tracks) > self.config.max_tracks:
            # Keep tracks with highest stability scores
            self.tracks.sort(key=lambda t: t.stability_score, reverse=True)
            self.tracks = self.tracks[:self.config.max_tracks]
    
    def _update_statistics(self, assignments: Dict[int, int], detected_planes: List[Plane]):
        """Update tracking statistics."""
        self.stats['current_active_tracks'] = len([
            t for t in self.tracks 
            if t.is_active(self.frame_id, self.config.max_missing_frames)
        ])
        
        # Association rate
        if detected_planes:
            self.stats['association_rate'] = len(assignments) / len(detected_planes)
        
        # Average track length
        if self.tracks:
            total_length = sum(t.total_detections for t in self.tracks)
            self.stats['average_track_length'] = total_length / len(self.tracks)


class TemporalConsistencyValidator:
    """Validates temporal consistency of plane detections."""
    
    def __init__(self, config: TemporalTrackingConfig):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_track_consistency(self, track: PlaneTrack) -> Dict[str, float]:
        """
        Validate temporal consistency of a plane track.
        
        Args:
            track: Plane track to validate
            
        Returns:
            Dictionary of consistency metrics
        """
        metrics = {
            'normal_consistency': 0.0,
            'center_consistency': 0.0,
            'area_consistency': 0.0,
            'confidence_trend': 0.0,
            'overall_consistency': 0.0
        }
        
        if len(track.planes) < 2:
            return metrics
        
        # Normal vector consistency
        if len(track.normal_history) > 1:
            angles = []
            for i in range(1, len(track.normal_history)):
                angle = GeometryUtils.compute_angle_between_vectors(
                    track.normal_history[i-1], track.normal_history[i]
                )
                angles.append(angle.item())
            
            angle_var = np.var(angles) if angles else 0.0
            metrics['normal_consistency'] = max(0.0, 1.0 - angle_var / (np.pi/4)**2)
        
        # Center position consistency
        if len(track.center_history) > 1:
            center_distances = []
            for i in range(1, len(track.center_history)):
                dist = torch.norm(track.center_history[i] - track.center_history[i-1]).item()
                center_distances.append(dist)
            
            center_var = np.var(center_distances) if center_distances else 0.0
            metrics['center_consistency'] = max(0.0, 1.0 - center_var / 1.0)  # Assume 1m is high variance
        
        # Area consistency
        if len(track.area_history) > 1:
            area_ratios = []
            for i in range(1, len(track.area_history)):
                ratio = track.area_history[i] / max(track.area_history[i-1], 1e-6)
                area_ratios.append(abs(1.0 - ratio))  # Deviation from no change
            
            area_var = np.var(area_ratios) if area_ratios else 0.0
            metrics['area_consistency'] = max(0.0, 1.0 - area_var / 0.25)  # 25% variance threshold
        
        # Confidence trend (increasing confidence is better)
        if len(track.confidence_history) > 1:
            confidence_diff = track.confidence_history[-1] - track.confidence_history[0]
            metrics['confidence_trend'] = max(0.0, min(1.0, 0.5 + confidence_diff))
        
        # Overall consistency (weighted average)
        weights = [0.4, 0.3, 0.2, 0.1]  # Normal, center, area, confidence
        metrics['overall_consistency'] = sum(
            w * metrics[k] for w, k in zip(weights, 
            ['normal_consistency', 'center_consistency', 'area_consistency', 'confidence_trend'])
        )
        
        return metrics
    
    def filter_consistent_tracks(self, tracks: List[PlaneTrack], 
                               min_consistency: float = 0.5) -> List[PlaneTrack]:
        """Filter tracks by temporal consistency."""
        consistent_tracks = []
        
        for track in tracks:
            metrics = self.validate_track_consistency(track)
            if metrics['overall_consistency'] >= min_consistency:
                consistent_tracks.append(track)
        
        return consistent_tracks
    
    def interpolate_missing_detections(self, track: PlaneTrack, 
                                     missing_frames: List[int]) -> List[Plane]:
        """Interpolate plane detections for missing frames."""
        if len(track.planes) < 2:
            return []
        
        interpolated_planes = []
        
        for frame_id in missing_frames:
            # Find surrounding detections
            before_detection = None
            after_detection = None
            
            for i, plane in enumerate(track.planes):
                if track.last_seen < frame_id:
                    before_detection = plane
                elif track.first_seen > frame_id:
                    after_detection = plane
                    break
            
            if before_detection is None or after_detection is None:
                continue
            
            # Linear interpolation
            alpha = 0.5  # Simple midpoint interpolation
            
            interpolated_normal = GeometryUtils.normalize_vector(
                (1 - alpha) * before_detection.normal + alpha * after_detection.normal
            )
            interpolated_center = (1 - alpha) * before_detection.point + alpha * after_detection.point
            interpolated_area = (1 - alpha) * before_detection.area + alpha * after_detection.area
            interpolated_confidence = (1 - alpha) * before_detection.confidence + alpha * after_detection.confidence
            
            # Create interpolated plane
            interpolated_plane = Plane(
                normal=interpolated_normal,
                point=interpolated_center,
                equation=torch.cat([interpolated_normal, -torch.dot(interpolated_normal, interpolated_center).unsqueeze(0)]),
                inlier_mask=before_detection.inlier_mask,  # Use template mask
                confidence=interpolated_confidence,
                area=interpolated_area,
                stability_score=track.stability_score,
                plane_type=before_detection.plane_type,
                quality=before_detection.quality
            )
            interpolated_plane.is_interpolated = True
            
            interpolated_planes.append(interpolated_plane)
        
        return interpolated_planes 