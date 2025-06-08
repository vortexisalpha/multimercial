"""
Track Association Algorithms for Multi-Object Tracking

This module implements various association algorithms for matching detections
to tracks, including the Hungarian algorithm and distance-based methods.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
from abc import ABC, abstractmethod
import logging

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AssociationMethod(Enum):
    """Association algorithm types."""
    HUNGARIAN = "hungarian"
    GREEDY = "greedy"
    AUCTION = "auction"


class BaseAssociator(ABC):
    """Base class for association algorithms."""
    
    @abstractmethod
    def associate(self, cost_matrix: np.ndarray, 
                 threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks and detections based on cost matrix.
        
        Args:
            cost_matrix: Cost matrix (tracks x detections)
            threshold: Association threshold
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        pass


class HungarianAssociator(BaseAssociator):
    """Hungarian algorithm for optimal assignment."""
    
    def associate(self, cost_matrix: np.ndarray, 
                 threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate using Hungarian algorithm.
        
        Args:
            cost_matrix: Cost matrix where higher values indicate better matches
            threshold: Minimum cost threshold for valid associations
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # Convert to minimization problem (Hungarian expects costs, not similarities)
        cost_matrix_min = 1.0 - cost_matrix
        
        if SCIPY_AVAILABLE:
            # Use scipy's implementation
            track_indices, detection_indices = linear_sum_assignment(cost_matrix_min)
            
            # Filter by threshold
            matched_pairs = []
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, det_idx] >= threshold:
                    matched_pairs.append((track_idx, det_idx))
            
            # Find unmatched tracks and detections
            matched_track_indices = {pair[0] for pair in matched_pairs}
            matched_detection_indices = {pair[1] for pair in matched_pairs}
            
            unmatched_tracks = [i for i in range(cost_matrix.shape[0]) 
                              if i not in matched_track_indices]
            unmatched_detections = [i for i in range(cost_matrix.shape[1]) 
                                  if i not in matched_detection_indices]
            
            return matched_pairs, unmatched_tracks, unmatched_detections
        
        else:
            # Fallback to greedy assignment
            logger.warning("scipy not available, using greedy assignment")
            return self._greedy_assignment(cost_matrix, threshold)
    
    def _greedy_assignment(self, cost_matrix: np.ndarray, 
                          threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy assignment fallback."""
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        # Find best matches iteratively
        while True:
            # Find best remaining match
            best_score = -1
            best_match = None
            
            for i in range(cost_matrix.shape[0]):
                if i in used_tracks:
                    continue
                for j in range(cost_matrix.shape[1]):
                    if j in used_detections:
                        continue
                    
                    if cost_matrix[i, j] > best_score and cost_matrix[i, j] >= threshold:
                        best_score = cost_matrix[i, j]
                        best_match = (i, j)
            
            if best_match is None:
                break
            
            matched_pairs.append(best_match)
            used_tracks.add(best_match[0])
            used_detections.add(best_match[1])
        
        # Find unmatched
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections


class GreedyAssociator(BaseAssociator):
    """Greedy association algorithm."""
    
    def associate(self, cost_matrix: np.ndarray, 
                 threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy association based on highest scores first.
        
        Args:
            cost_matrix: Cost matrix where higher values indicate better matches
            threshold: Minimum score threshold for valid associations
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # Get all potential matches above threshold
        potential_matches = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] >= threshold:
                    potential_matches.append((cost_matrix[i, j], i, j))
        
        # Sort by score (descending)
        potential_matches.sort(reverse=True)
        
        # Greedily select matches
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        for score, track_idx, det_idx in potential_matches:
            if track_idx not in used_tracks and det_idx not in used_detections:
                matched_pairs.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # Find unmatched
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections


class AuctionAssociator(BaseAssociator):
    """Auction algorithm for assignment (simplified implementation)."""
    
    def __init__(self, max_iterations: int = 100, epsilon: float = 1e-3):
        self.max_iterations = max_iterations
        self.epsilon = epsilon
    
    def associate(self, cost_matrix: np.ndarray, 
                 threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Auction algorithm association.
        
        Args:
            cost_matrix: Cost matrix where higher values indicate better matches
            threshold: Minimum score threshold for valid associations
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # For simplicity, fall back to greedy for now
        # A full auction algorithm implementation would be more complex
        greedy_associator = GreedyAssociator()
        return greedy_associator.associate(cost_matrix, threshold)


class TrackAssociation:
    """Main interface for track association."""
    
    def __init__(self, method: AssociationMethod = AssociationMethod.HUNGARIAN):
        """
        Initialize track association.
        
        Args:
            method: Association method to use
        """
        self.method = method
        self.associator = self._create_associator(method)
        
        logger.info(f"Track association initialized with method: {method.value}")
    
    def associate(self, cost_matrix: np.ndarray, 
                 threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks and detections.
        
        Args:
            cost_matrix: Cost matrix (tracks x detections)
            threshold: Association threshold
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        try:
            return self.associator.associate(cost_matrix, threshold)
        except Exception as e:
            logger.error(f"Association failed with {self.method.value}: {e}")
            # Fallback to greedy
            if self.method != AssociationMethod.GREEDY:
                logger.info("Falling back to greedy association")
                greedy_associator = GreedyAssociator()
                return greedy_associator.associate(cost_matrix, threshold)
            else:
                raise
    
    def _create_associator(self, method: AssociationMethod) -> BaseAssociator:
        """Create associator instance based on method."""
        if method == AssociationMethod.HUNGARIAN:
            return HungarianAssociator()
        elif method == AssociationMethod.GREEDY:
            return GreedyAssociator()
        elif method == AssociationMethod.AUCTION:
            return AuctionAssociator()
        else:
            raise ValueError(f"Unknown association method: {method}")


class MultiStageAssociation:
    """Multi-stage association for complex scenarios."""
    
    def __init__(self, stages: List[Dict]):
        """
        Initialize multi-stage association.
        
        Args:
            stages: List of stage configurations, each containing:
                - method: AssociationMethod
                - threshold: float
                - cost_function: Optional callable for custom cost computation
        """
        self.stages = stages
        self.associators = [TrackAssociation(stage['method']) for stage in stages]
    
    def associate(self, tracks, detections, 
                 cost_functions: List[callable]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Multi-stage association process.
        
        Args:
            tracks: List of tracks
            detections: List of detections
            cost_functions: List of functions to compute cost matrices for each stage
            
        Returns:
            Final association results
        """
        all_matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for stage_idx, (stage, associator) in enumerate(zip(self.stages, self.associators)):
            if not unmatched_tracks or not unmatched_detections:
                break
            
            # Get remaining tracks and detections
            remaining_tracks = [tracks[i] for i in unmatched_tracks]
            remaining_detections = [detections[i] for i in unmatched_detections]
            
            # Compute cost matrix for this stage
            if stage_idx < len(cost_functions):
                cost_matrix = cost_functions[stage_idx](remaining_tracks, remaining_detections)
            else:
                # Default IoU-based cost
                cost_matrix = self._default_cost_matrix(remaining_tracks, remaining_detections)
            
            # Associate for this stage
            stage_matches, stage_unmatched_tracks, stage_unmatched_dets = associator.associate(
                cost_matrix, stage['threshold']
            )
            
            # Convert back to original indices
            for track_idx, det_idx in stage_matches:
                original_track_idx = unmatched_tracks[track_idx]
                original_det_idx = unmatched_detections[det_idx]
                all_matched_pairs.append((original_track_idx, original_det_idx))
            
            # Update unmatched lists
            unmatched_tracks = [unmatched_tracks[i] for i in stage_unmatched_tracks]
            unmatched_detections = [unmatched_detections[i] for i in stage_unmatched_dets]
        
        return all_matched_pairs, unmatched_tracks, unmatched_detections
    
    def _default_cost_matrix(self, tracks, detections) -> np.ndarray:
        """Default IoU-based cost matrix."""
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_bbox = track.get_current_bbox() if hasattr(track, 'get_current_bbox') else None
            if track_bbox is None:
                continue
                
            for j, detection in enumerate(detections):
                det_bbox = detection.bbox if hasattr(detection, 'bbox') else None
                if det_bbox is None:
                    continue
                
                iou = self._compute_iou(track_bbox, det_bbox)
                cost_matrix[i, j] = iou
        
        return cost_matrix
    
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], 
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


# Utility functions for different cost computations

def compute_iou_cost_matrix(tracks, detections) -> np.ndarray:
    """Compute IoU-based cost matrix."""
    associator = TrackAssociation()
    return associator._default_cost_matrix(tracks, detections)


def compute_distance_cost_matrix(tracks, detections, max_distance: float = 100.0) -> np.ndarray:
    """Compute Euclidean distance-based cost matrix."""
    cost_matrix = np.zeros((len(tracks), len(detections)))
    
    for i, track in enumerate(tracks):
        track_center = track.get_center() if hasattr(track, 'get_center') else None
        if track_center is None:
            continue
            
        for j, detection in enumerate(detections):
            det_center = detection.center() if hasattr(detection, 'center') else None
            if det_center is None:
                continue
            
            distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                             (track_center[1] - det_center[1])**2)
            
            # Convert distance to similarity score
            similarity = max(0.0, 1.0 - distance / max_distance)
            cost_matrix[i, j] = similarity
    
    return cost_matrix


def compute_combined_cost_matrix(tracks, detections, 
                               iou_weight: float = 0.7,
                               distance_weight: float = 0.3) -> np.ndarray:
    """Compute combined IoU and distance cost matrix."""
    iou_matrix = compute_iou_cost_matrix(tracks, detections)
    distance_matrix = compute_distance_cost_matrix(tracks, detections)
    
    # Normalize weights
    total_weight = iou_weight + distance_weight
    iou_weight /= total_weight
    distance_weight /= total_weight
    
    combined_matrix = iou_weight * iou_matrix + distance_weight * distance_matrix
    return combined_matrix 