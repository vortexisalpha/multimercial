"""
Tracking Evaluation and Metrics for Multi-Object Tracking

This module provides comprehensive evaluation metrics for tracking performance,
including MOT metrics, trajectory analysis, and advertisement-specific metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .track_models import Track
from ..detection_models import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackingMetrics:
    """Core tracking performance metrics."""
    
    # MOT metrics
    mota: float = 0.0  # Multiple Object Tracking Accuracy
    motp: float = 0.0  # Multiple Object Tracking Precision
    idf1: float = 0.0  # ID F1 Score
    
    # Detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Counting metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    
    # Track quality metrics
    mostly_tracked: int = 0  # Tracks covered >80% of their groundtruth
    partially_tracked: int = 0  # Tracks covered 20-80% of their groundtruth
    mostly_lost: int = 0  # Tracks covered <20% of their groundtruth
    
    # Fragmentation metrics
    fragments: int = 0  # Number of track fragmentations
    
    # Advertisement-specific metrics
    avg_track_length: float = 0.0
    track_stability: float = 0.0
    advertisement_relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'mota': self.mota,
            'motp': self.motp,
            'idf1': self.idf1,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'id_switches': self.id_switches,
            'mostly_tracked': self.mostly_tracked,
            'partially_tracked': self.partially_tracked,
            'mostly_lost': self.mostly_lost,
            'fragments': self.fragments,
            'avg_track_length': self.avg_track_length,
            'track_stability': self.track_stability,
            'advertisement_relevance_score': self.advertisement_relevance_score
        }


@dataclass
class MOTMetrics:
    """MOT challenge specific metrics."""
    
    # Core MOT metrics
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    
    # Detection metrics
    recall: float = 0.0
    precision: float = 0.0
    
    # Assignment metrics
    idsw: int = 0  # ID switches
    frag: int = 0  # Fragmentations
    
    # Count metrics
    gt: int = 0    # Ground truth objects
    tp: int = 0    # True positives
    fp: int = 0    # False positives
    fn: int = 0    # False negatives
    
    # Track analysis
    mt: int = 0    # Mostly tracked
    pt: int = 0    # Partially tracked
    ml: int = 0    # Mostly lost
    
    def compute_derived_metrics(self):
        """Compute derived metrics from basic counts."""
        # Precision and Recall
        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        
        if self.tp + self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        
        # MOTA
        if self.gt > 0:
            self.mota = 1 - (self.fn + self.fp + self.idsw) / self.gt
        
        # MOTP (if distance information available)
        # This would typically be computed during evaluation


class TrackingEvaluator:
    """Comprehensive tracking evaluation system."""
    
    def __init__(self, iou_threshold: float = 0.5, 
                 distance_threshold: float = 1.0):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for positive matches
            distance_threshold: Distance threshold for matching
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        # Evaluation state
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.frame_metrics = []
        self.track_matches = defaultdict(dict)  # track_id -> {frame_id -> gt_id}
        self.gt_track_matches = defaultdict(dict)  # gt_id -> {frame_id -> track_id}
        self.track_trajectories = defaultdict(list)
        self.gt_trajectories = defaultdict(list)
    
    def evaluate_frame(self, predicted_tracks: List[Track], 
                      ground_truth: List[Detection], 
                      frame_id: int) -> Dict[str, Any]:
        """
        Evaluate tracking performance for a single frame.
        
        Args:
            predicted_tracks: List of predicted tracks
            ground_truth: List of ground truth detections
            frame_id: Frame identifier
            
        Returns:
            Frame-level metrics
        """
        # Convert tracks to detections for easier processing
        predictions = []
        for track in predicted_tracks:
            if track.is_confirmed_track():
                bbox = track.get_current_bbox()
                if bbox is not None:
                    detection = Detection(
                        bbox=bbox,
                        confidence=track.metadata.avg_confidence,
                        class_id=track.metadata.object_class_id,
                        class_name=track.metadata.object_class,
                        track_id=track.track_id,
                        frame_id=frame_id
                    )
                    predictions.append(detection)
        
        # Compute association between predictions and ground truth
        matches, fp_indices, fn_indices = self._associate_frame(predictions, ground_truth)
        
        # Count true positives, false positives, false negatives
        tp = len(matches)
        fp = len(fp_indices)
        fn = len(fn_indices)
        
        # Track matches for ID consistency analysis
        for pred_idx, gt_idx in matches:
            pred_track_id = predictions[pred_idx].track_id
            gt_track_id = ground_truth[gt_idx].track_id if hasattr(ground_truth[gt_idx], 'track_id') else gt_idx
            
            self.track_matches[pred_track_id][frame_id] = gt_track_id
            self.gt_track_matches[gt_track_id][frame_id] = pred_track_id
        
        # Store trajectories
        for pred in predictions:
            if hasattr(pred, 'track_id'):
                self.track_trajectories[pred.track_id].append((frame_id, pred.bbox))
        
        for i, gt in enumerate(ground_truth):
            gt_id = gt.track_id if hasattr(gt, 'track_id') else i
            self.gt_trajectories[gt_id].append((frame_id, gt.bbox))
        
        frame_metrics = {
            'frame_id': frame_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth)
        }
        
        self.frame_metrics.append(frame_metrics)
        return frame_metrics
    
    def compute_summary_metrics(self) -> TrackingMetrics:
        """Compute summary tracking metrics across all frames."""
        if not self.frame_metrics:
            return TrackingMetrics()
        
        # Aggregate counts
        total_tp = sum(m['tp'] for m in self.frame_metrics)
        total_fp = sum(m['fp'] for m in self.frame_metrics)
        total_fn = sum(m['fn'] for m in self.frame_metrics)
        
        # Compute ID switches
        id_switches = self._compute_id_switches()
        
        # Compute track coverage
        mostly_tracked, partially_tracked, mostly_lost = self._compute_track_coverage()
        
        # Compute fragmentations
        fragments = self._compute_fragmentations()
        
        # Basic metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # MOTA calculation
        total_gt = sum(m['num_ground_truth'] for m in self.frame_metrics)
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0
        
        # MOTP calculation (simplified - would need distance information)
        motp = precision  # Simplified version
        
        # Advertisement-specific metrics
        avg_track_length = self._compute_average_track_length()
        track_stability = self._compute_track_stability()
        ad_relevance = self._compute_advertisement_relevance()
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=f1_score,  # Simplified
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            id_switches=id_switches,
            mostly_tracked=mostly_tracked,
            partially_tracked=partially_tracked,
            mostly_lost=mostly_lost,
            fragments=fragments,
            avg_track_length=avg_track_length,
            track_stability=track_stability,
            advertisement_relevance_score=ad_relevance
        )
    
    def compute_mot_metrics(self) -> MOTMetrics:
        """Compute MOT challenge metrics."""
        if not self.frame_metrics:
            return MOTMetrics()
        
        # Aggregate counts
        total_tp = sum(m['tp'] for m in self.frame_metrics)
        total_fp = sum(m['fp'] for m in self.frame_metrics)
        total_fn = sum(m['fn'] for m in self.frame_metrics)
        total_gt = sum(m['num_ground_truth'] for m in self.frame_metrics)
        
        # Compute advanced metrics
        id_switches = self._compute_id_switches()
        fragments = self._compute_fragmentations()
        mostly_tracked, partially_tracked, mostly_lost = self._compute_track_coverage()
        
        mot_metrics = MOTMetrics(
            gt=total_gt,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            idsw=id_switches,
            frag=fragments,
            mt=mostly_tracked,
            pt=partially_tracked,
            ml=mostly_lost
        )
        
        # Compute derived metrics
        mot_metrics.compute_derived_metrics()
        
        return mot_metrics
    
    def _associate_frame(self, predictions: List[Detection], 
                        ground_truth: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate predictions with ground truth for a frame."""
        if not predictions or not ground_truth:
            return [], list(range(len(predictions))), list(range(len(ground_truth)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                iou = self._compute_iou(pred.bbox, gt.bbox)
                iou_matrix[i, j] = iou
        
        # Find matches using Hungarian algorithm (simplified greedy approach)
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Sort potential matches by IoU
        potential_matches = []
        for i in range(len(predictions)):
            for j in range(len(ground_truth)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    potential_matches.append((iou_matrix[i, j], i, j))
        
        potential_matches.sort(reverse=True)
        
        # Greedily assign matches
        for _, i, j in potential_matches:
            if i not in used_pred and j not in used_gt:
                matches.append((i, j))
                used_pred.add(i)
                used_gt.add(j)
        
        # Find unmatched
        fp_indices = [i for i in range(len(predictions)) if i not in used_pred]
        fn_indices = [j for j in range(len(ground_truth)) if j not in used_gt]
        
        return matches, fp_indices, fn_indices
    
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], 
                    bbox2: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_id_switches(self) -> int:
        """Compute number of ID switches."""
        id_switches = 0
        
        for track_id, frame_matches in self.track_matches.items():
            if len(frame_matches) < 2:
                continue
            
            sorted_frames = sorted(frame_matches.items())
            prev_gt_id = sorted_frames[0][1]
            
            for frame_id, gt_id in sorted_frames[1:]:
                if gt_id != prev_gt_id:
                    id_switches += 1
                prev_gt_id = gt_id
        
        return id_switches
    
    def _compute_track_coverage(self) -> Tuple[int, int, int]:
        """Compute track coverage statistics."""
        mostly_tracked = 0
        partially_tracked = 0
        mostly_lost = 0
        
        for gt_id, frame_matches in self.gt_track_matches.items():
            if not frame_matches:
                mostly_lost += 1
                continue
            
            # Get ground truth trajectory length
            gt_frames = [frame_id for frame_id, _ in self.gt_trajectories[gt_id]]
            total_gt_frames = len(gt_frames)
            
            if total_gt_frames == 0:
                continue
            
            # Count tracked frames
            tracked_frames = len(frame_matches)
            coverage = tracked_frames / total_gt_frames
            
            if coverage >= 0.8:
                mostly_tracked += 1
            elif coverage >= 0.2:
                partially_tracked += 1
            else:
                mostly_lost += 1
        
        return mostly_tracked, partially_tracked, mostly_lost
    
    def _compute_fragmentations(self) -> int:
        """Compute number of track fragmentations."""
        fragments = 0
        
        for track_id, trajectory in self.track_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            # Sort by frame ID
            trajectory.sort(key=lambda x: x[0])
            
            # Count gaps in trajectory
            for i in range(1, len(trajectory)):
                frame_gap = trajectory[i][0] - trajectory[i-1][0]
                if frame_gap > 1:  # Gap detected
                    fragments += 1
        
        return fragments
    
    def _compute_average_track_length(self) -> float:
        """Compute average track length."""
        if not self.track_trajectories:
            return 0.0
        
        total_length = sum(len(trajectory) for trajectory in self.track_trajectories.values())
        return total_length / len(self.track_trajectories)
    
    def _compute_track_stability(self) -> float:
        """Compute track stability score."""
        if not self.track_trajectories:
            return 0.0
        
        stability_scores = []
        
        for trajectory in self.track_trajectories.values():
            if len(trajectory) < 2:
                stability_scores.append(0.0)
                continue
            
            # Sort by frame ID
            trajectory.sort(key=lambda x: x[0])
            
            # Compute velocity consistency
            velocities = []
            for i in range(1, len(trajectory)):
                frame1, bbox1 = trajectory[i-1]
                frame2, bbox2 = trajectory[i]
                
                center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                
                velocity = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
                velocities.append(velocity)
            
            if velocities:
                # Stability is inverse of velocity variance
                velocity_std = np.std(velocities)
                stability = 1.0 / (1.0 + velocity_std)
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _compute_advertisement_relevance(self) -> float:
        """Compute advertisement relevance score."""
        # This would be computed based on track properties like:
        # - Object classes (person, furniture, etc.)
        # - Track stability
        # - Track duration
        # - Visibility
        
        # Placeholder implementation
        return 0.7
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed evaluation report."""
        summary_metrics = self.compute_summary_metrics()
        mot_metrics = self.compute_mot_metrics()
        
        report = {
            'summary_metrics': summary_metrics.to_dict(),
            'mot_metrics': mot_metrics.__dict__,
            'frame_level_metrics': self.frame_metrics,
            'track_analysis': {
                'total_tracks': len(self.track_trajectories),
                'total_gt_tracks': len(self.gt_trajectories),
                'avg_track_length': summary_metrics.avg_track_length,
                'track_stability': summary_metrics.track_stability,
                'id_switches': summary_metrics.id_switches,
                'fragmentations': summary_metrics.fragments
            },
            'temporal_analysis': {
                'total_frames': len(self.frame_metrics),
                'frames_with_detections': sum(1 for m in self.frame_metrics if m['num_predictions'] > 0),
                'frames_with_gt': sum(1 for m in self.frame_metrics if m['num_ground_truth'] > 0)
            }
        }
        
        return report 