"""
Visualization Tools for Multi-Object Tracking

This module provides comprehensive visualization capabilities for tracking
analysis, debugging, and performance evaluation in video advertisement scenarios.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import json

from .track_models import Track, TrackState
from ..detection_models import Detection

logger = logging.getLogger(__name__)


class TrackingVisualizer:
    """Comprehensive tracking visualization system."""
    
    def __init__(self, frame_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize visualizer.
        
        Args:
            frame_size: Default frame size (width, height)
        """
        self.frame_size = frame_size
        
        # Color schemes
        self.track_colors = self._generate_colors(100)
        self.state_colors = {
            TrackState.NEW: (0, 255, 255),      # Yellow
            TrackState.TRACKED: (0, 255, 0),    # Green  
            TrackState.LOST: (0, 165, 255),     # Orange
            TrackState.REMOVED: (0, 0, 255),    # Red
            TrackState.CONFIRMED: (255, 0, 255) # Magenta
        }
        
        # Visualization settings
        self.line_thickness = 2
        self.font_scale = 0.6
        self.font_thickness = 2
        self.trail_length = 30
        
    def visualize_frame(self, frame: np.ndarray, 
                       tracks: List[Track], 
                       detections: Optional[List[Detection]] = None,
                       show_trails: bool = True,
                       show_predictions: bool = False,
                       show_metadata: bool = True) -> np.ndarray:
        """
        Visualize tracking results on a frame.
        
        Args:
            frame: Input frame image
            tracks: List of tracks to visualize
            detections: Optional raw detections
            show_trails: Whether to show track trails
            show_predictions: Whether to show predicted positions
            show_metadata: Whether to show track metadata
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw raw detections if provided
        if detections:
            vis_frame = self._draw_detections(vis_frame, detections)
        
        # Draw tracks
        for track in tracks:
            vis_frame = self._draw_track(
                vis_frame, track, 
                show_trails=show_trails,
                show_predictions=show_predictions,
                show_metadata=show_metadata
            )
        
        # Add frame-level information
        vis_frame = self._add_frame_info(vis_frame, tracks, detections)
        
        return vis_frame
    
    def create_tracking_video(self, frames: List[np.ndarray],
                            all_tracks: List[List[Track]],
                            output_path: str,
                            fps: int = 30) -> bool:
        """
        Create tracking visualization video.
        
        Args:
            frames: List of video frames
            all_tracks: List of track lists for each frame
            output_path: Output video path
            fps: Video frame rate
            
        Returns:
            True if successful
        """
        try:
            if not frames or not all_tracks:
                logger.error("No frames or tracks provided")
                return False
            
            # Get video properties
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame, tracks in zip(frames, all_tracks):
                # Visualize frame
                vis_frame = self.visualize_frame(frame, tracks)
                
                # Write frame
                out.write(vis_frame)
            
            out.release()
            logger.info(f"Tracking video saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tracking video: {e}")
            return False
    
    def plot_tracking_metrics(self, metrics_history: List[Dict],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot tracking metrics over time.
        
        Args:
            metrics_history: List of frame metrics
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tracking Performance Metrics', fontsize=16)
        
        frames = [m['frame_id'] for m in metrics_history]
        
        # Precision and Recall
        precision = [m.get('precision', 0) for m in metrics_history]
        recall = [m.get('recall', 0) for m in metrics_history]
        
        axes[0, 0].plot(frames, precision, label='Precision', color='blue')
        axes[0, 0].plot(frames, recall, label='Recall', color='red')
        axes[0, 0].set_title('Precision and Recall')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Detection counts
        tp = [m.get('tp', 0) for m in metrics_history]
        fp = [m.get('fp', 0) for m in metrics_history]
        fn = [m.get('fn', 0) for m in metrics_history]
        
        axes[0, 1].plot(frames, tp, label='True Positives', color='green')
        axes[0, 1].plot(frames, fp, label='False Positives', color='red')
        axes[0, 1].plot(frames, fn, label='False Negatives', color='orange')
        axes[0, 1].set_title('Detection Counts')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Track counts
        num_predictions = [m.get('num_predictions', 0) for m in metrics_history]
        num_ground_truth = [m.get('num_ground_truth', 0) for m in metrics_history]
        
        axes[1, 0].plot(frames, num_predictions, label='Predicted Tracks', color='blue')
        axes[1, 0].plot(frames, num_ground_truth, label='Ground Truth', color='green')
        axes[1, 0].set_title('Track Counts')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1 Score
        f1_scores = []
        for m in metrics_history:
            p = m.get('precision', 0)
            r = m.get('recall', 0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        
        axes[1, 1].plot(frames, f1_scores, label='F1 Score', color='purple')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to: {save_path}")
        
        return fig
    
    def plot_trajectory_analysis(self, tracks: List[Track],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trajectory analysis for tracks.
        
        Args:
            tracks: List of tracks to analyze
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Track Trajectory Analysis', fontsize=16)
        
        # Trajectory visualization
        for track in tracks:
            if track.bbox_history:
                centers = []
                for bbox in track.bbox_history:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centers.append((center_x, center_y))
                
                if centers:
                    x_coords = [c[0] for c in centers]
                    y_coords = [c[1] for c in centers]
                    
                    color = self.track_colors[track.track_id % len(self.track_colors)]
                    color_norm = (color[2]/255, color[1]/255, color[0]/255)  # BGR to RGB
                    
                    axes[0, 0].plot(x_coords, y_coords, color=color_norm, alpha=0.7)
                    axes[0, 0].scatter(x_coords[0], y_coords[0], color=color_norm, marker='o', s=20)
                    axes[0, 0].scatter(x_coords[-1], y_coords[-1], color=color_norm, marker='s', s=20)
        
        axes[0, 0].set_title('Track Trajectories')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        axes[0, 0].invert_yaxis()  # Invert Y-axis to match image coordinates
        axes[0, 0].grid(True)
        
        # Track length distribution
        track_lengths = [len(track.bbox_history) for track in tracks if track.bbox_history]
        if track_lengths:
            axes[0, 1].hist(track_lengths, bins=20, alpha=0.7, color='blue')
            axes[0, 1].set_title('Track Length Distribution')
            axes[0, 1].set_xlabel('Track Length (frames)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True)
        
        # Confidence distribution
        confidences = []
        for track in tracks:
            if track.confidence_history:
                confidences.extend(track.confidence_history)
        
        if confidences:
            axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True)
        
        # Track state distribution
        state_counts = {}
        for track in tracks:
            state = track.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        if state_counts:
            states = list(state_counts.keys())
            counts = list(state_counts.values())
            colors = ['blue', 'green', 'orange', 'red', 'purple'][:len(states)]
            
            axes[1, 1].pie(counts, labels=states, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('Track State Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory analysis saved to: {save_path}")
        
        return fig
    
    def create_track_heatmap(self, tracks: List[Track],
                           frame_size: Optional[Tuple[int, int]] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Create heatmap of track density.
        
        Args:
            tracks: List of tracks
            frame_size: Frame size (width, height)
            save_path: Optional path to save heatmap
            
        Returns:
            Heatmap image
        """
        if frame_size is None:
            frame_size = self.frame_size
        
        width, height = frame_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate track positions
        for track in tracks:
            for bbox in track.bbox_history:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Clamp coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                if x2 > x1 and y2 > y1:
                    heatmap[y1:y2, x1:x2] += 1
        
        # Normalize and apply colormap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply Gaussian blur for smoothing
        heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_blurred * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        if save_path:
            cv2.imwrite(save_path, heatmap_colored)
            logger.info(f"Track heatmap saved to: {save_path}")
        
        return heatmap_colored
    
    def _draw_track(self, frame: np.ndarray, track: Track,
                   show_trails: bool = True,
                   show_predictions: bool = False,
                   show_metadata: bool = True) -> np.ndarray:
        """Draw a single track on the frame."""
        if not track.bbox_history:
            return frame
        
        track_color = self.track_colors[track.track_id % len(self.track_colors)]
        state_color = self.state_colors.get(track.state, (255, 255, 255))
        
        # Draw current bounding box
        current_bbox = track.bbox_history[-1]
        x1, y1, x2, y2 = map(int, current_bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, self.line_thickness)
        
        # Draw state indicator (small rectangle in top-left corner)
        cv2.rectangle(frame, (x1, y1), (x1 + 20, y1 + 10), state_color, -1)
        
        # Draw track ID
        text = f"ID:{track.track_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0] + 5, y1), track_color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 
                   self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Draw trail
        if show_trails and len(track.bbox_history) > 1:
            trail_points = []
            for bbox in track.bbox_history[-self.trail_length:]:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                trail_points.append((center_x, center_y))
            
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                thickness = max(1, int(self.line_thickness * alpha))
                cv2.line(frame, trail_points[i-1], trail_points[i], track_color, thickness)
        
        # Draw predicted position
        if show_predictions and track.kalman_filter is not None:
            # This would require implementing prediction visualization
            pass
        
        # Draw metadata
        if show_metadata:
            metadata_text = []
            
            if hasattr(track.metadata, 'object_class') and track.metadata.object_class:
                metadata_text.append(f"Class: {track.metadata.object_class}")
            
            if hasattr(track.metadata, 'avg_confidence'):
                metadata_text.append(f"Conf: {track.metadata.avg_confidence:.2f}")
            
            if hasattr(track.metadata, 'ad_relevance_score'):
                metadata_text.append(f"Ad: {track.metadata.ad_relevance_score:.2f}")
            
            # Draw metadata text
            for i, text in enumerate(metadata_text):
                text_y = y2 + 15 + i * 15
                cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                           self.font_scale * 0.8, track_color, 1)
        
        return frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw raw detections on the frame."""
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Draw detection with dashed line
            color = (128, 128, 128)  # Gray color for raw detections
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color)
            
            # Draw confidence
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale * 0.7, color, 1)
        
        return frame
    
    def _draw_dashed_rectangle(self, frame: np.ndarray, pt1: Tuple[int, int],
                              pt2: Tuple[int, int], color: Tuple[int, int, int],
                              dash_length: int = 5):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 1)
        
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 1)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 1)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 1)
    
    def _add_frame_info(self, frame: np.ndarray, tracks: List[Track],
                       detections: Optional[List[Detection]] = None) -> np.ndarray:
        """Add frame-level information overlay."""
        height, width = frame.shape[:2]
        
        # Create info panel
        info_panel_height = 120
        info_panel = np.zeros((info_panel_height, width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)  # Dark gray background
        
        # Count tracks by state
        state_counts = {}
        for track in tracks:
            state = track.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Add text information
        info_lines = [
            f"Total Tracks: {len(tracks)}",
            f"Active: {state_counts.get('tracked', 0)} | Lost: {state_counts.get('lost', 0)} | New: {state_counts.get('new', 0)}",
            f"Detections: {len(detections) if detections else 0}",
            "States: " + " | ".join([f"{k.title()}: {v}" for k, v in state_counts.items()])
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(info_panel, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale, (255, 255, 255), 1)
        
        # Concatenate with original frame
        return np.vstack([frame, info_panel])
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for tracks."""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def save_tracking_summary(self, tracks: List[Track], 
                            metrics: Dict[str, Any],
                            output_dir: str) -> bool:
        """Save comprehensive tracking summary."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save track data
            track_data = []
            for track in tracks:
                track_info = {
                    'track_id': track.track_id,
                    'state': track.state.value,
                    'length': len(track.bbox_history),
                    'confidence_history': track.confidence_history,
                    'bbox_history': track.bbox_history,
                    'first_frame': track.first_frame_id,
                    'last_frame': track.last_frame_id
                }
                track_data.append(track_info)
            
            with open(output_path / 'tracks.json', 'w') as f:
                json.dump(track_data, f, indent=2)
            
            # Save metrics
            with open(output_path / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Tracking summary saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tracking summary: {e}")
            return False 