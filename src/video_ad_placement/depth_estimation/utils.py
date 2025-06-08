"""
Utilities for Depth Estimation

Visualization, video processing, and metrics collection utilities
for depth estimation workflows.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from dataclasses import dataclass, field

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingConfig:
    """Configuration for video depth processing."""
    output_format: str = "mp4"
    output_fps: int = 30
    visualization_mode: str = "colormap"  # colormap, heatmap, side_by_side
    save_individual_frames: bool = False
    frame_output_dir: Optional[str] = None
    depth_colormap: str = "plasma"  # plasma, viridis, jet, turbo
    quality: int = 95  # JPEG quality for frame saving


class DepthVisualization:
    """Depth map visualization utilities."""
    
    def __init__(self, colormap: str = "plasma"):
        self.colormap = colormap
        self.colormap_func = cm.get_cmap(colormap)
    
    def depth_to_colormap(self, depth_map: torch.Tensor, 
                         normalize: bool = True,
                         min_depth: Optional[float] = None,
                         max_depth: Optional[float] = None) -> np.ndarray:
        """Convert depth map to colormap visualization."""
        # Convert to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = depth_map
        
        # Squeeze dimensions if needed
        while depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        
        # Normalize depth values
        if normalize:
            if min_depth is not None and max_depth is not None:
                depth_np = np.clip(depth_np, min_depth, max_depth)
                depth_np = (depth_np - min_depth) / (max_depth - min_depth)
            else:
                depth_min, depth_max = depth_np.min(), depth_np.max()
                if depth_max > depth_min:
                    depth_np = (depth_np - depth_min) / (depth_max - depth_min)
        
        # Apply colormap
        colored = self.colormap_func(depth_np)
        
        # Convert to RGB (0-255)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return colored_rgb
    
    def create_side_by_side(self, image: np.ndarray, 
                           depth_map: torch.Tensor,
                           depth_title: str = "Depth") -> np.ndarray:
        """Create side-by-side visualization of image and depth."""
        # Ensure image is RGB
        if image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Get depth colormap
        depth_colored = self.depth_to_colormap(depth_map)
        
        # Resize to match if needed
        if image.shape[:2] != depth_colored.shape[:2]:
            h, w = image.shape[:2]
            depth_colored = cv2.resize(depth_colored, (w, h))
        
        # Concatenate horizontally
        combined = np.hstack([image, depth_colored])
        
        return combined
    
    def create_overlay(self, image: np.ndarray, 
                      depth_map: torch.Tensor,
                      alpha: float = 0.4) -> np.ndarray:
        """Create overlay visualization of image and depth."""
        # Get depth colormap
        depth_colored = self.depth_to_colormap(depth_map)
        
        # Resize to match if needed
        if image.shape[:2] != depth_colored.shape[:2]:
            h, w = image.shape[:2]
            depth_colored = cv2.resize(depth_colored, (w, h))
        
        # Blend images
        overlay = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)
        
        return overlay
    
    def save_depth_visualization(self, depth_map: torch.Tensor,
                               output_path: str,
                               title: Optional[str] = None,
                               show_colorbar: bool = True,
                               dpi: int = 300):
        """Save depth map as high-quality visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = depth_map
        
        # Squeeze dimensions
        while depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        
        # Create visualization
        im = ax.imshow(depth_np, cmap=self.colormap, aspect='auto')
        
        if title:
            ax.set_title(title, fontsize=16)
        
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Depth', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def create_3d_visualization(self, depth_map: torch.Tensor,
                              image: Optional[np.ndarray] = None,
                              subsample: int = 4) -> go.Figure:
        """Create interactive 3D visualization using Plotly."""
        # Convert to numpy
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = depth_map
        
        # Squeeze dimensions
        while depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        
        h, w = depth_np.shape
        
        # Subsample for performance
        if subsample > 1:
            depth_sub = depth_np[::subsample, ::subsample]
            h_sub, w_sub = depth_sub.shape
        else:
            depth_sub = depth_np
            h_sub, w_sub = h, w
        
        # Create coordinate meshes
        x = np.arange(w_sub)
        y = np.arange(h_sub)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=depth_sub,
            x=X,
            y=Y,
            colorscale=self.colormap,
            showscale=True
        )])
        
        fig.update_layout(
            title="3D Depth Visualization",
            scene=dict(
                xaxis_title="Width",
                yaxis_title="Height", 
                zaxis_title="Depth",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
            ),
            width=800,
            height=600
        )
        
        return fig


class VideoDepthProcessor:
    """Video processing with depth estimation."""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.visualizer = DepthVisualization(config.depth_colormap)
        
        # Setup output directory for frames
        if config.save_individual_frames and config.frame_output_dir:
            Path(config.frame_output_dir).mkdir(parents=True, exist_ok=True)
    
    def process_video_with_depth(self, 
                                video_path: str,
                                depth_estimator: Any,
                                output_path: str,
                                start_frame: int = 0,
                                end_frame: Optional[int] = None) -> Dict[str, Any]:
        """
        Process video with depth estimation and create visualization.
        
        Args:
            video_path: Input video path
            depth_estimator: Depth estimation model
            output_path: Output video path
            start_frame: Starting frame index
            end_frame: Ending frame index (None for full video)
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set frame range
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)
        
        # Setup output video writer
        if self.config.visualization_mode == "side_by_side":
            output_width = width * 2
            output_height = height
        else:
            output_width = width
            output_height = height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.config.output_fps, 
                             (output_width, output_height))
        
        # Initialize progress tracking
        frame_count = end_frame - start_frame
        pbar = tqdm(total=frame_count, desc="Processing video")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        processed_frames = 0
        depth_stats = []
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for depth estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Estimate depth
            depth_map = depth_estimator.estimate_depth(frame_rgb)
            
            # Create visualization
            if self.config.visualization_mode == "colormap":
                vis_frame = self.visualizer.depth_to_colormap(depth_map)
            elif self.config.visualization_mode == "side_by_side":
                vis_frame = self.visualizer.create_side_by_side(frame_rgb, depth_map)
            elif self.config.visualization_mode == "overlay":
                vis_frame = self.visualizer.create_overlay(frame_rgb, depth_map)
            else:
                vis_frame = frame_rgb
            
            # Convert back to BGR for video writing
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            out.write(vis_frame_bgr)
            
            # Save individual frame if requested
            if self.config.save_individual_frames:
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(self.config.frame_output_dir, frame_filename)
                cv2.imwrite(frame_path, vis_frame_bgr, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.config.quality])
            
            # Collect depth statistics
            if isinstance(depth_map, torch.Tensor):
                depth_np = depth_map.detach().cpu().numpy()
            else:
                depth_np = depth_map
            
            depth_stats.append({
                'frame_idx': frame_idx,
                'mean_depth': float(np.mean(depth_np)),
                'std_depth': float(np.std(depth_np)),
                'min_depth': float(np.min(depth_np)),
                'max_depth': float(np.max(depth_np))
            })
            
            processed_frames += 1
            pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        pbar.close()
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        stats = {
            'input_video': video_path,
            'output_video': output_path,
            'total_frames': processed_frames,
            'processing_time': processing_time,
            'fps': processed_frames / processing_time if processing_time > 0 else 0,
            'input_resolution': (width, height),
            'output_resolution': (output_width, output_height),
            'depth_stats': depth_stats
        }
        
        logger.info(f"Video processing completed: {processed_frames} frames in {processing_time:.2f}s")
        
        return stats
    
    def extract_depth_sequence(self, 
                              video_path: str,
                              depth_estimator: Any,
                              output_dir: str,
                              frame_step: int = 1) -> List[str]:
        """
        Extract depth maps from video and save as individual files.
        
        Args:
            video_path: Input video path
            depth_estimator: Depth estimation model
            output_dir: Directory to save depth maps
            frame_step: Process every N-th frame
            
        Returns:
            List of depth map file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        depth_files = []
        
        # Process video sequence
        for frame_idx, depth_map in enumerate(depth_estimator.estimate_depth_sequence(video_path)):
            if frame_idx % frame_step != 0:
                continue
                
            # Save depth map as numpy array
            depth_filename = f"depth_{frame_idx:06d}.npy"
            depth_path = output_dir / depth_filename
            
            if isinstance(depth_map, torch.Tensor):
                depth_np = depth_map.detach().cpu().numpy()
            else:
                depth_np = depth_map
            
            np.save(depth_path, depth_np)
            depth_files.append(str(depth_path))
            
            # Also save visualization
            vis_filename = f"depth_vis_{frame_idx:06d}.png"
            vis_path = output_dir / vis_filename
            self.visualizer.save_depth_visualization(depth_map, str(vis_path))
        
        return depth_files


class DepthMetricsCollector:
    """Collect and analyze depth estimation metrics."""
    
    def __init__(self):
        self.frame_metrics = []
        self.sequence_metrics = {}
    
    def collect_frame_metrics(self, frame_idx: int, 
                            depth_map: torch.Tensor,
                            processing_time: float,
                            gpu_memory: Optional[float] = None) -> Dict[str, float]:
        """Collect metrics for a single frame."""
        if isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = depth_map
        
        # Squeeze dimensions
        while depth_np.ndim > 2:
            depth_np = depth_np.squeeze()
        
        metrics = {
            'frame_idx': frame_idx,
            'processing_time': processing_time,
            'mean_depth': float(np.mean(depth_np)),
            'std_depth': float(np.std(depth_np)),
            'min_depth': float(np.min(depth_np)),
            'max_depth': float(np.max(depth_np)),
            'depth_range': float(np.ptp(depth_np)),  # Peak-to-peak
            'non_zero_ratio': float(np.count_nonzero(depth_np) / depth_np.size),
        }
        
        if gpu_memory is not None:
            metrics['gpu_memory_gb'] = gpu_memory
        
        # Compute edge density
        edges = cv2.Canny((depth_np * 255).astype(np.uint8), 50, 150)
        metrics['edge_density'] = float(np.count_nonzero(edges) / edges.size)
        
        self.frame_metrics.append(metrics)
        return metrics
    
    def compute_sequence_metrics(self) -> Dict[str, float]:
        """Compute metrics for the entire sequence."""
        if not self.frame_metrics:
            return {}
        
        # Extract time series
        processing_times = [m['processing_time'] for m in self.frame_metrics]
        mean_depths = [m['mean_depth'] for m in self.frame_metrics]
        std_depths = [m['std_depth'] for m in self.frame_metrics]
        
        # Compute sequence-level metrics
        self.sequence_metrics = {
            'total_frames': len(self.frame_metrics),
            'total_processing_time': sum(processing_times),
            'average_fps': len(processing_times) / sum(processing_times) if sum(processing_times) > 0 else 0,
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'mean_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            
            # Depth statistics
            'overall_mean_depth': np.mean(mean_depths),
            'overall_std_depth': np.mean(std_depths),
            'depth_temporal_variance': np.var(mean_depths),
            
            # Temporal consistency
            'temporal_consistency': self._compute_temporal_consistency(mean_depths),
        }
        
        if any('gpu_memory_gb' in m for m in self.frame_metrics):
            gpu_memories = [m.get('gpu_memory_gb', 0) for m in self.frame_metrics]
            self.sequence_metrics.update({
                'peak_gpu_memory': max(gpu_memories),
                'average_gpu_memory': np.mean(gpu_memories)
            })
        
        return self.sequence_metrics
    
    def _compute_temporal_consistency(self, depth_series: List[float]) -> float:
        """Compute temporal consistency score."""
        if len(depth_series) < 2:
            return 1.0
        
        # Compute frame-to-frame differences
        differences = [abs(depth_series[i] - depth_series[i-1]) 
                      for i in range(1, len(depth_series))]
        
        # Normalize by overall depth range
        depth_range = max(depth_series) - min(depth_series)
        if depth_range > 0:
            normalized_differences = [d / depth_range for d in differences]
            consistency = 1.0 - np.mean(normalized_differences)
        else:
            consistency = 1.0
        
        return max(0.0, consistency)
    
    def save_metrics(self, output_path: str):
        """Save metrics to file."""
        import json
        
        metrics_data = {
            'sequence_metrics': self.sequence_metrics,
            'frame_metrics': self.frame_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def create_metrics_visualization(self, output_path: str):
        """Create visualization of metrics over time."""
        if not self.frame_metrics:
            return
        
        # Extract data
        frame_indices = [m['frame_idx'] for m in self.frame_metrics]
        processing_times = [m['processing_time'] for m in self.frame_metrics]
        mean_depths = [m['mean_depth'] for m in self.frame_metrics]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Processing time plot
        ax1.plot(frame_indices, processing_times, 'b-', linewidth=1)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time per Frame')
        ax1.grid(True, alpha=0.3)
        
        # Mean depth plot
        ax2.plot(frame_indices, mean_depths, 'r-', linewidth=1)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Mean Depth')
        ax2.set_title('Mean Depth per Frame')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics visualization saved to {output_path}")
    
    def get_summary_report(self) -> str:
        """Generate a summary report of metrics."""
        if not self.sequence_metrics:
            self.compute_sequence_metrics()
        
        report = []
        report.append("=== Depth Estimation Metrics Summary ===")
        report.append(f"Total Frames Processed: {self.sequence_metrics.get('total_frames', 0)}")
        report.append(f"Total Processing Time: {self.sequence_metrics.get('total_processing_time', 0):.2f}s")
        report.append(f"Average FPS: {self.sequence_metrics.get('average_fps', 0):.2f}")
        report.append(f"Temporal Consistency Score: {self.sequence_metrics.get('temporal_consistency', 0):.3f}")
        
        if 'peak_gpu_memory' in self.sequence_metrics:
            report.append(f"Peak GPU Memory: {self.sequence_metrics['peak_gpu_memory']:.2f} GB")
        
        report.append("\n=== Depth Statistics ===")
        report.append(f"Overall Mean Depth: {self.sequence_metrics.get('overall_mean_depth', 0):.3f}")
        report.append(f"Depth Temporal Variance: {self.sequence_metrics.get('depth_temporal_variance', 0):.6f}")
        
        return "\n".join(report) 