"""
Temporal Consistency Module for Depth Estimation

Advanced temporal consistency processing using optical flow warping,
Kalman filtering, and bilateral temporal filtering.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


@dataclass
class TemporalFilterConfig:
    """Configuration for temporal filtering."""
    window_size: int = 5
    alpha: float = 0.3
    bilateral_sigma_spatial: float = 5.0
    bilateral_sigma_intensity: float = 0.1
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 1e-2
    motion_threshold: float = 0.1


class TemporalFilter(ABC):
    """Abstract base class for temporal filters."""
    
    @abstractmethod
    def filter(self, current_depth: torch.Tensor, 
               previous_depths: List[torch.Tensor], 
               **kwargs) -> torch.Tensor:
        """Apply temporal filtering to depth maps."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset filter state."""
        pass


class ExponentialTemporalFilter(TemporalFilter):
    """Exponential moving average temporal filter."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.previous_depth = None
    
    def filter(self, current_depth: torch.Tensor, 
               previous_depths: List[torch.Tensor], 
               **kwargs) -> torch.Tensor:
        """Apply exponential smoothing."""
        if self.previous_depth is None:
            self.previous_depth = current_depth
            return current_depth
        
        smoothed = self.alpha * self.previous_depth + (1 - self.alpha) * current_depth
        self.previous_depth = smoothed
        return smoothed
    
    def reset(self):
        """Reset filter state."""
        self.previous_depth = None


class BilateralTemporalFilter(TemporalFilter):
    """Bilateral temporal filter preserving edges across time."""
    
    def __init__(self, config: TemporalFilterConfig):
        self.config = config
        self.depth_history = []
    
    def filter(self, current_depth: torch.Tensor, 
               previous_depths: List[torch.Tensor], 
               **kwargs) -> torch.Tensor:
        """Apply bilateral temporal filtering."""
        # Update history
        self.depth_history.append(current_depth)
        if len(self.depth_history) > self.config.window_size:
            self.depth_history.pop(0)
        
        if len(self.depth_history) < 2:
            return current_depth
        
        # Apply bilateral filtering across temporal dimension
        return self._bilateral_temporal_smooth(self.depth_history)
    
    def _bilateral_temporal_smooth(self, depth_sequence: List[torch.Tensor]) -> torch.Tensor:
        """Apply bilateral smoothing across temporal dimension."""
        current_depth = depth_sequence[-1]
        device = current_depth.device
        
        # Convert to numpy for bilateral filtering
        depth_stack = torch.stack(depth_sequence).cpu().numpy()
        
        # Apply bilateral filtering along temporal axis
        filtered_depth = np.zeros_like(depth_stack[-1])
        
        for i in range(depth_stack.shape[1]):  # Height
            for j in range(depth_stack.shape[2]):  # Width
                temporal_signal = depth_stack[:, i, j]
                
                # Apply Gaussian smoothing with intensity-based weights
                weights = np.exp(-0.5 * (temporal_signal - temporal_signal[-1])**2 / 
                               self.config.bilateral_sigma_intensity**2)
                weights = weights / weights.sum()
                
                filtered_depth[i, j] = np.sum(weights * temporal_signal)
        
        return torch.from_numpy(filtered_depth).to(device)
    
    def reset(self):
        """Reset filter state."""
        self.depth_history.clear()


class KalmanTemporalFilter(TemporalFilter):
    """Kalman filter for temporal consistency."""
    
    def __init__(self, config: TemporalFilterConfig):
        self.config = config
        self.state = None
        self.covariance = None
        self.process_noise = config.kalman_process_noise
        self.measurement_noise = config.kalman_measurement_noise
    
    def filter(self, current_depth: torch.Tensor, 
               previous_depths: List[torch.Tensor], 
               **kwargs) -> torch.Tensor:
        """Apply Kalman filtering."""
        if self.state is None:
            self._initialize_kalman(current_depth)
            return current_depth
        
        # Prediction step
        predicted_state = self.state
        predicted_covariance = self.covariance + self.process_noise
        
        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise)
        self.state = predicted_state + kalman_gain * (current_depth - predicted_state)
        self.covariance = (1 - kalman_gain) * predicted_covariance
        
        return self.state
    
    def _initialize_kalman(self, initial_depth: torch.Tensor):
        """Initialize Kalman filter state."""
        self.state = initial_depth.clone()
        self.covariance = torch.ones_like(initial_depth) * 0.1
    
    def reset(self):
        """Reset filter state."""
        self.state = None
        self.covariance = None


class OpticalFlowWarping:
    """Advanced optical flow computation and depth warping."""
    
    def __init__(self, flow_method: str = "farneback"):
        self.flow_method = flow_method
        self.flow_params = self._get_flow_params()
    
    def _get_flow_params(self) -> Dict[str, Any]:
        """Get optical flow parameters based on method."""
        if self.flow_method == "farneback":
            return {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
        elif self.flow_method == "lucas_kanade":
            return {
                'winSize': (15, 15),
                'maxLevel': 2,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }
        else:
            return {}
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """Compute optical flow between two frames."""
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1, gray2 = frame1, frame2
            
            if self.flow_method == "farneback":
                flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)
            elif self.flow_method == "lucas_kanade":
                # For Lucas-Kanade, we need feature points
                corners = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, 
                                                qualityLevel=0.01, minDistance=10)
                if corners is not None:
                    flow, status, error = cv2.calcOpticalFlowPyrLK(
                        gray1, gray2, corners, None, **self.flow_params
                    )
                else:
                    flow = None
            else:
                # Fallback to Farneback
                flow = cv2.calcOpticalFlowPyrLK(gray1, gray2)
            
            return flow
            
        except Exception as e:
            logger.warning(f"Optical flow computation failed: {e}")
            return None
    
    def compute_dense_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """Compute dense optical flow field."""
        try:
            # Convert to grayscale
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1, gray2 = frame1, frame2
            
            # Compute dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, **self.flow_params
            )
            
            return flow
            
        except Exception as e:
            logger.warning(f"Dense optical flow computation failed: {e}")
            return None
    
    def warp_depth_map(self, depth_map: torch.Tensor, 
                       flow: np.ndarray, 
                       occlusion_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warp depth map using optical flow with occlusion detection.
        
        Args:
            depth_map: Depth map to warp
            flow: Optical flow field
            occlusion_threshold: Threshold for occlusion detection
            
        Returns:
            Tuple of (warped_depth, occlusion_mask)
        """
        try:
            device = depth_map.device
            h, w = depth_map.shape[-2:]
            
            # Convert flow to torch tensor
            flow_tensor = torch.from_numpy(flow).float().to(device)
            
            # Create coordinate grids
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            
            # Apply flow to coordinates
            new_x = x_coords.float() + flow_tensor[..., 0]
            new_y = y_coords.float() + flow_tensor[..., 1]
            
            # Normalize coordinates to [-1, 1] for grid_sample
            new_x = 2.0 * new_x / (w - 1) - 1.0
            new_y = 2.0 * new_y / (h - 1) - 1.0
            
            # Create sampling grid
            grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
            
            # Warp depth map
            if depth_map.ndim == 2:
                depth_input = depth_map.unsqueeze(0).unsqueeze(0)
            else:
                depth_input = depth_map.unsqueeze(0)
            
            warped_depth = F.grid_sample(
                depth_input, grid, 
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
            
            # Remove batch dimension
            warped_depth = warped_depth.squeeze(0)
            if warped_depth.ndim == 3 and warped_depth.shape[0] == 1:
                warped_depth = warped_depth.squeeze(0)
            
            # Detect occlusions based on flow magnitude
            flow_magnitude = torch.sqrt(flow_tensor[..., 0]**2 + flow_tensor[..., 1]**2)
            occlusion_mask = flow_magnitude > occlusion_threshold
            
            return warped_depth, occlusion_mask
            
        except Exception as e:
            logger.warning(f"Depth warping failed: {e}")
            # Return original depth with no occlusions
            occlusion_mask = torch.zeros_like(depth_map, dtype=torch.bool)
            return depth_map, occlusion_mask
    
    def compute_flow_confidence(self, flow: np.ndarray, 
                              frame1: np.ndarray, 
                              frame2: np.ndarray) -> np.ndarray:
        """Compute confidence map for optical flow."""
        try:
            # Simple confidence based on flow consistency
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Normalize to [0, 1]
            confidence = 1.0 / (1.0 + flow_magnitude)
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Flow confidence computation failed: {e}")
            return np.ones(flow.shape[:2])


class TemporalConsistencyProcessor:
    """Main processor for temporal consistency in depth estimation."""
    
    def __init__(self, config: TemporalFilterConfig):
        self.config = config
        
        # Initialize components
        self.optical_flow = OpticalFlowWarping()
        self.temporal_filter = self._create_temporal_filter()
        
        # State tracking
        self.frame_history = []
        self.depth_history = []
        self.flow_history = []
    
    def _create_temporal_filter(self) -> TemporalFilter:
        """Create temporal filter based on configuration."""
        # For now, use bilateral filter as default
        return BilateralTemporalFilter(self.config)
    
    def process_frame(self, current_depth: torch.Tensor, 
                     current_frame: np.ndarray,
                     frame_idx: int) -> torch.Tensor:
        """
        Process a single frame with temporal consistency.
        
        Args:
            current_depth: Depth map for current frame
            current_frame: RGB frame
            frame_idx: Frame index
            
        Returns:
            Temporally consistent depth map
        """
        # Update frame history
        self.frame_history.append(current_frame)
        if len(self.frame_history) > self.config.window_size:
            self.frame_history.pop(0)
        
        # If this is the first frame, just return it
        if len(self.frame_history) < 2:
            self.depth_history.append(current_depth)
            return current_depth
        
        # Compute optical flow
        previous_frame = self.frame_history[-2]
        flow = self.optical_flow.compute_dense_flow(previous_frame, current_frame)
        
        if flow is not None:
            # Warp previous depth using optical flow
            previous_depth = self.depth_history[-1]
            warped_depth, occlusion_mask = self.optical_flow.warp_depth_map(
                previous_depth, flow
            )
            
            # Blend warped depth with current depth based on occlusion
            blended_depth = self._blend_depths(
                current_depth, warped_depth, occlusion_mask
            )
            
            # Apply temporal filtering
            consistent_depth = self.temporal_filter.filter(
                blended_depth, self.depth_history
            )
        else:
            # Fallback to simple temporal filtering without warping
            consistent_depth = self.temporal_filter.filter(
                current_depth, self.depth_history
            )
        
        # Update depth history
        self.depth_history.append(consistent_depth)
        if len(self.depth_history) > self.config.window_size:
            self.depth_history.pop(0)
        
        return consistent_depth
    
    def _blend_depths(self, current_depth: torch.Tensor, 
                     warped_depth: torch.Tensor, 
                     occlusion_mask: torch.Tensor) -> torch.Tensor:
        """Blend current and warped depth maps based on occlusion."""
        # Use current depth in occluded regions, warped depth elsewhere
        alpha = self.config.alpha
        
        # Adjust blending weight based on occlusion
        blend_weight = torch.where(occlusion_mask, 0.0, alpha)
        
        blended = blend_weight * warped_depth + (1 - blend_weight) * current_depth
        return blended
    
    def compute_temporal_consistency_score(self) -> float:
        """Compute temporal consistency score for the processed sequence."""
        if len(self.depth_history) < 2:
            return 1.0
        
        # Compute average frame-to-frame similarity
        similarities = []
        for i in range(1, len(self.depth_history)):
            prev_depth = self.depth_history[i-1]
            curr_depth = self.depth_history[i]
            
            # Compute normalized cross-correlation
            correlation = F.cosine_similarity(
                prev_depth.flatten(), curr_depth.flatten(), dim=0
            )
            similarities.append(correlation.item())
        
        return np.mean(similarities)
    
    def reset(self):
        """Reset processor state for new sequence."""
        self.frame_history.clear()
        self.depth_history.clear()
        self.flow_history.clear()
        self.temporal_filter.reset()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'frames_processed': len(self.depth_history),
            'temporal_consistency_score': self.compute_temporal_consistency_score(),
            'window_size': self.config.window_size,
            'filter_type': type(self.temporal_filter).__name__
        } 