"""
Surface Normal Estimation for 3D Scene Understanding

This module implements hybrid surface normal estimation combining geometric
calculations from depth maps with learned approaches for enhanced accuracy
and robustness in plane detection and wall surface analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

from .plane_models import SurfaceNormalConfig

logger = logging.getLogger(__name__)


class BaseSurfaceNormalEstimator(ABC):
    """Abstract base class for surface normal estimation."""
    
    @abstractmethod
    def estimate_surface_normals(self, depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor) -> torch.Tensor:
        """Estimate surface normals from depth map and RGB frame."""
        pass


class GeometricNormalEstimator(BaseSurfaceNormalEstimator):
    """Geometric surface normal estimation from depth gradients."""
    
    def __init__(self, neighborhood_size: int = 3, smoothing_sigma: float = 1.0):
        """
        Initialize geometric normal estimator.
        
        Args:
            neighborhood_size: Size of neighborhood for gradient computation
            smoothing_sigma: Gaussian smoothing sigma for depth preprocessing
        """
        self.neighborhood_size = neighborhood_size
        self.smoothing_sigma = smoothing_sigma
        
        # Create Sobel kernels for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def estimate_surface_normals(self, depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals using geometric approach.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3) - not used in geometric method
            
        Returns:
            Surface normals tensor (H, W, 3)
        """
        device = depth_map.device
        height, width = depth_map.shape
        
        # Move Sobel kernels to correct device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Smooth depth map if requested
        if self.smoothing_sigma > 0:
            depth_smoothed = self._gaussian_smooth(depth_map, self.smoothing_sigma)
        else:
            depth_smoothed = depth_map
        
        # Add batch and channel dimensions for conv2d
        depth_input = depth_smoothed.unsqueeze(0).unsqueeze(0)
        
        # Compute depth gradients
        grad_x = F.conv2d(depth_input, sobel_x, padding=1).squeeze()
        grad_y = F.conv2d(depth_input, sobel_y, padding=1).squeeze()
        
        # Convert gradients to 3D space normals
        normals = self._gradients_to_normals(grad_x, grad_y, depth_smoothed)
        
        # Normalize normals
        norm_magnitude = torch.norm(normals, dim=2, keepdim=True)
        normals = normals / (norm_magnitude + 1e-8)
        
        # Handle invalid normals (where depth is zero or gradients are invalid)
        invalid_mask = (depth_map == 0) | (norm_magnitude.squeeze() < 1e-6)
        normals[invalid_mask] = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        return normals
    
    def _gaussian_smooth(self, depth_map: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian smoothing to depth map."""
        # Create Gaussian kernel
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        kernel = self._create_gaussian_kernel(kernel_size, sigma, depth_map.device)
        
        # Apply convolution
        depth_input = depth_map.unsqueeze(0).unsqueeze(0)
        smoothed = F.conv2d(depth_input, kernel, padding=kernel_size//2)
        
        return smoothed.squeeze()
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2
        
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g2d = g1d[:, None] * g1d[None, :]
        g2d = g2d / g2d.sum()
        
        return g2d.unsqueeze(0).unsqueeze(0)
    
    def _gradients_to_normals(self, grad_x: torch.Tensor, grad_y: torch.Tensor, 
                             depth: torch.Tensor) -> torch.Tensor:
        """Convert depth gradients to surface normals."""
        height, width = depth.shape
        
        # Create normalized device coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height, device=depth.device),
            torch.linspace(-1, 1, width, device=depth.device),
            indexing='ij'
        )
        
        # Compute surface tangent vectors
        # T1 = d/dx [x, y, z(x,y)] = [1, 0, dz/dx]
        # T2 = d/dy [x, y, z(x,y)] = [0, 1, dz/dy]
        
        ones = torch.ones_like(grad_x)
        zeros = torch.zeros_like(grad_x)
        
        tangent_x = torch.stack([ones, zeros, grad_x], dim=2)
        tangent_y = torch.stack([zeros, ones, grad_y], dim=2)
        
        # Normal = T1 × T2
        normals = torch.cross(tangent_x, tangent_y, dim=2)
        
        return normals


class LearnedNormalEstimator(BaseSurfaceNormalEstimator):
    """Learned surface normal estimation using neural networks."""
    
    def __init__(self, config: SurfaceNormalConfig):
        """
        Initialize learned normal estimator.
        
        Args:
            config: Surface normal configuration
        """
        self.config = config
        self.model = None
        self.device = torch.device(config.model_device)
        
        if config.model_path:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained surface normal estimation model."""
        try:
            if self.config.model_type == "surface_normal_net":
                self.model = self._create_surface_normal_net()
            else:
                logger.warning(f"Unknown model type: {self.config.model_type}")
                return
            
            # Load weights if available
            if self.config.model_path:
                self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded surface normal model: {self.config.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load surface normal model: {e}")
            self.model = None
    
    def _create_surface_normal_net(self) -> nn.Module:
        """Create surface normal estimation network."""
        return SurfaceNormalNet()
    
    def estimate_surface_normals(self, depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals using learned approach.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3)
            
        Returns:
            Surface normals tensor (H, W, 3)
        """
        if self.model is None:
            logger.warning("No model loaded, returning zero normals")
            return torch.zeros((*depth_map.shape, 3), device=depth_map.device)
        
        try:
            # Prepare input
            input_tensor = self._prepare_input(depth_map, rgb_frame)
            
            # Run inference
            with torch.no_grad():
                if self.config.model_precision == "fp16":
                    input_tensor = input_tensor.half()
                
                output = self.model(input_tensor)
                
                if self.config.model_precision == "fp16":
                    output = output.float()
            
            # Post-process output
            normals = self._postprocess_output(output, depth_map.shape)
            
            return normals
            
        except Exception as e:
            logger.error(f"Learned normal estimation failed: {e}")
            return torch.zeros((*depth_map.shape, 3), device=depth_map.device)
    
    def _prepare_input(self, depth_map: torch.Tensor, rgb_frame: torch.Tensor) -> torch.Tensor:
        """Prepare input tensor for neural network."""
        # Normalize depth
        depth_normalized = depth_map / depth_map.max() if depth_map.max() > 0 else depth_map
        
        # Normalize RGB
        rgb_normalized = rgb_frame / 255.0 if rgb_frame.max() > 1 else rgb_frame
        
        # Concatenate depth and RGB
        if rgb_frame is not None and rgb_frame.numel() > 0:
            input_tensor = torch.cat([
                depth_normalized.unsqueeze(0),  # Add channel dimension
                rgb_normalized.permute(2, 0, 1)  # CHW format
            ], dim=0)
        else:
            input_tensor = depth_normalized.unsqueeze(0)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor.to(self.device)
    
    def _postprocess_output(self, output: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """Post-process network output to surface normals."""
        # Remove batch dimension
        output = output.squeeze(0)
        
        # Reshape to (H, W, 3)
        if output.dim() == 3:  # (3, H, W)
            output = output.permute(1, 2, 0)
        
        # Resize to original shape if needed
        if output.shape[:2] != original_shape:
            output = F.interpolate(
                output.permute(2, 0, 1).unsqueeze(0),
                size=original_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        
        # Normalize normals
        norm_magnitude = torch.norm(output, dim=2, keepdim=True)
        output = output / (norm_magnitude + 1e-8)
        
        return output


class SurfaceNormalNet(nn.Module):
    """Neural network for surface normal estimation."""
    
    def __init__(self, input_channels: int = 4, output_channels: int = 3):
        """
        Initialize surface normal network.
        
        Args:
            input_channels: Number of input channels (depth + RGB)
            output_channels: Number of output channels (normal components)
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Tanh()  # Normals in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(x)
        normals = self.decoder(features)
        return normals


class HybridNormalEstimator(BaseSurfaceNormalEstimator):
    """Hybrid surface normal estimator combining geometric and learned approaches."""
    
    def __init__(self, config: SurfaceNormalConfig):
        """
        Initialize hybrid normal estimator.
        
        Args:
            config: Surface normal configuration
        """
        self.config = config
        config.validate()
        
        # Initialize component estimators
        self.geometric_estimator = GeometricNormalEstimator(
            neighborhood_size=config.neighborhood_size,
            smoothing_sigma=config.smoothing_sigma if config.depth_smoothing else 0.0
        )
        
        self.learned_estimator = None
        if config.use_learned_normals:
            self.learned_estimator = LearnedNormalEstimator(config)
        
        logger.info("HybridNormalEstimator initialized")
    
    def estimate_surface_normals(self, depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals using hybrid approach.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3)
            
        Returns:
            Surface normals tensor (H, W, 3)
        """
        device = depth_map.device
        height, width = depth_map.shape
        
        # Initialize result
        final_normals = torch.zeros((height, width, 3), device=device)
        total_weight = 0.0
        
        # Geometric normals
        if self.config.use_geometric_normals:
            geometric_normals = self.geometric_estimator.estimate_surface_normals(
                depth_map, rgb_frame
            )
            final_normals += self.config.geometric_weight * geometric_normals
            total_weight += self.config.geometric_weight
        
        # Learned normals
        if self.config.use_learned_normals and self.learned_estimator is not None:
            learned_normals = self.learned_estimator.estimate_surface_normals(
                depth_map, rgb_frame
            )
            final_normals += self.config.learned_weight * learned_normals
            total_weight += self.config.learned_weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_normals = final_normals / total_weight
        
        # Apply refinement if enabled
        if self.config.enable_refinement:
            final_normals = self._refine_normals(final_normals, depth_map)
        
        # Final normalization
        norm_magnitude = torch.norm(final_normals, dim=2, keepdim=True)
        final_normals = final_normals / (norm_magnitude + 1e-8)
        
        # Filter by confidence if available
        confidence_mask = self._compute_confidence_mask(final_normals, depth_map)
        final_normals[~confidence_mask] = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        return final_normals
    
    def _refine_normals(self, normals: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Refine normal estimates using iterative approach."""
        refined_normals = normals.clone()
        
        for iteration in range(self.config.refinement_iterations):
            # Apply bilateral filtering to smooth normals while preserving edges
            refined_normals = self._bilateral_filter_normals(refined_normals, depth_map)
            
            # Ensure unit length
            norm_magnitude = torch.norm(refined_normals, dim=2, keepdim=True)
            refined_normals = refined_normals / (norm_magnitude + 1e-8)
        
        return refined_normals
    
    def _bilateral_filter_normals(self, normals: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filtering to normal maps."""
        # Simplified bilateral filtering
        # In practice, you might use more sophisticated bilateral filtering
        
        kernel_size = 5
        sigma_spatial = 1.0
        sigma_range = 0.1
        
        # Create spatial kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=normals.device)
        coords = coords - kernel_size // 2
        spatial_kernel = torch.exp(-(coords[:, None]**2 + coords[None, :]**2) / (2 * sigma_spatial**2))
        
        height, width, _ = normals.shape
        filtered_normals = torch.zeros_like(normals)
        
        pad = kernel_size // 2
        # Add batch and channel dimensions for padding
        normals_padded = F.pad(normals.permute(2, 0, 1).unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
        depth_padded = F.pad(depth_map.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze()
        
        for i in range(height):
            for j in range(width):
                center_depth = depth_map[i, j]
                center_normal = normals[i, j]
                
                # Extract neighborhood
                normal_patch = normals_padded[:, i:i+kernel_size, j:j+kernel_size]
                depth_patch = depth_padded[i:i+kernel_size, j:j+kernel_size]
                
                # Compute range weights based on depth difference
                depth_diff = torch.abs(depth_patch - center_depth)
                range_weights = torch.exp(-(depth_diff**2) / (2 * sigma_range**2))
                
                # Combine spatial and range weights
                weights = spatial_kernel * range_weights
                weights = weights / weights.sum()
                
                # Apply weighted average
                for c in range(3):
                    filtered_normals[i, j, c] = torch.sum(weights * normal_patch[c])
        
        return filtered_normals
    
    def _compute_confidence_mask(self, normals: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Compute confidence mask for normal estimates."""
        height, width, _ = normals.shape
        
        # Check for valid depth
        valid_depth = (depth_map > 0) & (depth_map < 100)
        
        # Check for reasonable normal magnitude
        normal_magnitude = torch.norm(normals, dim=2)
        valid_normals = (normal_magnitude > 0.1) & (normal_magnitude < 2.0)
        
        # Check for depth discontinuities
        if self.config.gradient_threshold > 0:
            depth_grad = torch.gradient(depth_map)
            grad_magnitude = torch.sqrt(depth_grad[0]**2 + depth_grad[1]**2)
            no_discontinuity = grad_magnitude < self.config.gradient_threshold
        else:
            no_discontinuity = torch.ones_like(valid_depth)
        
        # Combine conditions
        confidence_mask = valid_depth & valid_normals & no_discontinuity
        
        return confidence_mask


class SurfaceNormalEstimator:
    """Main interface for surface normal estimation."""
    
    def __init__(self, config: SurfaceNormalConfig):
        """
        Initialize surface normal estimator.
        
        Args:
            config: Surface normal configuration
        """
        self.config = config
        self.estimator = HybridNormalEstimator(config)
        
        logger.info("SurfaceNormalEstimator initialized")
    
    def estimate_surface_normals(self, depth_map: torch.Tensor,
                               rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals from depth map and RGB frame.
        
        Args:
            depth_map: Depth map tensor (H, W)
            rgb_frame: RGB frame tensor (H, W, 3)
            
        Returns:
            Surface normals tensor (H, W, 3)
        """
        return self.estimator.estimate_surface_normals(depth_map, rgb_frame) 