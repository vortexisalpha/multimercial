"""
Ensemble Depth Estimation with Marigold and Depth Pro Models

This module provides advanced ensemble depth estimation combining multiple models
with learned fusion weights, confidence-based weighting, and adaptive model selection.
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from .marigold_estimator import MarigoldDepthEstimator
from .depth_pro_estimator import DepthProEstimator
from .utils import DepthMetricsCollector

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Fusion methods for ensemble estimation."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    LEARNED_FUSION = "learned_fusion"
    ADAPTIVE_SELECTION = "adaptive_selection"
    QUALITY_AWARE = "quality_aware"


class SceneComplexity(Enum):
    """Scene complexity levels for adaptive selection."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble depth estimation."""
    
    # Fusion configuration
    fusion_method: FusionMethod = FusionMethod.CONFIDENCE_BASED
    learned_fusion: bool = True
    adaptive_weights: bool = True
    
    # Model weighting
    marigold_weight: float = 0.6
    depth_pro_weight: float = 0.4
    confidence_threshold: float = 0.7
    
    # Scene analysis
    scene_analysis: bool = True
    complexity_threshold: float = 0.5
    edge_density_weight: float = 0.3
    texture_variance_weight: float = 0.4
    
    # Quality control
    quality_assessment: bool = True
    min_confidence: float = 0.3
    outlier_rejection: bool = True
    consistency_check: bool = True
    
    # Performance optimization
    parallel_inference: bool = True
    cache_predictions: bool = False
    memory_efficient: bool = True
    
    # Validation and metrics
    ground_truth_validation: bool = False
    metrics_collection: bool = True
    benchmark_mode: bool = False


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble depth estimation."""
    
    total_time: float = 0.0
    marigold_time: float = 0.0
    depth_pro_time: float = 0.0
    fusion_time: float = 0.0
    
    marigold_confidence: float = 0.0
    depth_pro_confidence: float = 0.0
    ensemble_confidence: float = 0.0
    
    scene_complexity: float = 0.0
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    
    quality_score: float = 0.0
    consistency_score: float = 0.0
    
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0


class SceneComplexityAnalyzer:
    """Analyze scene complexity for adaptive model selection."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        
    def analyze_complexity(self, image: torch.Tensor) -> Tuple[float, SceneComplexity]:
        """Analyze scene complexity from image."""
        try:
            # Convert to numpy for analysis
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                img_np = image.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Compute complexity metrics
            edge_density = self._compute_edge_density(gray)
            texture_variance = self._compute_texture_variance(gray)
            gradient_magnitude = self._compute_gradient_magnitude(gray)
            
            # Combine metrics
            complexity_score = (
                edge_density * self.config.edge_density_weight +
                texture_variance * self.config.texture_variance_weight +
                gradient_magnitude * (1.0 - self.config.edge_density_weight - self.config.texture_variance_weight)
            )
            
            # Determine complexity level
            if complexity_score < 0.3:
                complexity_level = SceneComplexity.SIMPLE
            elif complexity_score < 0.6:
                complexity_level = SceneComplexity.MODERATE
            elif complexity_score < 0.8:
                complexity_level = SceneComplexity.COMPLEX
            else:
                complexity_level = SceneComplexity.VERY_COMPLEX
            
            return complexity_score, complexity_level
            
        except Exception as e:
            logger.warning(f"Scene complexity analysis failed: {e}")
            return 0.5, SceneComplexity.MODERATE
    
    def _compute_edge_density(self, gray_image: np.ndarray) -> float:
        """Compute edge density using Canny edge detector."""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        return edge_density
    
    def _compute_texture_variance(self, gray_image: np.ndarray) -> float:
        """Compute texture variance using local standard deviation."""
        # Apply Gaussian blur and compute local variance
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        variance = cv2.Laplacian(blurred, cv2.CV_64F).var()
        # Normalize to [0, 1]
        normalized_variance = min(variance / 10000.0, 1.0)
        return normalized_variance
    
    def _compute_gradient_magnitude(self, gray_image: np.ndarray) -> float:
        """Compute gradient magnitude."""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        normalized_magnitude = magnitude.mean() / 255.0
        
        return normalized_magnitude


class LearnedFusionNetwork(nn.Module):
    """Learned fusion network for combining depth predictions."""
    
    def __init__(self, input_channels: int = 6):  # 2 depth maps + 2 confidence maps + 2 auxiliary features
        super().__init__()
        
        self.fusion_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),  # Output: fused depth + confidence + weights
            nn.Sigmoid()
        )
        
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Weights for marigold and depth_pro
            nn.Softmax(dim=1)
        )
    
    def forward(self, depth1: torch.Tensor, depth2: torch.Tensor,
                conf1: torch.Tensor, conf2: torch.Tensor,
                aux_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of fusion network.
        
        Args:
            depth1: First depth map (Marigold)
            depth2: Second depth map (Depth Pro)
            conf1: First confidence map
            conf2: Second confidence map
            aux_features: Additional features (optional)
            
        Returns:
            Tuple of (fused_depth, fused_confidence, fusion_weights)
        """
        # Prepare input tensor
        if aux_features is None:
            # Use gradient magnitude as auxiliary features
            grad1 = self._compute_gradient_features(depth1)
            grad2 = self._compute_gradient_features(depth2)
            inputs = torch.cat([depth1, depth2, conf1, conf2, grad1, grad2], dim=1)
        else:
            inputs = torch.cat([depth1, depth2, conf1, conf2, aux_features], dim=1)
        
        # Compute pixel-wise fusion
        fusion_output = self.fusion_net(inputs)
        fused_depth = fusion_output[:, 0:1]
        fused_confidence = fusion_output[:, 1:2]
        pixel_weights = fusion_output[:, 2:3]
        
        # Compute global weights
        global_weights = self.weight_net(inputs)
        
        return fused_depth, fused_confidence, global_weights
    
    def _compute_gradient_features(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute gradient features for auxiliary input."""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude


class QualityAssessmentModule:
    """Quality assessment for depth maps and ensemble predictions."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
    
    def assess_quality(self, depth_map: torch.Tensor, 
                      confidence_map: torch.Tensor,
                      original_image: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Assess quality of depth prediction."""
        try:
            quality_metrics = {}
            
            # Basic depth statistics
            depth_valid = depth_map[depth_map > 0]
            if len(depth_valid) > 0:
                quality_metrics['depth_coverage'] = len(depth_valid) / depth_map.numel()
                quality_metrics['depth_variance'] = depth_valid.var().item()
                quality_metrics['depth_range'] = (depth_valid.max() - depth_valid.min()).item()
            else:
                quality_metrics['depth_coverage'] = 0.0
                quality_metrics['depth_variance'] = 0.0
                quality_metrics['depth_range'] = 0.0
            
            # Confidence statistics
            quality_metrics['confidence_mean'] = confidence_map.mean().item()
            quality_metrics['confidence_std'] = confidence_map.std().item()
            quality_metrics['high_confidence_ratio'] = (confidence_map > self.config.confidence_threshold).float().mean().item()
            
            # Edge preservation if original image is available
            if original_image is not None:
                edge_score = self._compute_edge_preservation(depth_map, original_image)
                quality_metrics['edge_preservation'] = edge_score
            
            # Smoothness metrics
            smoothness_score = self._compute_smoothness(depth_map)
            quality_metrics['smoothness'] = smoothness_score
            
            # Overall quality score
            quality_score = self._compute_overall_quality(quality_metrics)
            quality_metrics['overall_quality'] = quality_score
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {'overall_quality': 0.5}
    
    def _compute_edge_preservation(self, depth_map: torch.Tensor, image: torch.Tensor) -> float:
        """Compute edge preservation score."""
        try:
            # Convert to grayscale if needed
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # Compute edges
            image_edges = self._compute_edges(gray.unsqueeze(0))
            depth_edges = self._compute_edges(depth_map)
            
            # Compute correlation
            correlation = F.cosine_similarity(
                image_edges.flatten(), depth_edges.flatten(), dim=0
            )
            
            return correlation.item()
            
        except Exception as e:
            logger.warning(f"Edge preservation computation failed: {e}")
            return 0.5
    
    def _compute_edges(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute edges using Sobel operator."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges
    
    def _compute_smoothness(self, depth_map: torch.Tensor) -> float:
        """Compute depth smoothness score."""
        try:
            # Compute second derivatives
            laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                   dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
            
            laplacian_response = F.conv2d(depth_map, laplacian, padding=1)
            smoothness = (-laplacian_response.abs().mean()).exp().item()
            
            return smoothness
            
        except Exception as e:
            logger.warning(f"Smoothness computation failed: {e}")
            return 0.5
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score from individual metrics."""
        try:
            # Weighted combination of metrics
            weights = {
                'depth_coverage': 0.2,
                'confidence_mean': 0.3,
                'high_confidence_ratio': 0.2,
                'edge_preservation': 0.2,
                'smoothness': 0.1
            }
            
            quality_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    quality_score += metrics[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                quality_score /= total_weight
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Overall quality computation failed: {e}")
            return 0.5


class EnsembleDepthEstimator:
    """
    Advanced ensemble depth estimator combining Marigold and Depth Pro models.
    
    Features:
    - Multiple fusion methods (weighted average, confidence-based, learned fusion)
    - Adaptive model selection based on scene complexity
    - Quality assessment and validation
    - Performance optimization and monitoring
    """
    
    def __init__(self, 
                 marigold_estimator: MarigoldDepthEstimator,
                 depth_pro_estimator: DepthProEstimator,
                 config: Optional[DictConfig] = None):
        """
        Initialize ensemble estimator.
        
        Args:
            marigold_estimator: Marigold depth estimator
            depth_pro_estimator: Depth Pro estimator
            config: Ensemble configuration
        """
        self.config = EnsembleConfig(**(config or {}))
        
        # Model estimators
        self.marigold_estimator = marigold_estimator
        self.depth_pro_estimator = depth_pro_estimator
        
        # Initialize components
        self.scene_analyzer = SceneComplexityAnalyzer(self.config)
        self.quality_assessor = QualityAssessmentModule(self.config)
        self.metrics_collector = DepthMetricsCollector() if self.config.metrics_collection else None
        
        # Learned fusion network
        self.fusion_network = None
        if self.config.learned_fusion:
            self._initialize_fusion_network()
        
        # Performance tracking
        self.metrics = EnsembleMetrics()
        self.prediction_cache = {}
        
        # Device management
        self.device = self._get_device()
        
        logger.info("EnsembleDepthEstimator initialized")
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for ensemble processing."""
        # Use the same device as the estimators
        if hasattr(self.marigold_estimator, 'device'):
            return self.marigold_estimator.device
        elif hasattr(self.depth_pro_estimator, 'device'):
            return self.depth_pro_estimator.device
        else:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _initialize_fusion_network(self):
        """Initialize learned fusion network."""
        try:
            self.fusion_network = LearnedFusionNetwork().to(self.device)
            
            # Load pre-trained weights if available
            fusion_weights_path = "models/fusion_network_weights.pth"
            if os.path.exists(fusion_weights_path):
                self.fusion_network.load_state_dict(
                    torch.load(fusion_weights_path, map_location=self.device)
                )
                logger.info("Loaded pre-trained fusion network weights")
            else:
                logger.info("Using randomly initialized fusion network")
                
        except Exception as e:
            logger.warning(f"Failed to initialize fusion network: {e}")
            self.fusion_network = None
    
    def estimate_depth_ensemble(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> torch.Tensor:
        """
        Estimate depth using ensemble of models.
        
        Args:
            frames: Input frames (single or batch)
            
        Returns:
            Ensemble depth predictions
        """
        start_time = time.time()
        
        try:
            # Preprocess frames if needed
            processed_frames = self._preprocess_frames(frames)
            
            # Parallel or sequential inference
            if self.config.parallel_inference:
                depth_predictions = self._parallel_inference(processed_frames)
            else:
                depth_predictions = self._sequential_inference(processed_frames)
            
            # Fuse predictions
            fusion_start = time.time()
            ensemble_depth = self._fuse_predictions(depth_predictions, processed_frames)
            self.metrics.fusion_time = time.time() - fusion_start
            
            # Post-process and validate
            ensemble_depth = self._post_process_ensemble(ensemble_depth, processed_frames)
            
            # Update metrics
            self.metrics.total_time = time.time() - start_time
            self._update_ensemble_metrics(depth_predictions, ensemble_depth)
            
            return ensemble_depth
            
        except Exception as e:
            logger.error(f"Ensemble depth estimation failed: {e}")
            # Fallback to single model
            return self._fallback_estimation(frames)
    
    def _preprocess_frames(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> torch.Tensor:
        """Preprocess frames for ensemble estimation."""
        # Convert to consistent format
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        
        processed = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            elif isinstance(frame, np.ndarray):
                if frame.dtype == np.uint8:
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                else:
                    frame = torch.from_numpy(frame)
                    if frame.dim() == 3:
                        frame = frame.permute(2, 0, 1)
            
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
            
            processed.append(frame)
        
        return torch.cat(processed, dim=0).to(self.device)
    
    def _parallel_inference(self, frames: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Run parallel inference on both models."""
        import concurrent.futures
        import threading
        
        results = {}
        
        # Thread-safe inference functions
        def run_marigold():
            try:
                marigold_start = time.time()
                depth = self.marigold_estimator.estimate_depth(frames)
                confidence = torch.ones_like(depth) * 0.8  # Default confidence for Marigold
                self.metrics.marigold_time = time.time() - marigold_start
                return depth, confidence
            except Exception as e:
                logger.error(f"Marigold inference failed: {e}")
                return None, None
        
        def run_depth_pro():
            try:
                depth_pro_start = time.time()
                depth, confidence = self.depth_pro_estimator.estimate_depth(frames)
                self.metrics.depth_pro_time = time.time() - depth_pro_start
                return depth, confidence
            except Exception as e:
                logger.error(f"Depth Pro inference failed: {e}")
                return None, None
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            marigold_future = executor.submit(run_marigold)
            depth_pro_future = executor.submit(run_depth_pro)
            
            # Get results
            marigold_depth, marigold_conf = marigold_future.result()
            depth_pro_depth, depth_pro_conf = depth_pro_future.result()
        
        if marigold_depth is not None:
            results['marigold'] = (marigold_depth, marigold_conf)
        
        if depth_pro_depth is not None:
            results['depth_pro'] = (depth_pro_depth, depth_pro_conf)
        
        return results
    
    def _sequential_inference(self, frames: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Run sequential inference on both models."""
        results = {}
        
        try:
            # Marigold inference
            marigold_start = time.time()
            marigold_depth = self.marigold_estimator.estimate_depth(frames)
            marigold_conf = torch.ones_like(marigold_depth) * 0.8
            self.metrics.marigold_time = time.time() - marigold_start
            
            results['marigold'] = (marigold_depth, marigold_conf)
            
        except Exception as e:
            logger.error(f"Marigold inference failed: {e}")
        
        try:
            # Depth Pro inference
            depth_pro_start = time.time()
            depth_pro_depth, depth_pro_conf = self.depth_pro_estimator.estimate_depth(frames)
            self.metrics.depth_pro_time = time.time() - depth_pro_start
            
            results['depth_pro'] = (depth_pro_depth, depth_pro_conf)
            
        except Exception as e:
            logger.error(f"Depth Pro inference failed: {e}")
        
        return results
    
    def _fuse_predictions(self, predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
                         original_frames: torch.Tensor) -> torch.Tensor:
        """Fuse depth predictions from multiple models."""
        if len(predictions) == 0:
            raise ValueError("No valid predictions to fuse")
        
        if len(predictions) == 1:
            # Only one model succeeded
            model_name = list(predictions.keys())[0]
            return predictions[model_name][0]
        
        # Get predictions
        marigold_depth, marigold_conf = predictions.get('marigold', (None, None))
        depth_pro_depth, depth_pro_conf = predictions.get('depth_pro', (None, None))
        
        if marigold_depth is None or depth_pro_depth is None:
            # One model failed, use the other
            if marigold_depth is not None:
                return marigold_depth
            elif depth_pro_depth is not None:
                return depth_pro_depth
            else:
                raise ValueError("Both models failed")
        
        # Ensure compatible shapes
        marigold_depth, depth_pro_depth = self._align_predictions(marigold_depth, depth_pro_depth)
        marigold_conf, depth_pro_conf = self._align_predictions(marigold_conf, depth_pro_conf)
        
        # Choose fusion method
        if self.config.fusion_method == FusionMethod.LEARNED_FUSION and self.fusion_network is not None:
            return self._learned_fusion(marigold_depth, depth_pro_depth, marigold_conf, depth_pro_conf)
        elif self.config.fusion_method == FusionMethod.CONFIDENCE_BASED:
            return self._confidence_based_fusion(marigold_depth, depth_pro_depth, marigold_conf, depth_pro_conf)
        elif self.config.fusion_method == FusionMethod.ADAPTIVE_SELECTION:
            return self._adaptive_selection_fusion(marigold_depth, depth_pro_depth, original_frames)
        elif self.config.fusion_method == FusionMethod.QUALITY_AWARE:
            return self._quality_aware_fusion(marigold_depth, depth_pro_depth, marigold_conf, depth_pro_conf, original_frames)
        else:
            return self._weighted_average_fusion(marigold_depth, depth_pro_depth)
    
    def _align_predictions(self, pred1: torch.Tensor, pred2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align predictions to have compatible shapes."""
        if pred1.shape != pred2.shape:
            # Resize to match the smaller resolution (more efficient)
            target_shape = [min(pred1.shape[i], pred2.shape[i]) for i in range(len(pred1.shape))]
            
            if pred1.shape[-2:] != target_shape[-2:]:
                pred1 = F.interpolate(pred1, size=target_shape[-2:], mode='bilinear', align_corners=False)
            
            if pred2.shape[-2:] != target_shape[-2:]:
                pred2 = F.interpolate(pred2, size=target_shape[-2:], mode='bilinear', align_corners=False)
        
        return pred1, pred2
    
    def _weighted_average_fusion(self, depth1: torch.Tensor, depth2: torch.Tensor) -> torch.Tensor:
        """Simple weighted average fusion."""
        w1 = self.config.marigold_weight
        w2 = self.config.depth_pro_weight
        
        # Normalize weights
        total_weight = w1 + w2
        w1 /= total_weight
        w2 /= total_weight
        
        fused_depth = w1 * depth1 + w2 * depth2
        
        # Store fusion weights
        self.metrics.fusion_weights = {'marigold': w1, 'depth_pro': w2}
        
        return fused_depth
    
    def _confidence_based_fusion(self, depth1: torch.Tensor, depth2: torch.Tensor,
                                conf1: torch.Tensor, conf2: torch.Tensor) -> torch.Tensor:
        """Confidence-based adaptive fusion."""
        # Compute pixel-wise weights based on confidence
        total_conf = conf1 + conf2 + 1e-8  # Avoid division by zero
        w1 = conf1 / total_conf
        w2 = conf2 / total_conf
        
        # Apply confidence threshold
        high_conf_mask1 = conf1 > self.config.confidence_threshold
        high_conf_mask2 = conf2 > self.config.confidence_threshold
        
        # Use high-confidence predictions directly
        fused_depth = w1 * depth1 + w2 * depth2
        fused_depth = torch.where(high_conf_mask1 & ~high_conf_mask2, depth1, fused_depth)
        fused_depth = torch.where(high_conf_mask2 & ~high_conf_mask1, depth2, fused_depth)
        
        # Store average fusion weights
        self.metrics.fusion_weights = {
            'marigold': w1.mean().item(),
            'depth_pro': w2.mean().item()
        }
        
        return fused_depth
    
    def _learned_fusion(self, depth1: torch.Tensor, depth2: torch.Tensor,
                       conf1: torch.Tensor, conf2: torch.Tensor) -> torch.Tensor:
        """Learned fusion using neural network."""
        try:
            with torch.no_grad():
                fused_depth, fused_conf, fusion_weights = self.fusion_network(
                    depth1, depth2, conf1, conf2
                )
            
            # Store fusion weights
            self.metrics.fusion_weights = {
                'marigold': fusion_weights[:, 0].mean().item(),
                'depth_pro': fusion_weights[:, 1].mean().item()
            }
            
            return fused_depth
            
        except Exception as e:
            logger.warning(f"Learned fusion failed, falling back to confidence-based: {e}")
            return self._confidence_based_fusion(depth1, depth2, conf1, conf2)
    
    def _adaptive_selection_fusion(self, depth1: torch.Tensor, depth2: torch.Tensor,
                                  original_frames: torch.Tensor) -> torch.Tensor:
        """Adaptive model selection based on scene complexity."""
        batch_size = original_frames.shape[0]
        fused_depths = []
        
        for i in range(batch_size):
            frame = original_frames[i]
            complexity_score, complexity_level = self.scene_analyzer.analyze_complexity(frame)
            
            # Select model based on complexity
            if complexity_level in [SceneComplexity.SIMPLE, SceneComplexity.MODERATE]:
                # Use Depth Pro for simple scenes (faster)
                selected_depth = depth2[i:i+1]
                fusion_weights = {'marigold': 0.0, 'depth_pro': 1.0}
            else:
                # Use Marigold for complex scenes (higher quality)
                selected_depth = depth1[i:i+1]
                fusion_weights = {'marigold': 1.0, 'depth_pro': 0.0}
            
            fused_depths.append(selected_depth)
        
        # Store fusion weights (average across batch)
        self.metrics.fusion_weights = fusion_weights
        
        return torch.cat(fused_depths, dim=0)
    
    def _quality_aware_fusion(self, depth1: torch.Tensor, depth2: torch.Tensor,
                             conf1: torch.Tensor, conf2: torch.Tensor,
                             original_frames: torch.Tensor) -> torch.Tensor:
        """Quality-aware fusion based on prediction quality assessment."""
        batch_size = original_frames.shape[0]
        fused_depths = []
        fusion_weights_list = []
        
        for i in range(batch_size):
            # Assess quality of each prediction
            quality1 = self.quality_assessor.assess_quality(
                depth1[i:i+1], conf1[i:i+1], original_frames[i:i+1]
            )
            quality2 = self.quality_assessor.assess_quality(
                depth2[i:i+1], conf2[i:i+1], original_frames[i:i+1]
            )
            
            # Compute quality-based weights
            q1 = quality1.get('overall_quality', 0.5)
            q2 = quality2.get('overall_quality', 0.5)
            
            total_quality = q1 + q2 + 1e-8
            w1 = q1 / total_quality
            w2 = q2 / total_quality
            
            # Fuse based on quality weights
            fused_depth = w1 * depth1[i:i+1] + w2 * depth2[i:i+1]
            fused_depths.append(fused_depth)
            
            fusion_weights_list.append({'marigold': w1, 'depth_pro': w2})
        
        # Store average fusion weights
        avg_marigold_weight = np.mean([fw['marigold'] for fw in fusion_weights_list])
        avg_depth_pro_weight = np.mean([fw['depth_pro'] for fw in fusion_weights_list])
        
        self.metrics.fusion_weights = {
            'marigold': avg_marigold_weight,
            'depth_pro': avg_depth_pro_weight
        }
        
        return torch.cat(fused_depths, dim=0)
    
    def _post_process_ensemble(self, ensemble_depth: torch.Tensor, 
                              original_frames: torch.Tensor) -> torch.Tensor:
        """Post-process ensemble depth predictions."""
        try:
            # Outlier rejection
            if self.config.outlier_rejection:
                ensemble_depth = self._reject_outliers(ensemble_depth)
            
            # Consistency check
            if self.config.consistency_check:
                ensemble_depth = self._enforce_consistency(ensemble_depth)
            
            return ensemble_depth
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return ensemble_depth
    
    def _reject_outliers(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Reject outlier depth values."""
        # Use median filtering to remove outliers
        depth_np = depth_map.cpu().numpy()
        
        for i in range(depth_np.shape[0]):
            for c in range(depth_np.shape[1]):
                filtered = cv2.medianBlur(depth_np[i, c].astype(np.float32), 5)
                depth_np[i, c] = filtered
        
        return torch.from_numpy(depth_np).to(depth_map.device)
    
    def _enforce_consistency(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Enforce spatial consistency in depth predictions."""
        # Apply edge-preserving smoothing
        for i in range(depth_map.shape[0]):
            for c in range(depth_map.shape[1]):
                depth_slice = depth_map[i, c].cpu().numpy()
                
                # Convert to 8-bit for bilateral filter
                depth_8bit = ((depth_slice - depth_slice.min()) / 
                             (depth_slice.max() - depth_slice.min() + 1e-8) * 255).astype(np.uint8)
                
                # Apply bilateral filter
                filtered = cv2.bilateralFilter(depth_8bit, 9, 75, 75)
                
                # Convert back to original range
                filtered_normalized = filtered.astype(np.float32) / 255.0
                depth_map[i, c] = torch.from_numpy(
                    filtered_normalized * (depth_slice.max() - depth_slice.min()) + depth_slice.min()
                ).to(depth_map.device)
        
        return depth_map
    
    def _fallback_estimation(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> torch.Tensor:
        """Fallback depth estimation using single model."""
        try:
            # Try Marigold first
            return self.marigold_estimator.estimate_depth(frames)
        except Exception:
            try:
                # Try Depth Pro
                depth, _ = self.depth_pro_estimator.estimate_depth(frames)
                return depth
            except Exception as e:
                logger.error(f"All models failed: {e}")
                # Return dummy depth
                if isinstance(frames, (list, tuple)):
                    batch_size = len(frames)
                    height, width = 480, 640  # Default resolution
                else:
                    if hasattr(frames, 'shape'):
                        batch_size = frames.shape[0] if frames.ndim == 4 else 1
                        height, width = frames.shape[-2:]
                    else:
                        batch_size = 1
                        height, width = 480, 640
                
                return torch.zeros(batch_size, 1, height, width, device=self.device)
    
    def _update_ensemble_metrics(self, predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                                ensemble_depth: torch.Tensor):
        """Update ensemble performance metrics."""
        # Update confidence metrics
        if 'marigold' in predictions:
            self.metrics.marigold_confidence = predictions['marigold'][1].mean().item()
        
        if 'depth_pro' in predictions:
            self.metrics.depth_pro_confidence = predictions['depth_pro'][1].mean().item()
        
        # Compute ensemble confidence (average of component confidences weighted by fusion weights)
        marigold_weight = self.metrics.fusion_weights.get('marigold', 0.0)
        depth_pro_weight = self.metrics.fusion_weights.get('depth_pro', 0.0)
        
        self.metrics.ensemble_confidence = (
            marigold_weight * self.metrics.marigold_confidence +
            depth_pro_weight * self.metrics.depth_pro_confidence
        )
        
        # Memory usage
        if torch.cuda.is_available():
            self.metrics.memory_usage = torch.cuda.max_memory_allocated() / 1e9
    
    def get_metrics(self) -> EnsembleMetrics:
        """Get ensemble performance metrics."""
        return self.metrics
    
    def benchmark(self, test_data: List[torch.Tensor], iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive benchmark of ensemble estimation."""
        logger.info(f"Running ensemble benchmark with {iterations} iterations")
        
        results = {
            'total_times': [],
            'marigold_times': [],
            'depth_pro_times': [],
            'fusion_times': [],
            'memory_usage': [],
            'fusion_weights': []
        }
        
        for i in tqdm(range(iterations), desc="Benchmarking"):
            # Select random test sample
            test_sample = test_data[i % len(test_data)]
            
            # Reset metrics
            self.metrics = EnsembleMetrics()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Run estimation
            start_time = time.time()
            _ = self.estimate_depth_ensemble(test_sample)
            total_time = time.time() - start_time
            
            # Collect metrics
            results['total_times'].append(total_time)
            results['marigold_times'].append(self.metrics.marigold_time)
            results['depth_pro_times'].append(self.metrics.depth_pro_time)
            results['fusion_times'].append(self.metrics.fusion_time)
            results['fusion_weights'].append(self.metrics.fusion_weights.copy())
            
            if torch.cuda.is_available():
                results['memory_usage'].append(torch.cuda.max_memory_allocated() / 1e9)
        
        # Compute statistics
        benchmark_stats = {}
        for key in ['total_times', 'marigold_times', 'depth_pro_times', 'fusion_times', 'memory_usage']:
            if results[key]:
                benchmark_stats[f'{key}_mean'] = np.mean(results[key])
                benchmark_stats[f'{key}_std'] = np.std(results[key])
                benchmark_stats[f'{key}_min'] = np.min(results[key])
                benchmark_stats[f'{key}_max'] = np.max(results[key])
        
        # Fusion weights statistics
        if results['fusion_weights']:
            marigold_weights = [fw.get('marigold', 0) for fw in results['fusion_weights']]
            depth_pro_weights = [fw.get('depth_pro', 0) for fw in results['fusion_weights']]
            
            benchmark_stats['avg_marigold_weight'] = np.mean(marigold_weights)
            benchmark_stats['avg_depth_pro_weight'] = np.mean(depth_pro_weights)
        
        # Performance metrics
        if benchmark_stats.get('total_times_mean', 0) > 0:
            benchmark_stats['avg_fps'] = 1.0 / benchmark_stats['total_times_mean']
        
        logger.info(f"Benchmark completed. Average FPS: {benchmark_stats.get('avg_fps', 0):.2f}")
        
        return benchmark_stats
    
    def cleanup(self):
        """Cleanup ensemble resources."""
        if self.fusion_network is not None:
            del self.fusion_network
        
        self.prediction_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("EnsembleDepthEstimator cleanup completed") 