"""
Advanced Depth Estimation Module using Marigold Diffusion Model

This module provides comprehensive depth estimation capabilities with:
- Marigold diffusion model integration
- Temporal consistency using optical flow
- Memory-efficient processing for high-resolution videos
- GPU memory management and optimization
- Configurable quality/speed trade-offs
- Comprehensive metrics and logging
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from omegaconf import DictConfig
import psutil
from tqdm import tqdm

try:
    from diffusers import MarigoldDepthPipeline
    from transformers import pipeline
    MARIGOLD_AVAILABLE = True
except ImportError:
    MARIGOLD_AVAILABLE = False
    warnings.warn("Marigold dependencies not available. Install diffusers and transformers.")

# Configure logging
logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Precision modes for depth estimation."""
    FP32 = "fp32"
    FP16 = "fp16"
    MIXED = "mixed"


class QualityMode(Enum):
    """Quality vs speed trade-off modes."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    ULTRA = "ultra"


class TemporalSmoothingMethod(Enum):
    """Temporal smoothing methods."""
    OPTICAL_FLOW = "optical_flow"
    EXPONENTIAL = "exponential"
    KALMAN = "kalman"
    BILATERAL = "bilateral"


@dataclass
class DepthEstimationConfig:
    """Configuration for depth estimation."""
    
    # Model configuration
    model_name: str = "prs-eth/marigold-lcm-v1-0"
    model_cache_dir: str = "/models/depth_estimation"
    precision_mode: PrecisionMode = PrecisionMode.FP16
    device: str = "cuda"
    
    # Quality configuration
    quality_mode: QualityMode = QualityMode.BALANCED
    ensemble_size: int = 5
    num_inference_steps: int = 4
    
    # Processing configuration
    max_resolution: Tuple[int, int] = (1080, 1920)  # height, width
    batch_size: int = 4
    adaptive_batch_size: bool = True
    memory_limit_gb: float = 8.0
    
    # Temporal consistency
    enable_temporal_consistency: bool = True
    temporal_smoothing_method: TemporalSmoothingMethod = TemporalSmoothingMethod.OPTICAL_FLOW
    temporal_window_size: int = 5
    optical_flow_strength: float = 0.7
    temporal_alpha: float = 0.3
    
    # Optimization settings
    compile_model: bool = True
    use_xformers: bool = True
    gradient_checkpointing: bool = False
    
    # Output configuration
    output_dtype: torch.dtype = torch.float16
    normalize_output: bool = True
    depth_range: Tuple[float, float] = (0.1, 10.0)  # meters
    
    # Performance monitoring
    enable_metrics: bool = True
    log_gpu_memory: bool = True
    benchmark_mode: bool = False


@dataclass
class DepthEstimationMetrics:
    """Metrics for depth estimation performance."""
    
    processing_time: float = 0.0
    gpu_memory_peak: float = 0.0
    gpu_memory_allocated: float = 0.0
    cpu_memory_used: float = 0.0
    frames_processed: int = 0
    fps: float = 0.0
    batch_size_used: int = 0
    
    # Quality metrics
    depth_variance: float = 0.0
    temporal_consistency_score: float = 0.0
    edge_preservation_score: float = 0.0


class GPUMemoryManager:
    """GPU memory management utility."""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.initial_memory = 0
        
    def get_available_memory(self) -> float:
        """Get available GPU memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        return 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "total": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    def estimate_batch_size(self, frame_size: Tuple[int, int], base_batch_size: int = 4) -> int:
        """Estimate optimal batch size based on available memory."""
        available_memory = self.get_available_memory()
        
        # Rough estimation: each frame uses ~4 bytes per pixel * channels * processing overhead
        h, w = frame_size
        memory_per_frame = h * w * 3 * 4 * 8  # 8x overhead for processing
        
        max_batch_size = max(1, int(available_memory * 0.7 / memory_per_frame))
        return min(base_batch_size, max_batch_size)
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory monitoring."""
        self.initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()


class OpticalFlowProcessor:
    """Optical flow computation for temporal consistency."""
    
    def __init__(self):
        self.flow_estimator = None
        self._initialize_flow_estimator()
    
    def _initialize_flow_estimator(self):
        """Initialize optical flow estimator."""
        try:
            # Use OpenCV's optical flow if available
            self.flow_estimator = cv2.FarnebackOpticalFlow_create()
        except Exception as e:
            logger.warning(f"Failed to initialize optical flow: {e}")
            self.flow_estimator = None
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """Compute optical flow between two frames."""
        if self.flow_estimator is None:
            return None
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
            
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
            return flow
        except Exception as e:
            logger.warning(f"Optical flow computation failed: {e}")
            return None
    
    def warp_depth(self, depth_map: torch.Tensor, flow: Optional[np.ndarray]) -> torch.Tensor:
        """Warp depth map using optical flow."""
        if flow is None:
            return depth_map
        
        try:
            # Convert flow to torch and apply warping
            flow_tensor = torch.from_numpy(flow).to(depth_map.device)
            
            # Simple warping implementation
            # In practice, you'd use more sophisticated warping
            warped_depth = F.grid_sample(
                depth_map.unsqueeze(0).unsqueeze(0),
                flow_tensor.unsqueeze(0),
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            ).squeeze()
            
            return warped_depth
        except Exception as e:
            logger.warning(f"Depth warping failed: {e}")
            return depth_map


class MarigoldDepthEstimator:
    """
    Advanced depth estimation using Marigold diffusion model.
    
    Features:
    - Temporal consistency with optical flow
    - Memory-efficient processing
    - Configurable quality/speed trade-offs
    - Comprehensive metrics and logging
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the Marigold depth estimator.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
        """
        self.config = DepthEstimationConfig(**config)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.memory_manager = GPUMemoryManager(self.config.memory_limit_gb)
        self.optical_flow = OpticalFlowProcessor()
        self.metrics = DepthEstimationMetrics()
        
        # Model components
        self.pipeline = None
        self.transform = None
        self.inverse_transform = None
        
        # State management
        self.previous_depth = None
        self.depth_history = []
        self.flow_history = []
        
        # Initialize model
        self._initialize_model()
        self._setup_transforms()
        
        logger.info(f"MarigoldDepthEstimator initialized on {self.device}")
    
    def _initialize_model(self):
        """Initialize the Marigold model."""
        if not MARIGOLD_AVAILABLE:
            raise ImportError("Marigold dependencies not available")
        
        try:
            # Load Marigold pipeline
            logger.info(f"Loading Marigold model: {self.config.model_name}")
            
            self.pipeline = MarigoldDepthPipeline.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.model_cache_dir,
                torch_dtype=torch.float16 if self.config.precision_mode == PrecisionMode.FP16 else torch.float32,
            ).to(self.device)
            
            # Optimization settings
            if self.config.use_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Failed to enable xformers: {e}")
            
            if self.config.compile_model and hasattr(torch, 'compile'):
                try:
                    self.pipeline.unet = torch.compile(self.pipeline.unet)
                    logger.info("Model compiled for optimization")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Set quality parameters based on mode
            self._configure_quality_settings()
            
            logger.info("Marigold model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Marigold model: {e}")
            raise
    
    def _configure_quality_settings(self):
        """Configure quality settings based on quality mode."""
        quality_configs = {
            QualityMode.FAST: {
                "ensemble_size": 1,
                "num_inference_steps": 1,
                "batch_size": 8
            },
            QualityMode.BALANCED: {
                "ensemble_size": 3,
                "num_inference_steps": 4,
                "batch_size": 4
            },
            QualityMode.HIGH_QUALITY: {
                "ensemble_size": 5,
                "num_inference_steps": 10,
                "batch_size": 2
            },
            QualityMode.ULTRA: {
                "ensemble_size": 10,
                "num_inference_steps": 20,
                "batch_size": 1
            }
        }
        
        config_update = quality_configs.get(self.config.quality_mode, quality_configs[QualityMode.BALANCED])
        
        for key, value in config_update.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _setup_transforms(self):
        """Setup image preprocessing and postprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize(self.config.max_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
        ])
    
    def estimate_depth(self, frames: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Estimate depth for single frame or batch of frames.
        
        Args:
            frames: Input frames as tensor, numpy array, or PIL Image
            
        Returns:
            Depth maps as torch.Tensor
        """
        start_time = time.time()
        
        try:
            with self.memory_manager.memory_context():
                # Preprocess input
                processed_frames = self._preprocess_frames(frames)
                
                # Adapt batch size if needed
                if self.config.adaptive_batch_size:
                    frame_size = processed_frames.shape[-2:]
                    optimal_batch_size = self.memory_manager.estimate_batch_size(frame_size, self.config.batch_size)
                    self.metrics.batch_size_used = optimal_batch_size
                else:
                    optimal_batch_size = self.config.batch_size
                
                # Process in batches
                depth_maps = []
                num_frames = processed_frames.shape[0]
                
                for i in range(0, num_frames, optimal_batch_size):
                    batch = processed_frames[i:i + optimal_batch_size]
                    batch_depth = self._estimate_batch_depth(batch)
                    depth_maps.append(batch_depth)
                
                # Concatenate results
                result = torch.cat(depth_maps, dim=0)
                
                # Post-process
                result = self._postprocess_depth(result)
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, num_frames)
                
                return result
                
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise
    
    def estimate_depth_sequence(self, video_path: str, **kwargs) -> Iterator[torch.Tensor]:
        """
        Process video sequence with temporal consistency.
        
        Args:
            video_path: Path to input video file
            **kwargs: Additional parameters
            
        Yields:
            Depth maps for each frame with temporal consistency applied
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
            
            # Initialize progress tracking
            pbar = tqdm(total=total_frames, desc="Processing video")
            
            previous_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Estimate depth for current frame
                current_depth = self.estimate_depth(frame)
                
                # Apply temporal consistency if enabled
                if self.config.enable_temporal_consistency and previous_frame is not None:
                    current_depth = self._apply_temporal_consistency(
                        current_depth, previous_frame, frame, frame_idx
                    )
                
                # Update history
                self._update_history(current_depth, frame)
                
                # Yield result
                yield current_depth
                
                # Update for next iteration
                previous_frame = frame
                frame_idx += 1
                pbar.update(1)
                
                # Memory cleanup periodically
                if frame_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            cap.release()
            pbar.close()
            
            logger.info(f"Video processing completed: {frame_idx} frames")
            
        except Exception as e:
            logger.error(f"Video sequence processing failed: {e}")
            raise
    
    def _preprocess_frames(self, frames: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess input frames for depth estimation."""
        if isinstance(frames, Image.Image):
            frames = self.transform(frames).unsqueeze(0)
        elif isinstance(frames, np.ndarray):
            if frames.ndim == 3:  # Single frame
                frames = Image.fromarray(frames)
                frames = self.transform(frames).unsqueeze(0)
            else:  # Multiple frames
                processed = []
                for frame in frames:
                    img = Image.fromarray(frame)
                    processed.append(self.transform(img))
                frames = torch.stack(processed)
        elif isinstance(frames, torch.Tensor):
            if frames.ndim == 3:  # Single frame CHW
                frames = frames.unsqueeze(0)
            # Assume already preprocessed if tensor
        else:
            raise ValueError(f"Unsupported input type: {type(frames)}")
        
        return frames.to(self.device)
    
    def _estimate_batch_depth(self, batch: torch.Tensor) -> torch.Tensor:
        """Estimate depth for a batch of frames."""
        try:
            # Convert to PIL Images for Marigold pipeline
            pil_images = []
            for frame in batch:
                # Denormalize
                frame = self.inverse_transform(frame)
                frame = torch.clamp(frame, 0, 1)
                
                # Convert to PIL
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(frame_np))
            
            # Run Marigold pipeline
            with torch.no_grad():
                depth_outputs = self.pipeline(
                    pil_images,
                    ensemble_size=self.config.ensemble_size,
                    num_inference_steps=self.config.num_inference_steps,
                    processing_res=self.config.max_resolution,
                    match_input_res=True,
                    output_type="np"
                )
            
            # Convert to tensor
            depth_maps = []
            for depth_output in depth_outputs:
                if isinstance(depth_output, dict):
                    depth_np = depth_output.get('depth', depth_output.get('prediction'))
                else:
                    depth_np = depth_output
                
                depth_tensor = torch.from_numpy(depth_np).to(self.device)
                if depth_tensor.ndim == 2:
                    depth_tensor = depth_tensor.unsqueeze(0)  # Add channel dimension
                depth_maps.append(depth_tensor)
            
            return torch.stack(depth_maps)
            
        except Exception as e:
            logger.error(f"Batch depth estimation failed: {e}")
            # Return zero depth as fallback
            return torch.zeros(batch.shape[0], 1, *self.config.max_resolution, device=self.device)
    
    def _postprocess_depth(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Post-process depth maps."""
        # Normalize to specified range if enabled
        if self.config.normalize_output:
            min_depth, max_depth = self.config.depth_range
            depth_maps = torch.clamp(depth_maps, min_depth, max_depth)
            depth_maps = (depth_maps - min_depth) / (max_depth - min_depth)
        
        # Convert to specified output dtype
        if self.config.output_dtype != depth_maps.dtype:
            depth_maps = depth_maps.to(self.config.output_dtype)
        
        return depth_maps
    
    def _apply_temporal_consistency(self, current_depth: torch.Tensor, 
                                  previous_frame: np.ndarray, 
                                  current_frame: np.ndarray,
                                  frame_idx: int) -> torch.Tensor:
        """Apply temporal consistency to depth maps."""
        try:
            method = self.config.temporal_smoothing_method
            
            if method == TemporalSmoothingMethod.OPTICAL_FLOW:
                return self._apply_optical_flow_smoothing(current_depth, previous_frame, current_frame)
            elif method == TemporalSmoothingMethod.EXPONENTIAL:
                return self._apply_exponential_smoothing(current_depth)
            elif method == TemporalSmoothingMethod.BILATERAL:
                return self._apply_bilateral_smoothing(current_depth)
            else:
                logger.warning(f"Unknown temporal smoothing method: {method}")
                return current_depth
                
        except Exception as e:
            logger.warning(f"Temporal consistency failed: {e}")
            return current_depth
    
    def _apply_optical_flow_smoothing(self, current_depth: torch.Tensor,
                                    previous_frame: np.ndarray,
                                    current_frame: np.ndarray) -> torch.Tensor:
        """Apply optical flow-based temporal smoothing."""
        if self.previous_depth is None:
            self.previous_depth = current_depth
            return current_depth
        
        # Compute optical flow
        flow = self.optical_flow.compute_flow(previous_frame, current_frame)
        
        if flow is not None:
            # Warp previous depth using optical flow
            warped_depth = self.optical_flow.warp_depth(self.previous_depth, flow)
            
            # Blend with current depth
            alpha = self.config.temporal_alpha
            smoothed_depth = alpha * warped_depth + (1 - alpha) * current_depth
        else:
            # Fallback to simple exponential smoothing
            alpha = self.config.temporal_alpha
            smoothed_depth = alpha * self.previous_depth + (1 - alpha) * current_depth
        
        self.previous_depth = smoothed_depth
        return smoothed_depth
    
    def _apply_exponential_smoothing(self, current_depth: torch.Tensor) -> torch.Tensor:
        """Apply exponential smoothing for temporal consistency."""
        if self.previous_depth is None:
            self.previous_depth = current_depth
            return current_depth
        
        alpha = self.config.temporal_alpha
        smoothed_depth = alpha * self.previous_depth + (1 - alpha) * current_depth
        self.previous_depth = smoothed_depth
        return smoothed_depth
    
    def _apply_bilateral_smoothing(self, current_depth: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filtering for temporal smoothing."""
        if len(self.depth_history) < self.config.temporal_window_size:
            return current_depth
        
        # Simple averaging over temporal window
        window_depths = self.depth_history[-self.config.temporal_window_size:] + [current_depth]
        smoothed_depth = torch.stack(window_depths).mean(dim=0)
        return smoothed_depth
    
    def _update_history(self, depth_map: torch.Tensor, frame: np.ndarray):
        """Update depth and frame history for temporal processing."""
        self.depth_history.append(depth_map.clone())
        
        # Maintain window size
        if len(self.depth_history) > self.config.temporal_window_size:
            self.depth_history.pop(0)
    
    def _update_metrics(self, processing_time: float, num_frames: int):
        """Update performance metrics."""
        self.metrics.processing_time = processing_time
        self.metrics.frames_processed = num_frames
        self.metrics.fps = num_frames / processing_time if processing_time > 0 else 0
        
        if torch.cuda.is_available() and self.config.log_gpu_memory:
            memory_stats = self.memory_manager.get_memory_usage()
            self.metrics.gpu_memory_allocated = memory_stats["allocated"]
            self.metrics.gpu_memory_peak = memory_stats["cached"]
        
        # CPU memory
        process = psutil.Process()
        self.metrics.cpu_memory_used = process.memory_info().rss / 1024**3  # GB
        
        if self.config.enable_metrics:
            logger.info(f"Depth estimation metrics: {self.metrics}")
    
    def get_metrics(self) -> DepthEstimationMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def reset_state(self):
        """Reset internal state for new video sequence."""
        self.previous_depth = None
        self.depth_history.clear()
        self.flow_history.clear()
        
        # Reset metrics
        self.metrics = DepthEstimationMetrics()
    
    def save_checkpoint(self, checkpoint_path: str, frame_idx: int):
        """Save processing checkpoint for long videos."""
        checkpoint = {
            'frame_idx': frame_idx,
            'config': self.config.__dict__,
            'metrics': self.metrics.__dict__,
            'previous_depth': self.previous_depth.cpu() if self.previous_depth is not None else None,
            'depth_history': [d.cpu() for d in self.depth_history],
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load processing checkpoint for resuming long videos."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        frame_idx = checkpoint['frame_idx']
        
        if checkpoint['previous_depth'] is not None:
            self.previous_depth = checkpoint['previous_depth'].to(self.device)
        
        self.depth_history = [d.to(self.device) for d in checkpoint['depth_history']]
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}, resuming from frame {frame_idx}")
        return frame_idx
    
    def cleanup(self):
        """Cleanup resources and free memory."""
        self.reset_state()
        
        if self.pipeline is not None:
            del self.pipeline
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("MarigoldDepthEstimator cleanup completed")


# Quality assessment utilities
def compute_depth_quality_metrics(depth_map: torch.Tensor, 
                                 reference_depth: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute quality assessment metrics for depth maps."""
    metrics = {}
    
    # Basic statistics
    metrics['mean_depth'] = depth_map.mean().item()
    metrics['std_depth'] = depth_map.std().item()
    metrics['min_depth'] = depth_map.min().item()
    metrics['max_depth'] = depth_map.max().item()
    
    # Edge preservation (using Sobel operator)
    if depth_map.ndim == 4:  # Batch dimension
        depth_map = depth_map.squeeze(1)  # Remove channel dimension
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth_map.dtype, device=depth_map.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=depth_map.dtype, device=depth_map.device)
    
    edges_x = F.conv2d(depth_map.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    edges_y = F.conv2d(depth_map.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    metrics['edge_density'] = (edges > 0.1).float().mean().item()
    metrics['edge_strength'] = edges.mean().item()
    
    # If reference depth is provided, compute error metrics
    if reference_depth is not None:
        mse = F.mse_loss(depth_map, reference_depth)
        mae = F.l1_loss(depth_map, reference_depth)
        
        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['rmse'] = torch.sqrt(mse).item()
        
        # Relative error
        rel_error = torch.abs(depth_map - reference_depth) / (reference_depth + 1e-8)
        metrics['relative_error'] = rel_error.mean().item()
    
    return metrics 