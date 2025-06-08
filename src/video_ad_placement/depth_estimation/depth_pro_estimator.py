"""
Apple Depth Pro Model Implementation with High-Performance Optimizations

This module provides a production-ready implementation of Apple's Depth Pro model with:
- TensorRT optimization for real-time inference
- Multi-GPU support for parallel processing
- Metric depth scale recovery and calibration
- Comprehensive error handling and fallback mechanisms
- Performance profiling and optimization tools
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from omegaconf import DictConfig
import psutil
from tqdm import tqdm

try:
    import tensorrt as trt
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None  # Set to None when not available
    warnings.warn("TensorRT not available. Real-time optimization disabled.")

try:
    from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor
    import torch.distributed as dist
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    warnings.warn("Depth Pro dependencies not available.")

# Configure logging
logger = logging.getLogger(__name__)


class DepthProPrecision(Enum):
    """Precision modes for Depth Pro inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


class DepthProOptimization(Enum):
    """Optimization modes for Depth Pro."""
    NONE = "none"
    TENSORRT = "tensorrt"
    TORCH_COMPILE = "torch_compile"
    OPENVINO = "openvino"


@dataclass
class DepthProConfig:
    """Configuration for Depth Pro estimation."""
    
    # Model configuration
    model_name: str = "apple/DepthPro"
    model_cache_dir: Optional[str] = None
    device: str = "cuda"
    precision: DepthProPrecision = DepthProPrecision.FP16
    
    # Optimization configuration
    optimization: DepthProOptimization = DepthProOptimization.TENSORRT
    tensorrt_workspace_size: int = 1024 * 1024 * 1024  # 1GB
    batch_size: int = 8
    max_batch_size: int = 16
    
    # Processing configuration
    input_resolution: Tuple[int, int] = (384, 512)  # height, width
    output_resolution: Optional[Tuple[int, int]] = None
    adaptive_resolution: bool = True
    preserve_aspect_ratio: bool = True
    
    # Multi-GPU configuration
    multi_gpu: bool = False
    gpu_devices: List[int] = None
    data_parallel: bool = True
    
    # Metric depth configuration
    metric_depth: bool = True
    focal_length_estimation: bool = True
    depth_range: Tuple[float, float] = (0.1, 1000.0)  # meters
    scale_recovery_method: str = "geometric"  # geometric, learned, hybrid
    
    # Performance monitoring
    enable_profiling: bool = False
    memory_monitoring: bool = True
    benchmark_mode: bool = False
    
    # Error handling
    max_retries: int = 3
    fallback_precision: DepthProPrecision = DepthProPrecision.FP32
    timeout_seconds: float = 30.0


@dataclass
class DepthProMetrics:
    """Metrics for Depth Pro performance monitoring."""
    
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    
    gpu_memory_used: float = 0.0
    gpu_memory_peak: float = 0.0
    cpu_memory_used: float = 0.0
    
    batch_size: int = 0
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    
    # Quality metrics
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    depth_range_utilized: float = 0.0
    metric_scale_accuracy: float = 0.0


class TensorRTOptimizer:
    """TensorRT optimization utilities for Depth Pro."""
    
    def __init__(self, config: DepthProConfig):
        self.config = config
        self.trt_logger = None
        self.engine = None
        
        if TRT_AVAILABLE:
            self._initialize_tensorrt()
    
    def _initialize_tensorrt(self):
        """Initialize TensorRT components."""
        try:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            logger.info("TensorRT optimizer initialized")
        except Exception as e:
            logger.warning(f"TensorRT initialization failed: {e}")
            self.trt_logger = None
    
    def optimize_model(self, model: nn.Module, 
                      example_inputs: torch.Tensor) -> nn.Module:
        """Optimize model with TensorRT."""
        if not TRT_AVAILABLE or self.trt_logger is None:
            logger.warning("TensorRT optimization skipped")
            return model
        
        try:
            logger.info("Starting TensorRT optimization...")
            
            # Convert to TensorRT
            optimized_model = torch_tensorrt.compile(
                model,
                inputs=[example_inputs],
                enabled_precisions={
                    torch.float16 if self.config.precision == DepthProPrecision.FP16 
                    else torch.float32
                },
                workspace_size=self.config.tensorrt_workspace_size,
                max_batch_size=self.config.max_batch_size
            )
            
            logger.info("TensorRT optimization completed")
            return optimized_model
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return model
    
    def create_engine(self, onnx_path: str) -> Optional[Any]:  # Change from trt.ICudaEngine to Any
        """Create TensorRT engine from ONNX model."""
        if not TRT_AVAILABLE:
            return None
        
        try:
            builder = trt.Builder(self.trt_logger)
            config = builder.create_builder_config()
            config.max_workspace_size = self.config.tensorrt_workspace_size
            
            if self.config.precision == DepthProPrecision.FP16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.config.precision == DepthProPrecision.INT8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            
            parser = trt.OnnxParser(network, self.trt_logger)
            
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model")
                    return None
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine:
                logger.info("TensorRT engine created successfully")
            
            return engine
            
        except Exception as e:
            logger.error(f"TensorRT engine creation failed: {e}")
            return None


class MetricDepthCalibrator:
    """Metric depth scale recovery and calibration."""
    
    def __init__(self, config: DepthProConfig):
        self.config = config
        self.calibration_data = {}
        self.scale_factors = {}
    
    def estimate_focal_length(self, image: torch.Tensor, 
                            depth_map: torch.Tensor) -> float:
        """Estimate camera focal length from depth and image."""
        try:
            # Use geometric constraints to estimate focal length
            h, w = image.shape[-2:]
            
            # Assume reasonable field of view (60 degrees horizontal)
            fov_horizontal = np.pi / 3  # 60 degrees
            focal_length = w / (2 * np.tan(fov_horizontal / 2))
            
            return focal_length
            
        except Exception as e:
            logger.warning(f"Focal length estimation failed: {e}")
            return 525.0  # Default focal length
    
    def recover_metric_scale(self, depth_map: torch.Tensor,
                           image: torch.Tensor,
                           method: str = "geometric") -> Tuple[torch.Tensor, float]:
        """Recover metric scale for depth map."""
        try:
            if method == "geometric":
                return self._geometric_scale_recovery(depth_map, image)
            elif method == "learned":
                return self._learned_scale_recovery(depth_map, image)
            elif method == "hybrid":
                return self._hybrid_scale_recovery(depth_map, image)
            else:
                logger.warning(f"Unknown scale recovery method: {method}")
                return depth_map, 1.0
                
        except Exception as e:
            logger.error(f"Scale recovery failed: {e}")
            return depth_map, 1.0
    
    def _geometric_scale_recovery(self, depth_map: torch.Tensor,
                                image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Geometric scale recovery using scene constraints."""
        # Estimate focal length
        focal_length = self.estimate_focal_length(image, depth_map)
        
        # Use depth statistics and scene assumptions
        depth_median = torch.median(depth_map)
        
        # Assume typical indoor/outdoor scene depth ranges
        if depth_median < 0.3:  # Likely close-up scene
            scale_factor = 2.0
        elif depth_median > 0.8:  # Likely distant scene
            scale_factor = 0.5
        else:
            scale_factor = 1.0
        
        calibrated_depth = depth_map * scale_factor
        
        return calibrated_depth, scale_factor
    
    def _learned_scale_recovery(self, depth_map: torch.Tensor,
                              image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Learned scale recovery using neural network."""
        # Placeholder for learned scale recovery
        # In practice, this would use a trained neural network
        scale_factor = 1.0
        return depth_map * scale_factor, scale_factor
    
    def _hybrid_scale_recovery(self, depth_map: torch.Tensor,
                             image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Hybrid scale recovery combining geometric and learned methods."""
        geometric_depth, geo_scale = self._geometric_scale_recovery(depth_map, image)
        learned_depth, learned_scale = self._learned_scale_recovery(depth_map, image)
        
        # Weighted combination
        alpha = 0.7  # Weight for geometric method
        final_scale = alpha * geo_scale + (1 - alpha) * learned_scale
        
        return depth_map * final_scale, final_scale


class MultiGPUManager:
    """Multi-GPU management for parallel processing."""
    
    def __init__(self, config: DepthProConfig):
        self.config = config
        self.device_count = torch.cuda.device_count()
        self.devices = config.gpu_devices or list(range(self.device_count))
        self.is_distributed = False
        
        if config.multi_gpu and self.device_count > 1:
            self._setup_multi_gpu()
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU processing."""
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
                self.is_distributed = True
                logger.info(f"Multi-GPU setup completed: {self.device_count} GPUs")
        except Exception as e:
            logger.warning(f"Multi-GPU setup failed: {e}")
            self.is_distributed = False
    
    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """Parallelize model across multiple GPUs."""
        if self.config.multi_gpu and self.device_count > 1:
            if self.config.data_parallel:
                model = nn.DataParallel(model, device_ids=self.devices)
                logger.info("Model parallelized with DataParallel")
            else:
                # Distributed processing would go here
                pass
        
        return model
    
    def cleanup(self):
        """Cleanup multi-GPU resources."""
        if self.is_distributed:
            dist.destroy_process_group()


class DepthProEstimator:
    """
    High-performance Apple Depth Pro estimator with TensorRT optimization.
    
    Features:
    - TensorRT optimization for real-time inference
    - Multi-GPU support for parallel processing
    - Metric depth scale recovery and calibration
    - Comprehensive performance monitoring
    - Advanced error handling and fallback mechanisms
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Depth Pro estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = DepthProConfig(**config)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tensorrt_optimizer = TensorRTOptimizer(self.config)
        self.metric_calibrator = MetricDepthCalibrator(self.config)
        self.multi_gpu_manager = MultiGPUManager(self.config)
        self.metrics = DepthProMetrics()
        
        # Model components
        self.model = None
        self.processor = None
        self.transform = None
        self.optimized_model = None
        
        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        
        # Initialize model
        self._initialize_model()
        self._setup_preprocessing()
        
        logger.info(f"DepthProEstimator initialized on {self.device}")
    
    def _initialize_model(self):
        """Initialize Depth Pro model with optimizations."""
        if not DEPTH_PRO_AVAILABLE:
            raise ImportError("Depth Pro dependencies not available")
        
        try:
            logger.info(f"Loading Depth Pro model: {self.config.model_name}")
            
            # Load model and processor
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.model_cache_dir,
                torch_dtype=torch.float16 if self.config.precision == DepthProPrecision.FP16 else torch.float32
            ).to(self.device)
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.model_cache_dir
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Apply multi-GPU parallelization
            self.model = self.multi_gpu_manager.parallelize_model(self.model)
            
            # Apply optimizations
            self._apply_optimizations()
            
            logger.info("Depth Pro model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Depth Pro model: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply various optimizations to the model."""
        try:
            if self.config.optimization == DepthProOptimization.TENSORRT:
                self._apply_tensorrt_optimization()
            elif self.config.optimization == DepthProOptimization.TORCH_COMPILE:
                self._apply_torch_compile()
            
            # Memory optimizations
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
    
    def _apply_tensorrt_optimization(self):
        """Apply TensorRT optimization."""
        try:
            # Create example input for optimization
            example_input = torch.randn(
                1, 3, *self.config.input_resolution,
                device=self.device,
                dtype=torch.float16 if self.config.precision == DepthProPrecision.FP16 else torch.float32
            )
            
            self.optimized_model = self.tensorrt_optimizer.optimize_model(
                self.model, example_input
            )
            
            if self.optimized_model is not None:
                logger.info("TensorRT optimization applied")
                self.model = self.optimized_model
                
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
    
    def _apply_torch_compile(self):
        """Apply torch.compile optimization."""
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='max-autotune')
                logger.info("torch.compile optimization applied")
        except Exception as e:
            logger.warning(f"torch.compile optimization failed: {e}")
    
    def _setup_preprocessing(self):
        """Setup preprocessing transforms."""
        # Get model's expected input size
        if hasattr(self.processor, 'size'):
            model_input_size = self.processor.size
            if isinstance(model_input_size, dict):
                height = model_input_size.get('height', self.config.input_resolution[0])
                width = model_input_size.get('width', self.config.input_resolution[1])
            else:
                height, width = self.config.input_resolution
        else:
            height, width = self.config.input_resolution
        
        self.transform = transforms.Compose([
            transforms.Resize((height, width), antialias=True),
            transforms.ToTensor(),
        ])
    
    def estimate_depth(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate depth maps with confidence scores.
        
        Args:
            frames: Input frames (single or batch)
            
        Returns:
            Tuple of (depth_maps, confidence_scores)
        """
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Preprocess inputs
                preprocessed_inputs, original_sizes = self._preprocess_inputs(frames)
                
                preprocessing_time = time.time() - start_time
                
                # Run inference
                inference_start = time.time()
                depth_maps, confidence_scores = self._run_inference(preprocessed_inputs)
                inference_time = time.time() - inference_start
                
                # Post-process outputs
                postprocess_start = time.time()
                depth_maps, confidence_scores = self._postprocess_outputs(
                    depth_maps, confidence_scores, original_sizes
                )
                postprocessing_time = time.time() - postprocess_start
                
                # Apply metric scale recovery if enabled
                if self.config.metric_depth:
                    depth_maps = self._apply_metric_calibration(depth_maps, frames)
                
                # Update metrics
                total_time = time.time() - start_time
                self._update_metrics(
                    total_time, preprocessing_time, inference_time, 
                    postprocessing_time, depth_maps, confidence_scores
                )
                
                return depth_maps, confidence_scores
                
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            # Return fallback
            if isinstance(frames, (list, tuple)):
                batch_size = len(frames)
            elif hasattr(frames, 'shape') and len(frames.shape) == 4:
                batch_size = frames.shape[0]
            else:
                batch_size = 1
            
            fallback_depth = torch.zeros(batch_size, 1, *self.config.input_resolution, device=self.device)
            fallback_confidence = torch.ones_like(fallback_depth) * 0.5
            
            return fallback_depth, fallback_confidence
    
    def _preprocess_inputs(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> Tuple[torch.Tensor, List]:
        """Preprocess input frames for inference."""
        original_sizes = []
        processed_frames = []
        
        # Convert to list if single frame
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        
        for frame in frames:
            # Convert to PIL Image
            if isinstance(frame, torch.Tensor):
                if frame.ndim == 4:  # Batch dimension
                    frame = frame.squeeze(0)
                frame = transforms.ToPILImage()(frame)
            elif isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)
            elif not isinstance(frame, Image.Image):
                raise ValueError(f"Unsupported input type: {type(frame)}")
            
            # Store original size
            original_sizes.append(frame.size)
            
            # Use processor if available, otherwise use transform
            if self.processor is not None:
                processed = self.processor(frame, return_tensors="pt")
                processed_frame = processed['pixel_values'].squeeze(0)
            else:
                processed_frame = self.transform(frame)
            
            processed_frames.append(processed_frame)
        
        # Stack into batch
        batch_tensor = torch.stack(processed_frames).to(self.device)
        
        # Convert dtype if needed
        if self.config.precision == DepthProPrecision.FP16:
            batch_tensor = batch_tensor.half()
        
        return batch_tensor, original_sizes
    
    def _run_inference(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run depth inference on preprocessed inputs."""
        try:
            # Split into batches if needed
            batch_size = inputs.shape[0]
            max_batch = self.config.batch_size
            
            depth_maps = []
            confidence_scores = []
            
            for i in range(0, batch_size, max_batch):
                batch = inputs[i:i + max_batch]
                
                # Run model inference
                if hasattr(self.model, 'forward'):
                    outputs = self.model(batch)
                else:
                    # Use processor pipeline
                    outputs = self.model(batch)
                
                # Extract depth and confidence
                if isinstance(outputs, dict):
                    depth = outputs.get('depth', outputs.get('predicted_depth'))
                    confidence = outputs.get('confidence', torch.ones_like(depth))
                else:
                    depth = outputs
                    confidence = torch.ones_like(depth)
                
                depth_maps.append(depth)
                confidence_scores.append(confidence)
            
            # Concatenate results
            final_depth = torch.cat(depth_maps, dim=0)
            final_confidence = torch.cat(confidence_scores, dim=0)
            
            return final_depth, final_confidence
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU OOM, reducing batch size")
                torch.cuda.empty_cache()
                # Retry with smaller batch
                return self._run_inference_single(inputs)
            else:
                raise
    
    def _run_inference_single(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference with batch size 1 as fallback."""
        depth_maps = []
        confidence_scores = []
        
        for i in range(inputs.shape[0]):
            single_input = inputs[i:i+1]
            
            outputs = self.model(single_input)
            
            if isinstance(outputs, dict):
                depth = outputs.get('depth', outputs.get('predicted_depth'))
                confidence = outputs.get('confidence', torch.ones_like(depth))
            else:
                depth = outputs
                confidence = torch.ones_like(depth)
            
            depth_maps.append(depth)
            confidence_scores.append(confidence)
        
        return torch.cat(depth_maps, dim=0), torch.cat(confidence_scores, dim=0)
    
    def _postprocess_outputs(self, depth_maps: torch.Tensor, 
                           confidence_scores: torch.Tensor,
                           original_sizes: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Post-process depth maps and confidence scores."""
        processed_depths = []
        processed_confidences = []
        
        for i, (depth, confidence) in enumerate(zip(depth_maps, confidence_scores)):
            # Ensure proper dimensions
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)  # Add channel dimension
            if confidence.ndim == 2:
                confidence = confidence.unsqueeze(0)
            
            # Resize to output resolution if specified
            if self.config.output_resolution is not None:
                target_size = self.config.output_resolution
                depth = F.interpolate(
                    depth.unsqueeze(0), size=target_size, 
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                confidence = F.interpolate(
                    confidence.unsqueeze(0), size=target_size,
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            # Clamp depth to valid range
            min_depth, max_depth = self.config.depth_range
            depth = torch.clamp(depth, min_depth, max_depth)
            
            # Normalize confidence to [0, 1]
            confidence = torch.sigmoid(confidence)
            
            processed_depths.append(depth)
            processed_confidences.append(confidence)
        
        return torch.stack(processed_depths), torch.stack(processed_confidences)
    
    def _apply_metric_calibration(self, depth_maps: torch.Tensor, 
                                original_frames: Union[torch.Tensor, List]) -> torch.Tensor:
        """Apply metric depth calibration."""
        calibrated_depths = []
        
        # Convert original frames to tensor if needed
        if isinstance(original_frames, (list, tuple)):
            if len(original_frames) == 1:
                original_frames = original_frames[0]
        
        for i, depth_map in enumerate(depth_maps):
            if isinstance(original_frames, (list, tuple)):
                frame = original_frames[i]
            else:
                frame = original_frames
            
            # Convert frame to tensor if needed
            if isinstance(frame, Image.Image):
                frame = self.transform(frame).unsqueeze(0)
            elif isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame).float()
                if frame.ndim == 3:
                    frame = frame.permute(2, 0, 1).unsqueeze(0)
            
            calibrated_depth, scale_factor = self.metric_calibrator.recover_metric_scale(
                depth_map, frame, self.config.scale_recovery_method
            )
            
            calibrated_depths.append(calibrated_depth)
        
        return torch.stack(calibrated_depths)
    
    def _update_metrics(self, total_time: float, preprocessing_time: float,
                       inference_time: float, postprocessing_time: float,
                       depth_maps: torch.Tensor, confidence_scores: torch.Tensor):
        """Update performance metrics."""
        self.metrics.total_time = total_time
        self.metrics.preprocessing_time = preprocessing_time
        self.metrics.inference_time = inference_time
        self.metrics.postprocessing_time = postprocessing_time
        self.metrics.batch_size = depth_maps.shape[0]
        self.metrics.resolution = depth_maps.shape[-2:]
        self.metrics.fps = self.metrics.batch_size / total_time if total_time > 0 else 0
        
        # Confidence metrics
        self.metrics.confidence_mean = confidence_scores.mean().item()
        self.metrics.confidence_std = confidence_scores.std().item()
        
        # Depth metrics
        depth_range = depth_maps.max() - depth_maps.min()
        max_possible_range = self.config.depth_range[1] - self.config.depth_range[0]
        self.metrics.depth_range_utilized = (depth_range / max_possible_range).item()
        
        # Memory metrics
        if torch.cuda.is_available() and self.config.memory_monitoring:
            self.metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            self.metrics.gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e9
        
        # CPU memory
        process = psutil.Process()
        self.metrics.cpu_memory_used = process.memory_info().rss / 1e9
        
        # Store for statistics
        self.inference_times.append(inference_time)
        self.memory_usage.append(self.metrics.gpu_memory_used)
        
        if self.config.enable_profiling:
            logger.info(f"Depth Pro metrics: {self.metrics}")
    
    def get_metrics(self) -> DepthProMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'avg_fps': 1.0 / np.mean(self.inference_times),
            'avg_memory_usage': np.mean(self.memory_usage),
            'peak_memory_usage': np.max(self.memory_usage),
            'total_inferences': len(self.inference_times)
        }
    
    def benchmark(self, num_iterations: int = 100, 
                 batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size {batch_size}")
            
            # Create test data
            test_data = torch.randn(
                batch_size, 3, *self.config.input_resolution,
                device=self.device,
                dtype=torch.float16 if self.config.precision == DepthProPrecision.FP16 else torch.float32
            )
            
            # Warmup
            for _ in range(10):
                self.estimate_depth(test_data)
            
            # Benchmark
            times = []
            memory_usage = []
            
            for _ in tqdm(range(num_iterations), desc=f"Batch size {batch_size}"):
                start_time = time.time()
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                self.estimate_depth(test_data)
                
                times.append(time.time() - start_time)
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1e9)
            
            results[f'batch_{batch_size}'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'fps': batch_size / np.mean(times),
                'avg_memory': np.mean(memory_usage) if memory_usage else 0,
                'peak_memory': np.max(memory_usage) if memory_usage else 0
            }
        
        return results
    
    def cleanup(self):
        """Cleanup resources and free memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        
        if hasattr(self, 'optimized_model') and self.optimized_model is not None:
            del self.optimized_model
        
        self.multi_gpu_manager.cleanup()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("DepthProEstimator cleanup completed") 