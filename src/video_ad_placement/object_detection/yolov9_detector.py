"""
YOLOv9 Object Detector for Video Advertisement Placement

This module provides a production-ready YOLOv9 implementation optimized for
video advertisement placement scenarios with real-time inference, TensorRT
optimization, and adaptive processing capabilities.
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager
import math

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
    warnings.warn("TensorRT not available. Real-time optimization disabled.")

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import ops
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    warnings.warn("Ultralytics not available. Using custom YOLOv9 implementation.")

from .detection_models import (
    Detection, DetectionBatch, TrainingConfig, DetectionClass,
    SceneContext, InferenceMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class YOLOv9Config:
    """Configuration for YOLOv9 detector."""
    
    # Model configuration
    model_path: str = "yolov9c.pt"
    model_size: str = "yolov9c"
    input_size: Tuple[int, int] = (640, 640)
    num_classes: int = len(DetectionClass)
    class_names: List[str] = field(default_factory=lambda: [cls.name for cls in DetectionClass])
    
    # Inference configuration
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    agnostic_nms: bool = False
    multi_label: bool = False
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    half_precision: bool = True
    tensorrt_optimization: bool = True
    tensorrt_workspace_size: int = 4  # GB
    
    # Batch processing
    batch_size: int = 8
    max_batch_size: int = 32
    adaptive_batching: bool = True
    
    # Video-specific configuration
    temporal_smoothing: bool = True
    temporal_alpha: float = 0.7
    scene_adaptive_threshold: bool = True
    confidence_adaptation_rate: float = 0.1
    
    # Advertisement-specific configuration
    person_detection_boost: float = 1.2
    furniture_detection_boost: float = 1.1
    hand_tracking_boost: float = 1.3
    min_person_confidence: float = 0.3
    min_furniture_confidence: float = 0.4
    
    # Memory management
    memory_efficient: bool = True
    cache_size: int = 100
    garbage_collection_frequency: int = 50
    
    # Performance optimization
    compile_model: bool = False  # PyTorch 2.0+ compilation
    channels_last: bool = True
    use_autocast: bool = True
    
    # Debugging and logging
    verbose: bool = False
    profile_inference: bool = False
    save_debug_images: bool = False
    debug_output_dir: str = "debug_output"


class AdaptiveThresholdManager:
    """Manage adaptive confidence thresholds based on scene complexity."""
    
    def __init__(self, config: YOLOv9Config):
        self.config = config
        self.base_threshold = config.confidence_threshold
        self.current_threshold = config.confidence_threshold
        self.adaptation_rate = config.confidence_adaptation_rate
        self.scene_complexity_history = []
        self.detection_density_history = []
    
    def update_threshold(self, scene_context: SceneContext, detection_count: int, frame_area: float):
        """Update confidence threshold based on scene context."""
        if not self.config.scene_adaptive_threshold:
            return self.base_threshold
        
        # Compute detection density
        detection_density = detection_count / frame_area if frame_area > 0 else 0
        
        # Update history
        self.scene_complexity_history.append(scene_context.complexity_score)
        self.detection_density_history.append(detection_density)
        
        # Keep only recent history
        max_history = 30
        if len(self.scene_complexity_history) > max_history:
            self.scene_complexity_history = self.scene_complexity_history[-max_history:]
            self.detection_density_history = self.detection_density_history[-max_history:]
        
        # Compute adaptive threshold
        avg_complexity = np.mean(self.scene_complexity_history)
        avg_density = np.mean(self.detection_density_history)
        
        # Increase threshold for complex scenes to reduce false positives
        complexity_factor = 1.0 + (avg_complexity - 0.5) * 0.2
        
        # Decrease threshold for sparse scenes to catch more objects
        density_factor = 1.0 - (avg_density - 0.1) * 0.1
        density_factor = max(0.7, min(1.3, density_factor))
        
        target_threshold = self.base_threshold * complexity_factor * density_factor
        target_threshold = max(0.1, min(0.8, target_threshold))
        
        # Smooth transition
        self.current_threshold = (
            self.current_threshold * (1 - self.adaptation_rate) +
            target_threshold * self.adaptation_rate
        )
        
        return self.current_threshold


class SceneAnalyzer:
    """Analyze scene properties for adaptive detection."""
    
    def __init__(self):
        self.frame_history = []
        self.motion_estimator = None
    
    def analyze_scene(self, frame: torch.Tensor) -> SceneContext:
        """Analyze frame and return scene context."""
        context = SceneContext()
        
        try:
            # Convert to numpy for analysis
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 4:
                    frame = frame.squeeze(0)
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                if frame_np.dtype != np.uint8:
                    frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame
            
            # Compute complexity score
            context.complexity_score = self._compute_complexity(frame_np)
            
            # Analyze lighting
            context.lighting_condition = self._analyze_lighting(frame_np)
            
            # Detect indoor/outdoor
            context.indoor_outdoor = self._detect_indoor_outdoor(frame_np)
            
            # Analyze camera motion
            context.camera_motion = self._analyze_camera_motion(frame_np)
            
            # Compute object density estimate
            context.object_density = self._estimate_object_density(frame_np)
            
            # Compute placement suitability
            context.placement_suitability = self._compute_placement_suitability(context)
            
            # Update frame history
            self.frame_history.append(frame_np)
            if len(self.frame_history) > 10:
                self.frame_history.pop(0)
            
        except Exception as e:
            logger.warning(f"Scene analysis failed: {e}")
        
        return context
    
    def _compute_complexity(self, frame: np.ndarray) -> float:
        """Compute scene complexity score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Texture variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_score = min(laplacian_var / 1000.0, 1.0)
        
        # Color diversity
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        color_diversity = np.count_nonzero(hist) / 180.0
        
        # Combine scores
        complexity = (edge_density * 0.4 + texture_score * 0.4 + color_diversity * 0.2)
        return min(complexity, 1.0)
    
    def _analyze_lighting(self, frame: np.ndarray) -> str:
        """Analyze lighting conditions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 180:
            return "bright"
        elif mean_brightness < 80:
            return "dim"
        else:
            return "mixed"
    
    def _detect_indoor_outdoor(self, frame: np.ndarray) -> str:
        """Detect if scene is indoor or outdoor."""
        # Simple heuristic based on sky detection and color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Check for sky-like colors in upper portion
        upper_portion = hsv[:frame.shape[0]//3, :, :]
        blue_mask = cv2.inRange(upper_portion, (100, 50, 50), (130, 255, 255))
        blue_ratio = np.count_nonzero(blue_mask) / blue_mask.size
        
        if blue_ratio > 0.3:
            return "outdoor"
        else:
            return "indoor"
    
    def _analyze_camera_motion(self, frame: np.ndarray) -> str:
        """Analyze camera motion type."""
        if len(self.frame_history) < 2:
            return "static"
        
        try:
            prev_frame = cv2.cvtColor(self.frame_history[-1], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, None, None,
                winSize=(15, 15), maxLevel=2
            )
            
            if flow[1] is not None and len(flow[1]) > 0:
                # Analyze motion vectors
                motion_magnitude = np.mean(np.linalg.norm(flow[1] - flow[0], axis=1))
                
                if motion_magnitude > 5:
                    return "shake"
                elif motion_magnitude > 2:
                    return "pan"
                else:
                    return "static"
        except Exception:
            pass
        
        return "static"
    
    def _estimate_object_density(self, frame: np.ndarray) -> float:
        """Estimate object density in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Use contour detection as proxy for object density
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Normalize by frame area
        density = len(significant_contours) / (frame.shape[0] * frame.shape[1] / 10000)
        return min(density, 1.0)
    
    def _compute_placement_suitability(self, context: SceneContext) -> float:
        """Compute advertisement placement suitability score."""
        suitability = 0.5  # Base score
        
        # Prefer moderate complexity (not too simple, not too complex)
        if 0.3 <= context.complexity_score <= 0.7:
            suitability += 0.2
        
        # Prefer indoor scenes for many advertisement types
        if context.indoor_outdoor == "indoor":
            suitability += 0.1
        
        # Prefer stable camera
        if context.camera_motion == "static":
            suitability += 0.2
        
        # Prefer moderate object density
        if 0.2 <= context.object_density <= 0.6:
            suitability += 0.1
        
        return min(suitability, 1.0)


class YOLOv9Detector:
    """
    Advanced YOLOv9 object detector optimized for video advertisement placement.
    
    Features:
    - Real-time inference with TensorRT optimization
    - Adaptive confidence thresholding based on scene complexity
    - Batch processing for video sequences
    - Memory-efficient processing for long videos
    - Integration with tracking systems
    - Comprehensive metrics collection
    """
    
    def __init__(self, config: Union[DictConfig, YOLOv9Config]):
        """
        Initialize YOLOv9 detector.
        
        Args:
            config: Configuration object or dictionary
        """
        if isinstance(config, DictConfig):
            self.config = YOLOv9Config(**config)
        else:
            self.config = config
        
        # Initialize device
        self.device = self._setup_device()
        
        # Initialize model
        self.model = None
        self.model_compiled = False
        self._load_model()
        
        # Initialize components
        self.threshold_manager = AdaptiveThresholdManager(self.config)
        self.scene_analyzer = SceneAnalyzer()
        self.metrics = InferenceMetrics()
        
        # Processing state
        self.frame_cache = {}
        self.inference_count = 0
        self.temporal_cache = {}
        
        # Class mapping
        self.class_mapping = self._setup_class_mapping()
        self.advertisement_classes = DetectionClass.get_advertisement_relevant_classes()
        
        logger.info(f"YOLOv9Detector initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """Load and optimize YOLOv9 model."""
        try:
            if ULTRALYTICS_AVAILABLE:
                # Use Ultralytics YOLO implementation
                self.model = YOLO(self.config.model_path)
                
                # Move to device
                self.model.to(self.device)
                
                # Apply optimizations
                if self.config.half_precision and self.device.type == "cuda":
                    self.model.half()
                
                # Enable TensorRT optimization if available
                if self.config.tensorrt_optimization and TRT_AVAILABLE:
                    self._apply_tensorrt_optimization()
                
                # PyTorch compilation (if available)
                if self.config.compile_model and hasattr(torch, 'compile'):
                    self.model.model = torch.compile(self.model.model, mode='max-autotune')
                    self.model_compiled = True
                
            else:
                raise ImportError("Ultralytics not available")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_tensorrt_optimization(self):
        """Apply TensorRT optimization to the model."""
        try:
            if not TRT_AVAILABLE:
                logger.warning("TensorRT not available")
                return
            
            # Create example input for optimization
            example_input = torch.randn(
                1, 3, *self.config.input_size,
                device=self.device,
                dtype=torch.float16 if self.config.half_precision else torch.float32
            )
            
            # Apply TensorRT optimization
            optimized_model = torch_tensorrt.compile(
                self.model.model,
                inputs=[example_input],
                enabled_precisions={torch.float16 if self.config.half_precision else torch.float32},
                workspace_size=self.config.tensorrt_workspace_size * 1024 * 1024 * 1024,
                max_batch_size=self.config.max_batch_size
            )
            
            self.model.model = optimized_model
            logger.info("TensorRT optimization applied")
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
    
    def _setup_class_mapping(self) -> Dict[int, str]:
        """Setup class ID to name mapping."""
        mapping = {}
        for i, class_name in enumerate(self.config.class_names):
            mapping[i] = class_name
        return mapping
    
    def detect_objects(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> List[Detection]:
        """
        Detect objects in frames.
        
        Args:
            frames: Input frames (single frame or batch)
            
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        
        try:
            # Preprocess inputs
            processed_frames, original_shapes = self._preprocess_frames(frames)
            
            # Analyze scene context
            scene_context = self.scene_analyzer.analyze_scene(processed_frames[0])
            
            # Run inference
            predictions = self._run_inference(processed_frames, scene_context)
            
            # Post-process predictions
            detections = self._postprocess_predictions(predictions, original_shapes, scene_context)
            
            # Update metrics
            self._update_metrics(start_time, len(processed_frames), len(detections))
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def detect_sequence(self, video_path: str, **kwargs) -> Iterator[List[Detection]]:
        """
        Detect objects in video sequence with temporal optimization.
        
        Args:
            video_path: Path to video file
            **kwargs: Additional arguments for video processing
            
        Yields:
            List of Detection objects for each frame
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video: {frame_count} frames @ {fps:.2f} FPS")
            
            # Process frames in batches
            batch_frames = []
            frame_idx = 0
            
            with tqdm(total=frame_count, desc="Processing video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(frame)
                    
                    # Process batch when full or at end
                    if len(batch_frames) >= self.config.batch_size or frame_idx == frame_count - 1:
                        # Detect objects in batch
                        batch_detections = self._process_frame_batch(batch_frames, frame_idx)
                        
                        # Yield detections for each frame
                        for frame_detections in batch_detections:
                            yield frame_detections
                        
                        batch_frames = []
                        
                        # Garbage collection periodically
                        if frame_idx % self.config.garbage_collection_frequency == 0:
                            self._cleanup_cache()
                    
                    frame_idx += 1
                    pbar.update(1)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Video sequence detection failed: {e}")
            raise
    
    def _preprocess_frames(self, frames: Union[torch.Tensor, np.ndarray, Image.Image, List]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Preprocess input frames for inference."""
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        
        processed_frames = []
        original_shapes = []
        
        for frame in frames:
            # Convert to PIL Image if needed
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 4:
                    frame = frame.squeeze(0)
                frame = transforms.ToPILImage()(frame)
            elif isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            original_shapes.append(frame.size)  # (width, height)
            
            # Resize to input size
            frame = frame.resize(self.config.input_size[::-1])  # PIL uses (width, height)
            
            # Convert to tensor
            frame_tensor = transforms.ToTensor()(frame)
            
            # Apply channels last if configured
            if self.config.channels_last:
                frame_tensor = frame_tensor.contiguous(memory_format=torch.channels_last)
            
            processed_frames.append(frame_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(processed_frames).to(self.device)
        
        # Apply half precision if configured
        if self.config.half_precision and self.device.type == "cuda":
            batch_tensor = batch_tensor.half()
        
        return batch_tensor, original_shapes
    
    def _run_inference(self, frames: torch.Tensor, scene_context: SceneContext) -> torch.Tensor:
        """Run model inference with optimizations."""
        try:
            # Update adaptive threshold
            frame_area = frames.shape[-2] * frames.shape[-1]
            adaptive_threshold = self.threshold_manager.update_threshold(
                scene_context, 0, frame_area  # Detection count updated in post-processing
            )
            
            # Run inference with autocast if configured
            if self.config.use_autocast and self.device.type == "cuda":
                with torch.autocast(device_type='cuda'):
                    predictions = self.model(frames, verbose=False)
            else:
                predictions = self.model(frames, verbose=False)
            
            return predictions
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU OOM, reducing batch size")
                torch.cuda.empty_cache()
                # Process frames individually
                individual_predictions = []
                for i in range(frames.shape[0]):
                    single_frame = frames[i:i+1]
                    pred = self.model(single_frame, verbose=False)
                    individual_predictions.append(pred)
                return individual_predictions
            else:
                raise
    
    def _postprocess_predictions(self, predictions: Union[torch.Tensor, List], 
                               original_shapes: List[Tuple[int, int]],
                               scene_context: SceneContext) -> List[Detection]:
        """Post-process model predictions to Detection objects."""
        detections = []
        
        try:
            # Handle different prediction formats
            if hasattr(predictions, '__iter__') and not isinstance(predictions, torch.Tensor):
                # Multiple predictions (from individual processing)
                for i, pred in enumerate(predictions):
                    frame_detections = self._process_single_prediction(
                        pred, original_shapes[i], scene_context
                    )
                    detections.extend(frame_detections)
            else:
                # Single batch prediction
                for i in range(len(original_shapes)):
                    frame_detections = self._process_single_prediction(
                        predictions, original_shapes[i], scene_context, frame_idx=i
                    )
                    detections.extend(frame_detections)
            
            # Apply advertisement-specific filtering and boosting
            detections = self._apply_advertisement_filtering(detections, scene_context)
            
            return detections
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return []
    
    def _process_single_prediction(self, prediction, original_shape: Tuple[int, int], 
                                 scene_context: SceneContext, frame_idx: int = 0) -> List[Detection]:
        """Process a single frame prediction."""
        frame_detections = []
        
        try:
            # Extract prediction data (format depends on model implementation)
            if hasattr(prediction, 'boxes') and prediction.boxes is not None:
                boxes = prediction.boxes
                
                # Extract box coordinates, confidences, and class IDs
                if hasattr(boxes, 'xyxy'):
                    coords = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                else:
                    # Handle different box format
                    coords = boxes.data[:, :4].cpu().numpy()
                    confidences = boxes.data[:, 4].cpu().numpy()
                    class_ids = boxes.data[:, 5].cpu().numpy().astype(int)
                
                # Scale coordinates to original image size
                scale_x = original_shape[0] / self.config.input_size[1]
                scale_y = original_shape[1] / self.config.input_size[0]
                
                for coord, conf, cls_id in zip(coords, confidences, class_ids):
                    # Scale bounding box
                    x1, y1, x2, y2 = coord
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                    
                    # Get class name
                    class_name = self.class_mapping.get(cls_id, f"class_{cls_id}")
                    
                    # Apply confidence threshold
                    threshold = self._get_class_specific_threshold(class_name, scene_context)
                    if conf >= threshold:
                        detection = Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name,
                            frame_id=frame_idx,
                            scene_complexity=scene_context.complexity_score
                        )
                        frame_detections.append(detection)
            
        except Exception as e:
            logger.warning(f"Single prediction processing failed: {e}")
        
        return frame_detections
    
    def _get_class_specific_threshold(self, class_name: str, scene_context: SceneContext) -> float:
        """Get class-specific confidence threshold."""
        base_threshold = self.threshold_manager.current_threshold
        
        # Apply class-specific boosts
        if class_name.lower() in ['person', 'people']:
            threshold = base_threshold / self.config.person_detection_boost
            threshold = max(threshold, self.config.min_person_confidence)
        elif 'furniture' in class_name.lower():
            threshold = base_threshold / self.config.furniture_detection_boost
            threshold = max(threshold, self.config.min_furniture_confidence)
        elif class_name.lower() in ['hand', 'arm']:
            threshold = base_threshold / self.config.hand_tracking_boost
        else:
            threshold = base_threshold
        
        return threshold
    
    def _apply_advertisement_filtering(self, detections: List[Detection], 
                                     scene_context: SceneContext) -> List[Detection]:
        """Apply advertisement-specific filtering and enhancement."""
        if not detections:
            return detections
        
        # Filter by advertisement relevance
        relevant_detections = []
        for detection in detections:
            # Check if class is relevant for advertisement placement
            detection_class = None
            for cls in DetectionClass:
                if cls.name.lower() == detection.class_name.lower():
                    detection_class = cls
                    break
            
            if detection_class in self.advertisement_classes:
                relevant_detections.append(detection)
        
        # Apply temporal smoothing if enabled
        if self.config.temporal_smoothing:
            relevant_detections = self._apply_temporal_smoothing(relevant_detections)
        
        return relevant_detections
    
    def _apply_temporal_smoothing(self, detections: List[Detection]) -> List[Detection]:
        """Apply temporal smoothing to reduce detection jitter."""
        if not hasattr(self, 'previous_detections'):
            self.previous_detections = []
        
        smoothed_detections = []
        
        for detection in detections:
            # Find matching detection from previous frame
            best_match = None
            best_iou = 0.0
            
            for prev_detection in self.previous_detections:
                if prev_detection.class_id == detection.class_id:
                    iou = detection.iou(prev_detection)
                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                        best_iou = iou
                        best_match = prev_detection
            
            if best_match:
                # Smooth bounding box coordinates
                alpha = self.config.temporal_alpha
                x1, y1, x2, y2 = detection.bbox
                px1, py1, px2, py2 = best_match.bbox
                
                smoothed_bbox = (
                    alpha * x1 + (1 - alpha) * px1,
                    alpha * y1 + (1 - alpha) * py1,
                    alpha * x2 + (1 - alpha) * px2,
                    alpha * y2 + (1 - alpha) * py2
                )
                
                # Smooth confidence
                smoothed_confidence = alpha * detection.confidence + (1 - alpha) * best_match.confidence
                
                # Create smoothed detection
                smoothed_detection = Detection(
                    bbox=smoothed_bbox,
                    confidence=smoothed_confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    frame_id=detection.frame_id,
                    scene_complexity=detection.scene_complexity
                )
                smoothed_detections.append(smoothed_detection)
            else:
                smoothed_detections.append(detection)
        
        # Update previous detections
        self.previous_detections = smoothed_detections.copy()
        
        return smoothed_detections
    
    def _process_frame_batch(self, frames: List[np.ndarray], start_frame_idx: int) -> List[List[Detection]]:
        """Process a batch of frames efficiently."""
        batch_detections = []
        
        try:
            # Convert frames to tensor batch
            frame_tensors = []
            original_shapes = []
            
            for frame in frames:
                # Convert to PIL and resize
                pil_frame = Image.fromarray(frame)
                original_shapes.append(pil_frame.size)
                
                resized_frame = pil_frame.resize(self.config.input_size[::-1])
                tensor_frame = transforms.ToTensor()(resized_frame)
                
                if self.config.channels_last:
                    tensor_frame = tensor_frame.contiguous(memory_format=torch.channels_last)
                
                frame_tensors.append(tensor_frame)
            
            # Stack and move to device
            batch_tensor = torch.stack(frame_tensors).to(self.device)
            if self.config.half_precision and self.device.type == "cuda":
                batch_tensor = batch_tensor.half()
            
            # Analyze scene context for first frame
            scene_context = self.scene_analyzer.analyze_scene(batch_tensor[0])
            
            # Run batch inference
            predictions = self._run_inference(batch_tensor, scene_context)
            
            # Process each frame's predictions
            for i, original_shape in enumerate(original_shapes):
                frame_detections = self._process_single_prediction(
                    predictions, original_shape, scene_context, start_frame_idx + i
                )
                # Update frame IDs
                for detection in frame_detections:
                    detection.frame_id = start_frame_idx + i
                
                batch_detections.append(frame_detections)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            for i, frame in enumerate(frames):
                try:
                    frame_detections = self.detect_objects(frame)
                    for detection in frame_detections:
                        detection.frame_id = start_frame_idx + i
                    batch_detections.append(frame_detections)
                except Exception:
                    batch_detections.append([])
        
        return batch_detections
    
    def _update_metrics(self, start_time: float, frame_count: int, detection_count: int):
        """Update inference metrics."""
        self.metrics.total_inference_time = time.time() - start_time
        self.metrics.frame_count = frame_count
        self.metrics.total_detections = detection_count
        
        # GPU memory usage
        if torch.cuda.is_available():
            self.metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1e9
        
        # CPU utilization
        self.metrics.cpu_utilization = psutil.cpu_percent()
        
        self.inference_count += 1
    
    def _cleanup_cache(self):
        """Clean up caches and free memory."""
        self.frame_cache.clear()
        self.temporal_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def get_metrics(self) -> InferenceMetrics:
        """Get current inference metrics."""
        return self.metrics
    
    def fine_tune(self, dataset_path: str, training_config: TrainingConfig):
        """
        Fine-tune the model for specific scenarios.
        
        Args:
            dataset_path: Path to training dataset
            training_config: Training configuration
        """
        # This would integrate with the training module
        from .training.trainer import YOLOv9Trainer
        
        trainer = YOLOv9Trainer(training_config)
        trainer.train(dataset_path, self.model)
    
    def export_model(self, export_path: str, format: str = "onnx"):
        """
        Export model for deployment.
        
        Args:
            export_path: Path to save exported model
            format: Export format (onnx, torchscript, tensorrt)
        """
        try:
            if hasattr(self.model, 'export'):
                self.model.export(format=format, imgsz=self.config.input_size)
                logger.info(f"Model exported to {export_path} in {format} format")
            else:
                logger.warning("Model export not supported")
        except Exception as e:
            logger.error(f"Model export failed: {e}")
    
    def cleanup(self):
        """Cleanup detector resources."""
        self._cleanup_cache()
        
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        
        logger.info("YOLOv9Detector cleanup completed") 