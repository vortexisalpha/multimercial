"""
Core Data Models for Object Detection

This module defines the fundamental data structures used throughout the
object detection system, including Detection objects, training configurations,
and batch processing structures.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
import torch


class DetectionClass(Enum):
    """Standard detection classes for video advertisement placement."""
    PERSON = 0
    HAND = 1
    ARM = 2
    FACE = 3
    FURNITURE_CHAIR = 4
    FURNITURE_TABLE = 5
    FURNITURE_SOFA = 6
    FURNITURE_BED = 7
    FURNITURE_DESK = 8
    ELECTRONIC_TV = 9
    ELECTRONIC_LAPTOP = 10
    ELECTRONIC_PHONE = 11
    KITCHEN_APPLIANCE = 12
    DECORATION = 13
    PRODUCT = 14
    
    @classmethod
    def get_advertisement_relevant_classes(cls) -> List['DetectionClass']:
        """Get classes that are relevant for advertisement placement."""
        return [
            cls.PERSON,
            cls.HAND,
            cls.ARM,
            cls.FACE,
            cls.FURNITURE_CHAIR,
            cls.FURNITURE_TABLE,
            cls.FURNITURE_SOFA,
            cls.FURNITURE_BED,
            cls.FURNITURE_DESK,
            cls.ELECTRONIC_TV,
            cls.ELECTRONIC_LAPTOP,
            cls.PRODUCT
        ]
    
    @classmethod
    def get_class_colors(cls) -> Dict['DetectionClass', Tuple[int, int, int]]:
        """Get visualization colors for each class."""
        return {
            cls.PERSON: (255, 0, 0),       # Red
            cls.HAND: (0, 255, 0),         # Green
            cls.ARM: (0, 255, 128),        # Light Green
            cls.FACE: (255, 255, 0),       # Yellow
            cls.FURNITURE_CHAIR: (128, 0, 128),  # Purple
            cls.FURNITURE_TABLE: (128, 128, 0),  # Olive
            cls.FURNITURE_SOFA: (128, 128, 128), # Gray
            cls.FURNITURE_BED: (64, 64, 128),    # Dark Blue
            cls.FURNITURE_DESK: (128, 64, 64),   # Brown
            cls.ELECTRONIC_TV: (0, 0, 255),      # Blue
            cls.ELECTRONIC_LAPTOP: (0, 128, 255), # Light Blue
            cls.ELECTRONIC_PHONE: (255, 0, 255),  # Magenta
            cls.KITCHEN_APPLIANCE: (255, 128, 0), # Orange
            cls.DECORATION: (128, 255, 128),      # Light Green
            cls.PRODUCT: (255, 128, 128)          # Light Red
        }


@dataclass
class Detection:
    """Object detection result."""
    
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    features: Optional[torch.Tensor] = None  # For tracking
    frame_id: Optional[int] = None
    track_id: Optional[int] = None
    scene_complexity: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def area(self) -> float:
        """Calculate bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def center(self) -> Tuple[float, float]:
        """Calculate bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def iou(self, other: 'Detection') -> float:
        """Calculate Intersection over Union with another detection."""
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = self.bbox
        x1_2, y1_2, x2_2, y2_2 = other.bbox
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'frame_id': self.frame_id,
            'track_id': self.track_id,
            'scene_complexity': self.scene_complexity,
            'timestamp': self.timestamp,
            'area': self.area()
        }


@dataclass
class DetectionBatch:
    """Batch of detections with metadata."""
    
    detections: List[Detection]
    frame_ids: List[int]
    batch_size: int
    processing_time: float
    inference_time: float
    postprocessing_time: float
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionBatch':
        """Filter detections by class names."""
        filtered = [d for d in self.detections if d.class_name in class_names]
        return DetectionBatch(
            detections=filtered,
            frame_ids=self.frame_ids,
            batch_size=self.batch_size,
            processing_time=self.processing_time,
            inference_time=self.inference_time,
            postprocessing_time=self.postprocessing_time
        )
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionBatch':
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionBatch(
            detections=filtered,
            frame_ids=self.frame_ids,
            batch_size=self.batch_size,
            processing_time=self.processing_time,
            inference_time=self.inference_time,
            postprocessing_time=self.postprocessing_time
        )


@dataclass
class SceneContext:
    """Scene analysis context for adaptive detection."""
    
    complexity_score: float = 0.5
    lighting_condition: str = "mixed"  # bright, dim, mixed
    indoor_outdoor: str = "indoor"  # indoor, outdoor
    camera_motion: str = "static"  # static, pan, shake
    object_density: float = 0.3
    placement_suitability: float = 0.5
    temporal_stability: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene context to dictionary."""
        return {
            'complexity_score': self.complexity_score,
            'lighting_condition': self.lighting_condition,
            'indoor_outdoor': self.indoor_outdoor,
            'camera_motion': self.camera_motion,
            'object_density': self.object_density,
            'placement_suitability': self.placement_suitability,
            'temporal_stability': self.temporal_stability
        }


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    
    total_inference_time: float = 0.0
    preprocessing_time: float = 0.0
    model_inference_time: float = 0.0
    postprocessing_time: float = 0.0
    frame_count: int = 0
    total_detections: int = 0
    fps: float = 0.0
    
    # Resource usage
    gpu_memory_used: float = 0.0  # GB
    cpu_utilization: float = 0.0  # %
    memory_usage: float = 0.0  # MB
    
    # Quality metrics
    avg_confidence: float = 0.0
    detection_density: float = 0.0
    
    def update_fps(self):
        """Update FPS calculation."""
        if self.total_inference_time > 0:
            self.fps = self.frame_count / self.total_inference_time
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        self.update_fps()
        return {
            'total_inference_time': self.total_inference_time,
            'preprocessing_time': self.preprocessing_time,
            'model_inference_time': self.model_inference_time,
            'postprocessing_time': self.postprocessing_time,
            'frame_count': float(self.frame_count),
            'total_detections': float(self.total_detections),
            'fps': self.fps,
            'gpu_memory_used': self.gpu_memory_used,
            'cpu_utilization': self.cpu_utilization,
            'memory_usage': self.memory_usage,
            'avg_confidence': self.avg_confidence,
            'detection_density': self.detection_density
        }


@dataclass
class TrainingConfig:
    """Configuration for YOLOv9 training."""
    
    # Model configuration
    model_size: str = "yolov9c"
    input_size: Tuple[int, int] = (640, 640)
    num_classes: int = len(DetectionClass)
    class_names: List[str] = field(default_factory=lambda: [cls.name for cls in DetectionClass])
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Data paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: Optional[str] = None
    data_yaml_path: str = "data/dataset.yaml"
    
    # Augmentation configuration
    mosaic_prob: float = 1.0
    mixup_prob: float = 0.1
    copy_paste_prob: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.9
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    
    # Loss configuration
    box_loss_gain: float = 0.05
    cls_loss_gain: float = 0.3
    dfl_loss_gain: float = 1.5
    
    # Optimization
    optimizer: str = "AdamW"  # SGD, Adam, AdamW
    scheduler: str = "cosine"  # linear, cosine
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Validation and checkpointing
    val_interval: int = 1
    save_period: int = 10
    patience: int = 50
    
    # Hardware configuration
    device: str = "auto"
    workers: int = 8
    multi_gpu: bool = False
    amp: bool = True  # Automatic Mixed Precision
    
    # Video-specific training
    temporal_consistency: bool = True
    frame_sampling_rate: int = 1
    sequence_length: int = 8
    
    # Advertisement-specific configuration
    focus_on_person_detection: bool = True
    furniture_detection_weight: float = 1.5
    hand_tracking_weight: float = 2.0
    scene_complexity_adaptive: bool = True
    
    # Deployment configuration
    export_formats: List[str] = field(default_factory=lambda: ["torchscript", "onnx"])
    tensorrt_optimization: bool = True
    quantization: bool = False
    
    # Experiment tracking
    project_name: str = "video_ad_placement"
    experiment_name: str = "yolov9_training"
    tags: List[str] = field(default_factory=list)
    
    def validate(self):
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.model_size not in ["yolov9n", "yolov9s", "yolov9m", "yolov9c", "yolov9e"]:
            raise ValueError(f"Invalid model_size: {self.model_size}")
        
        if self.optimizer not in ["SGD", "Adam", "AdamW"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        if self.scheduler not in ["linear", "cosine"]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
    
    def get_model_url(self) -> str:
        """Get model download URL based on model size."""
        model_urls = {
            "yolov9n": "yolov9n.pt",
            "yolov9s": "yolov9s.pt", 
            "yolov9m": "yolov9m.pt",
            "yolov9c": "yolov9c.pt",
            "yolov9e": "yolov9e.pt"
        }
        return model_urls.get(self.model_size, "yolov9c.pt")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_size': self.model_size,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'device': self.device,
            'amp': self.amp,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'focus_on_person_detection': self.focus_on_person_detection,
            'furniture_detection_weight': self.furniture_detection_weight,
            'hand_tracking_weight': self.hand_tracking_weight
        } 