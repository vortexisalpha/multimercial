"""
Object Detection and Tracking Module

Implements YOLOv9 for object detection and ByteTrack for multi-object tracking
optimized for video advertisement placement scenarios.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms

from ..utils.logging_utils import get_logger
from ..utils.model_utils import load_checkpoint, download_model_weights
from ..utils.geometry_utils import bbox_iou, bbox_to_corners
from ..models.yolov9 import YOLOv9Model
from ..models.bytetrack import ByteTracker

logger = get_logger(__name__)


class DetectionModelType(Enum):
    """Supported object detection model types."""
    YOLOV9_S = "yolov9s"
    YOLOV9_M = "yolov9m"
    YOLOV9_L = "yolov9l"
    YOLOV9_X = "yolov9x"


@dataclass
class Detection:
    """Single object detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    timestamp: Optional[float] = None


@dataclass
class TrackingResult:
    """Multi-object tracking result for a single frame."""
    detections: List[Detection]
    frame_id: int
    timestamp: float
    processing_time: float


class ObjectDetectionConfig:
    """Configuration for object detection and tracking."""
    
    def __init__(
        self,
        model_type: DetectionModelType = DetectionModelType.YOLOV9_S,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        enable_tracking: bool = True,
        tracking_config: Optional[Dict] = None,
        target_classes: Optional[List[str]] = None,
        batch_size: int = 1,
        **kwargs
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.enable_tracking = enable_tracking
        self.tracking_config = tracking_config or {}
        self.target_classes = target_classes
        self.batch_size = batch_size
        self.kwargs = kwargs


class ObjectDetector:
    """
    Advanced object detection and tracking system using YOLOv9 and ByteTrack.
    
    This class provides real-time object detection and multi-object tracking
    capabilities optimized for video processing and advertisement placement.
    
    Attributes:
        config: Configuration object for detection parameters
        model: The YOLOv9 detection model
        tracker: ByteTrack tracking system
        class_names: List of object class names
        device: Computing device (CPU/GPU)
        is_initialized: Model initialization status
    """

    def __init__(self, config: ObjectDetectionConfig):
        """
        Initialize the object detector with specified configuration.
        
        Args:
            config: ObjectDetectionConfig with model and tracking parameters
            
        Raises:
            ValueError: If model type is not supported
            RuntimeError: If initialization fails
        """
        self.config = config
        self.model: Optional[nn.Module] = None
        self.tracker: Optional[ByteTracker] = None
        self.device = torch.device(config.device)
        self.is_initialized = False
        self.class_names: List[str] = []
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        # Initialize preprocessing
        self.preprocess_transform = self._setup_preprocessing()
        
        logger.info(f"Initializing ObjectDetector with {config.model_type.value}")
        self._initialize_model()

    def _setup_preprocessing(self) -> transforms.Compose:
        """Setup preprocessing transforms for input images."""
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
        ])

    def _initialize_model(self) -> None:
        """Initialize the detection model and tracker."""
        try:
            # Load YOLOv9 model
            self.model = self._load_yolo_model()
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize ByteTracker if tracking is enabled
            if self.config.enable_tracking:
                self.tracker = self._initialize_tracker()
            
            # Load class names
            self.class_names = self._load_class_names()
            
            self.is_initialized = True
            logger.info("Object detection model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _load_yolo_model(self) -> nn.Module:
        """Load and configure YOLOv9 model."""
        model_params = {
            'num_classes': 80,  # COCO classes by default
            'model_size': self.config.model_type.value.split('v9')[1],  # s, m, l, x
            **self.config.kwargs
        }
        
        model = YOLOv9Model(**model_params)
        
        if self.config.checkpoint_path:
            checkpoint = load_checkpoint(self.config.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded YOLOv9 checkpoint from {self.config.checkpoint_path}")
        else:
            # Download default weights
            weights_path = download_model_weights(
                self.config.model_type.value, 
                version="coco"
            )
            checkpoint = load_checkpoint(weights_path)
            model.load_state_dict(checkpoint['state_dict'])
        
        return model

    def _initialize_tracker(self) -> ByteTracker:
        """Initialize ByteTrack tracker with configuration."""
        tracker_config = {
            'frame_rate': 30,
            'track_thresh': 0.6,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'min_box_area': 10,
            **self.config.tracking_config
        }
        
        return ByteTracker(**tracker_config)

    def _load_class_names(self) -> List[str]:
        """Load object class names."""
        # Default COCO class names
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_names

    def detect(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        return_visualization: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], np.ndarray]]:
        """
        Detect objects in a single image.
        
        Args:
            image: Input image as numpy array, PIL Image, or torch tensor
            return_visualization: Whether to return annotated image
            
        Returns:
            List of Detection objects, optionally with visualization
            
        Raises:
            RuntimeError: If model is not initialized
            ValueError: If input format is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        start_time = time.time()
        
        try:
            # Preprocess input
            original_shape = self._get_original_shape(image)
            processed_input = self._preprocess_image(image)
            
            # Run detection
            with torch.no_grad():
                predictions = self.model(processed_input)
            
            # Postprocess results
            detections = self._postprocess_predictions(
                predictions, 
                original_shape,
                self.config.confidence_threshold,
                self.config.nms_threshold
            )
            
            # Filter by target classes if specified
            if self.config.target_classes:
                detections = self._filter_by_classes(detections, self.config.target_classes)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            logger.debug(f"Detected {len(detections)} objects in {inference_time:.3f}s")
            
            if return_visualization:
                vis_image = self._visualize_detections(image, detections)
                return detections, vis_image
            
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise RuntimeError(f"Detection error: {str(e)}")

    def track(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        frame_id: int = 0
    ) -> TrackingResult:
        """
        Detect and track objects in a single frame.
        
        Args:
            image: Input image
            frame_id: Frame identifier for tracking
            
        Returns:
            TrackingResult with tracked detections
            
        Raises:
            RuntimeError: If tracking is not enabled or model not initialized
        """
        if not self.config.enable_tracking:
            raise RuntimeError("Tracking not enabled in configuration")
        
        if not self.is_initialized or self.tracker is None:
            raise RuntimeError("Model or tracker not initialized")

        start_time = time.time()
        timestamp = time.time()
        
        try:
            # Get initial detections
            detections = self.detect(image)
            
            # Convert detections to tracking format
            detection_array = self._detections_to_array(detections)
            
            # Update tracker
            tracked_objects = self.tracker.update(detection_array)
            
            # Convert back to Detection objects with track IDs
            tracked_detections = self._array_to_detections(tracked_objects, timestamp)
            
            processing_time = time.time() - start_time
            
            return TrackingResult(
                detections=tracked_detections,
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Object tracking failed: {str(e)}")
            raise RuntimeError(f"Tracking error: {str(e)}")

    def _get_original_shape(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> Tuple[int, int]:
        """Get original image shape."""
        if isinstance(image, np.ndarray):
            return image.shape[:2]
        elif isinstance(image, Image.Image):
            return image.size[::-1]  # PIL returns (width, height)
        elif isinstance(image, torch.Tensor):
            return image.shape[-2:]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Preprocess image for model inference."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            return image.to(self.device)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        processed = self.preprocess_transform(image)
        
        if processed.dim() == 3:
            processed = processed.unsqueeze(0)
            
        return processed.to(self.device)

    def _postprocess_predictions(
        self,
        predictions: torch.Tensor,
        original_shape: Tuple[int, int],
        conf_thresh: float,
        nms_thresh: float
    ) -> List[Detection]:
        """Postprocess model predictions to Detection objects."""
        detections = []
        
        # Assuming predictions format: [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
        pred = predictions[0]  # First batch element
        
        # Filter by confidence
        valid_detections = pred[pred[:, 4] > conf_thresh]
        
        if len(valid_detections) == 0:
            return detections
        
        # Apply NMS
        boxes = valid_detections[:, :4]
        scores = valid_detections[:, 4]
        classes = valid_detections[:, 5]
        
        # Convert to numpy for NMS
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        classes_np = classes.cpu().numpy()
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(),
            scores_np.tolist(),
            conf_thresh,
            nms_thresh
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            
            # Scale boxes to original image size
            h_orig, w_orig = original_shape
            h_model, w_model = self.config.input_size
            
            scale_x = w_orig / w_model
            scale_y = h_orig / h_model
            
            for idx in indices:
                box = boxes_np[idx]
                conf = scores_np[idx]
                cls_id = int(classes_np[idx])
                
                # Scale coordinates
                x1 = box[0] * scale_x
                y1 = box[1] * scale_y
                x2 = box[2] * scale_x
                y2 = box[3] * scale_y
                
                # Create detection object
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    timestamp=time.time()
                )
                
                detections.append(detection)
        
        return detections

    def _filter_by_classes(self, detections: List[Detection], target_classes: List[str]) -> List[Detection]:
        """Filter detections by target class names."""
        return [det for det in detections if det.class_name in target_classes]

    def _detections_to_array(self, detections: List[Detection]) -> np.ndarray:
        """Convert Detection objects to numpy array for tracking."""
        if not detections:
            return np.empty((0, 5))
        
        array = np.zeros((len(detections), 5))
        for i, det in enumerate(detections):
            array[i] = [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence]
        
        return array

    def _array_to_detections(self, tracked_array: np.ndarray, timestamp: float) -> List[Detection]:
        """Convert tracking array back to Detection objects."""
        detections = []
        
        for track in tracked_array:
            x1, y1, x2, y2, track_id, conf, cls_id = track[:7]
            
            detection = Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else f"class_{int(cls_id)}",
                track_id=int(track_id),
                timestamp=timestamp
            )
            
            detections.append(detection)
        
        return detections

    def _visualize_detections(self, image: Union[np.ndarray, Image.Image], detections: List[Detection]) -> np.ndarray:
        """Create visualization of detections on image."""
        if isinstance(image, Image.Image):
            vis_image = np.array(image)
        else:
            vis_image = image.copy()
        
        # Convert RGB to BGR for OpenCV
        if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det.bbox]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"{det.class_name}: {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" ID:{det.track_id}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_image

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_inference_time = self.total_inference_time / max(self.frame_count, 1)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "total_frames": self.frame_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_inference_time,
            "average_fps": fps,
            "model_type": self.config.model_type.value,
            "tracking_enabled": self.config.enable_tracking,
        }

    def reset_tracker(self) -> None:
        """Reset the tracking system."""
        if self.tracker is not None:
            self.tracker.reset()
            logger.info("Tracker reset successfully")

    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tracker is not None:
            del self.tracker
            self.tracker = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_initialized = False
        logger.info("ObjectDetector cleanup completed") 