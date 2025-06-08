"""
Object Detection Package for Video Advertisement Placement

Advanced YOLOv9-based object detection system optimized for video advertisement
placement scenarios with real-time inference, custom training, and comprehensive
evaluation capabilities.
"""

from .yolov9_detector import YOLOv9Detector, YOLOv9Config
from .detection_models import Detection, DetectionBatch, TrainingConfig
from .training.trainer import YOLOv9Trainer
from .training.dataset import VideoAdDataset, YouTubeDatasetCreator
from .training.augmentation import VideoAugmentationPipeline
from .training.hyperparameter_tuning import HyperparameterOptimizer
from .optimization.tensorrt_optimizer import TensorRTOptimizer
from .optimization.quantization import ModelQuantizer
from .optimization.memory_manager import MemoryEfficientProcessor
from .evaluation.metrics import DetectionMetrics, EvaluationSuite
from .evaluation.validator import ModelValidator
from .evaluation.benchmarking import DetectionBenchmark
from .tracking_integration.tracker_interface import TrackingIntegration
from .utils.video_processor import VideoProcessor
from .utils.data_creation import DatasetCreationTools
from .utils.nms_optimizer import AdaptiveNMSOptimizer
from .experiments.mlflow_integration import MLflowExperimentTracker
from .experiments.ab_testing import ABTestingFramework

__all__ = [
    # Core detection classes
    'YOLOv9Detector',
    'YOLOv9Config',
    'Detection',
    'DetectionBatch',
    'TrainingConfig',
    
    # Training components
    'YOLOv9Trainer',
    'VideoAdDataset',
    'YouTubeDatasetCreator',
    'VideoAugmentationPipeline',
    'HyperparameterOptimizer',
    
    # Optimization components
    'TensorRTOptimizer',
    'ModelQuantizer',
    'MemoryEfficientProcessor',
    
    # Evaluation components
    'DetectionMetrics',
    'EvaluationSuite',
    'ModelValidator',
    'DetectionBenchmark',
    
    # Integration and utilities
    'TrackingIntegration',
    'VideoProcessor',
    'DatasetCreationTools',
    'AdaptiveNMSOptimizer',
    
    # Experiment tracking
    'MLflowExperimentTracker',
    'ABTestingFramework'
]

__version__ = "1.0.0" 