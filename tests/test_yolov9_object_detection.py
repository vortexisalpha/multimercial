"""
Comprehensive Test Suite for YOLOv9 Object Detection System

This module provides extensive testing coverage for all components of the
YOLOv9-based object detection system including detection, training, evaluation,
and optimization components.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import yaml

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_ad_placement.object_detection.yolov9_detector import (
    YOLOv9Detector, YOLOv9Config, AdaptiveThresholdManager, SceneAnalyzer
)
from video_ad_placement.object_detection.detection_models import (
    Detection, DetectionBatch, TrainingConfig, DetectionClass, SceneContext, InferenceMetrics
)
from video_ad_placement.object_detection.training.trainer import YOLOv9Trainer, TrainingMetrics

# Suppress warnings during testing
warnings.filterwarnings("ignore")


class MockYOLOModel:
    """Mock YOLO model for testing without actual model dependencies."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.training = False
        
    def __call__(self, x, verbose=False):
        """Mock inference."""
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else 1
        
        # Create mock results
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Mock detection data
        num_detections = np.random.randint(1, 6)  # 1-5 detections per image
        
        mock_boxes.xyxy = torch.rand(num_detections, 4) * 640  # Random boxes
        mock_boxes.conf = torch.rand(num_detections) * 0.5 + 0.3  # Confidence 0.3-0.8
        mock_boxes.cls = torch.randint(0, 15, (num_detections,))  # Random classes
        mock_boxes.data = torch.cat([
            mock_boxes.xyxy,
            mock_boxes.conf.unsqueeze(1),
            mock_boxes.cls.unsqueeze(1).float()
        ], dim=1)
        
        mock_result.boxes = mock_boxes
        
        return [mock_result] * batch_size
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        return self
    
    def half(self):
        """Convert to half precision."""
        return self
    
    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        return self
    
    def state_dict(self):
        """Return mock state dict."""
        return {'mock_param': torch.tensor([1.0])}
    
    def load_state_dict(self, state_dict):
        """Load mock state dict."""
        pass
    
    def save(self, path):
        """Save mock model."""
        torch.save(self.state_dict(), path)
    
    def export(self, format="onnx", imgsz=(640, 640)):
        """Mock model export."""
        pass


class TestDetectionModels(unittest.TestCase):
    """Test detection data models and utilities."""
    
    def test_detection_creation(self):
        """Test Detection object creation and methods."""
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=0,
            class_name="PERSON",
            frame_id=0
        )
        
        self.assertEqual(detection.bbox, (100, 100, 200, 200))
        self.assertEqual(detection.confidence, 0.8)
        self.assertEqual(detection.class_name, "PERSON")
        
        # Test area calculation
        self.assertEqual(detection.area(), 10000)
        
        # Test IoU calculation
        other_detection = Detection(
            bbox=(150, 150, 250, 250),
            confidence=0.7,
            class_id=0,
            class_name="PERSON"
        )
        
        iou = detection.iou(other_detection)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
    
    def test_detection_class_enumeration(self):
        """Test DetectionClass enumeration."""
        # Test that all required classes exist
        required_classes = ['PERSON', 'HAND', 'ARM', 'FURNITURE_CHAIR']
        
        for class_name in required_classes:
            self.assertTrue(hasattr(DetectionClass, class_name))
        
        # Test advertisement relevance
        ad_classes = DetectionClass.get_advertisement_relevant_classes()
        self.assertIn(DetectionClass.PERSON, ad_classes)
        self.assertIn(DetectionClass.FURNITURE_CHAIR, ad_classes)
    
    def test_scene_context(self):
        """Test SceneContext data structure."""
        context = SceneContext()
        
        # Default values
        self.assertEqual(context.complexity_score, 0.5)
        self.assertEqual(context.lighting_condition, "mixed")
        self.assertEqual(context.indoor_outdoor, "indoor")
        
        # Test serialization
        context_dict = context.to_dict()
        self.assertIsInstance(context_dict, dict)
        self.assertIn('complexity_score', context_dict)
    
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        config = TrainingConfig(
            model_size="yolov9c",
            epochs=10,
            batch_size=8,
            learning_rate=0.001
        )
        
        # Should not raise any exception
        config.validate()
        
        # Test invalid configuration
        invalid_config = TrainingConfig(
            model_size="invalid_model",
            epochs=-1,
            batch_size=0
        )
        
        with self.assertRaises(ValueError):
            invalid_config.validate()


class TestYOLOv9Config(unittest.TestCase):
    """Test YOLOv9 configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = YOLOv9Config()
        
        self.assertEqual(config.model_path, "yolov9c.pt")
        self.assertEqual(config.input_size, (640, 640))
        self.assertEqual(config.confidence_threshold, 0.25)
        self.assertTrue(config.tensorrt_optimization)
    
    def test_config_modification(self):
        """Test configuration modification."""
        config = YOLOv9Config(
            confidence_threshold=0.5,
            batch_size=16,
            device="cuda"
        )
        
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.device, "cuda")


class TestAdaptiveThresholdManager(unittest.TestCase):
    """Test adaptive threshold management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = YOLOv9Config()
        self.threshold_manager = AdaptiveThresholdManager(self.config)
    
    def test_threshold_initialization(self):
        """Test threshold manager initialization."""
        self.assertEqual(
            self.threshold_manager.base_threshold,
            self.config.confidence_threshold
        )
        self.assertEqual(
            self.threshold_manager.current_threshold,
            self.config.confidence_threshold
        )
    
    def test_threshold_update(self):
        """Test threshold update based on scene context."""
        scene_context = SceneContext(complexity_score=0.8)
        
        # Test with high complexity (should increase threshold)
        new_threshold = self.threshold_manager.update_threshold(
            scene_context, detection_count=10, frame_area=640*640
        )
        
        self.assertIsInstance(new_threshold, float)
        self.assertGreater(new_threshold, 0)
        self.assertLess(new_threshold, 1)
    
    def test_threshold_disabled(self):
        """Test with adaptive threshold disabled."""
        config = YOLOv9Config(scene_adaptive_threshold=False)
        manager = AdaptiveThresholdManager(config)
        
        scene_context = SceneContext(complexity_score=0.9)
        threshold = manager.update_threshold(scene_context, 5, 640*640)
        
        # Should return base threshold when disabled
        self.assertEqual(threshold, config.confidence_threshold)


class TestSceneAnalyzer(unittest.TestCase):
    """Test scene analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scene_analyzer = SceneAnalyzer()
    
    def test_scene_analysis(self):
        """Test scene analysis with synthetic image."""
        # Create synthetic RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        context = self.scene_analyzer.analyze_scene(image_tensor)
        
        self.assertIsInstance(context, SceneContext)
        self.assertGreaterEqual(context.complexity_score, 0)
        self.assertLessEqual(context.complexity_score, 1)
        self.assertIn(context.lighting_condition, ["bright", "dim", "mixed"])
        self.assertIn(context.indoor_outdoor, ["indoor", "outdoor"])
    
    def test_complexity_computation(self):
        """Test complexity score computation."""
        # Simple image (low complexity)
        simple_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        complexity = self.scene_analyzer._compute_complexity(simple_image)
        self.assertLess(complexity, 0.3)
        
        # Complex image (high complexity)
        complex_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        complexity = self.scene_analyzer._compute_complexity(complex_image)
        self.assertGreater(complexity, 0.1)


class TestYOLOv9Detector(unittest.TestCase):
    """Test YOLOv9 detector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = YOLOv9Config(
            model_path="mock_model.pt",
            device="cpu",
            tensorrt_optimization=False,
            half_precision=False
        )
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization."""
        mock_yolo.return_value = MockYOLOModel()
        
        detector = YOLOv9Detector(self.config)
        
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.device.type, 'cpu')
        self.assertIsInstance(detector.threshold_manager, AdaptiveThresholdManager)
        self.assertIsInstance(detector.scene_analyzer, SceneAnalyzer)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_single_frame_detection(self, mock_yolo):
        """Test single frame object detection."""
        mock_yolo.return_value = MockYOLOModel()
        
        detector = YOLOv9Detector(self.config)
        
        # Create test image
        test_image = Image.new('RGB', (640, 640), color='red')
        
        detections = detector.detect_objects(test_image)
        
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertIsInstance(detection, Detection)
            self.assertGreaterEqual(detection.confidence, 0)
            self.assertLessEqual(detection.confidence, 1)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_batch_detection(self, mock_yolo):
        """Test batch object detection."""
        mock_yolo.return_value = MockYOLOModel()
        
        detector = YOLOv9Detector(self.config)
        
        # Create batch of test images
        test_images = [
            Image.new('RGB', (640, 640), color='red'),
            Image.new('RGB', (640, 640), color='green'),
            Image.new('RGB', (640, 640), color='blue')
        ]
        
        detections = detector.detect_objects(test_images)
        
        self.assertIsInstance(detections, list)
        # Should have detections from all frames
        self.assertGreater(len(detections), 0)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    @patch('cv2.VideoCapture')
    def test_video_sequence_detection(self, mock_cap, mock_yolo):
        """Test video sequence detection."""
        mock_yolo.return_value = MockYOLOModel()
        
        # Mock video capture
        mock_cap_instance = Mock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Mock frame reading
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.side_effect = [(True, test_frame)] * 10 + [(False, None)]
        
        detector = YOLOv9Detector(self.config)
        
        detections_generator = detector.detect_sequence("test_video.mp4")
        detections_list = list(detections_generator)
        
        self.assertIsInstance(detections_list, list)
        self.assertGreater(len(detections_list), 0)
        
        # Each element should be a list of detections
        for frame_detections in detections_list:
            self.assertIsInstance(frame_detections, list)
    
    def test_preprocessing(self):
        """Test frame preprocessing."""
        detector = YOLOv9Detector(self.config)
        
        # Test with PIL Image
        pil_image = Image.new('RGB', (800, 600), color='red')
        processed, shapes = detector._preprocess_frames(pil_image)
        
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[-2:], self.config.input_size)
        self.assertEqual(shapes[0], (800, 600))  # Original size
        
        # Test with numpy array
        np_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        processed, shapes = detector._preprocess_frames(np_image)
        
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[-2:], self.config.input_size)
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing functionality."""
        detector = YOLOv9Detector(self.config)
        
        # Create test detections
        detections = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="PERSON"),
            Detection(bbox=(300, 300, 400, 400), confidence=0.7, class_id=1, class_name="HAND")
        ]
        
        # First call - should pass through unchanged
        smoothed = detector._apply_temporal_smoothing(detections)
        self.assertEqual(len(smoothed), len(detections))
        
        # Second call with similar detections - should be smoothed
        similar_detections = [
            Detection(bbox=(105, 105, 205, 205), confidence=0.75, class_id=0, class_name="PERSON"),
            Detection(bbox=(295, 295, 395, 395), confidence=0.72, class_id=1, class_name="HAND")
        ]
        
        smoothed = detector._apply_temporal_smoothing(similar_detections)
        self.assertEqual(len(smoothed), len(similar_detections))
        
        # Check that bounding boxes are smoothed (between original and new)
        original_bbox = detections[0].bbox
        new_bbox = similar_detections[0].bbox
        smoothed_bbox = smoothed[0].bbox
        
        for i in range(4):
            self.assertLessEqual(min(original_bbox[i], new_bbox[i]), smoothed_bbox[i])
            self.assertGreaterEqual(max(original_bbox[i], new_bbox[i]), smoothed_bbox[i])


class TestYOLOv9Trainer(unittest.TestCase):
    """Test YOLOv9 trainer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        self.config = TrainingConfig(
            model_size="yolov9n",
            epochs=2,  # Small for testing
            batch_size=2,
            learning_rate=0.001,
            train_data_path=os.path.join(self.temp_dir, "train"),
            val_data_path=os.path.join(self.temp_dir, "val"),
            experiment_name="test_training",
            device="cpu"
        )
        
        # Create dummy data directories
        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.val_data_path, exist_ok=True)
    
    @patch('video_ad_placement.object_detection.training.trainer.MLFLOW_AVAILABLE', False)
    @patch('video_ad_placement.object_detection.training.trainer.ULTRALYTICS_AVAILABLE', True)
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = YOLOv9Trainer(self.config)
        
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.device.type, 'cpu')
        self.assertEqual(trainer.current_epoch, 0)
        self.assertEqual(trainer.best_map, 0.0)
    
    def test_device_setup(self):
        """Test device setup logic."""
        # Test auto device selection
        config = TrainingConfig(device="auto")
        trainer = YOLOv9Trainer(config)
        
        self.assertIsInstance(trainer.device, torch.device)
        
        # Test specific device
        config = TrainingConfig(device="cpu")
        trainer = YOLOv9Trainer(config)
        
        self.assertEqual(trainer.device.type, 'cpu')
    
    def test_training_metrics(self):
        """Test training metrics tracking."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            map50=0.7,
            map50_95=0.65
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['epoch'], 1.0)
        self.assertEqual(metrics_dict['train_loss'], 0.5)
        self.assertEqual(metrics_dict['map50'], 0.7)
    
    @patch('video_ad_placement.object_detection.training.trainer.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.training.trainer.YOLO')
    def test_model_loading(self, mock_yolo):
        """Test model loading functionality."""
        mock_yolo.return_value = MockYOLOModel()
        
        trainer = YOLOv9Trainer(self.config)
        trainer.load_model()
        
        self.assertIsNotNone(trainer.model)
        mock_yolo.assert_called_once()
    
    def test_optimizer_setup(self):
        """Test optimizer setup."""
        trainer = YOLOv9Trainer(self.config)
        trainer.model = MockYOLOModel()  # Mock model
        
        trainer.setup_optimizer()
        
        self.assertIsNotNone(trainer.optimizer)
        # Scheduler might be None depending on configuration
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        from video_ad_placement.object_detection.training.trainer import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Should not stop initially
        self.assertFalse(early_stopping(1.0))
        self.assertFalse(early_stopping(0.95))  # Improvement
        self.assertFalse(early_stopping(0.94))  # Small improvement
        self.assertFalse(early_stopping(0.945))  # No significant improvement
        self.assertFalse(early_stopping(0.946))  # No significant improvement
        self.assertTrue(early_stopping(0.947))  # Should trigger stopping


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete object detection system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create test configuration file
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        test_config = {
            'model': {
                'model_path': 'yolov9n.pt',
                'input_size': [640, 640],
                'num_classes': 15
            },
            'inference': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'hardware': {
                'device': 'cpu',
                'tensorrt_optimization': False
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_end_to_end_detection(self, mock_yolo):
        """Test end-to-end detection pipeline."""
        mock_yolo.return_value = MockYOLOModel()
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create detector with flattened config
        flat_config = {}
        for section, params in config_dict.items():
            flat_config.update(params)
        
        config = YOLOv9Config(**flat_config)
        detector = YOLOv9Detector(config)
        
        # Create test video frames
        frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Process frames
        all_detections = []
        for frame in frames:
            detections = detector.detect_objects(frame)
            all_detections.extend(detections)
        
        # Verify results
        self.assertIsInstance(all_detections, list)
        for detection in all_detections:
            self.assertIsInstance(detection, Detection)
            self.assertIn(detection.class_name, config.class_names)
    
    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Test that configuration is properly structured
        self.assertIn('model', config_dict)
        self.assertIn('inference', config_dict)
        self.assertIn('hardware', config_dict)
        
        # Test model configuration
        model_config = config_dict['model']
        self.assertIn('model_path', model_config)
        self.assertIn('input_size', model_config)
        self.assertIn('num_classes', model_config)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        config = YOLOv9Config(model_path="nonexistent_model.pt")
        
        with patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True):
            with patch('video_ad_placement.object_detection.yolov9_detector.YOLO') as mock_yolo:
                mock_yolo.side_effect = FileNotFoundError("Model not found")
                
                with self.assertRaises(Exception):
                    detector = YOLOv9Detector(config)
    
    def test_empty_detection_results(self):
        """Test handling of empty detection results."""
        config = YOLOv9Config()
        detector = YOLOv9Detector(config)
        
        # Mock model that returns no detections
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        detector.model = mock_model
        
        # Should handle gracefully
        test_image = Image.new('RGB', (640, 640), color='red')
        detections = detector.detect_objects(test_image)
        
        self.assertIsInstance(detections, list)
    
    def test_gpu_out_of_memory_handling(self):
        """Test GPU out of memory handling."""
        config = YOLOv9Config(device="cuda")
        detector = YOLOv9Detector(config)
        
        # Mock model that raises OOM error
        mock_model = Mock()
        mock_model.side_effect = RuntimeError("CUDA out of memory")
        
        detector.model = mock_model
        
        # Should handle gracefully and return empty results
        test_image = Image.new('RGB', (640, 640), color='red')
        detections = detector.detect_objects(test_image)
        
        self.assertIsInstance(detections, list)
    
    def test_invalid_image_formats(self):
        """Test handling of invalid image formats."""
        config = YOLOv9Config()
        detector = YOLOv9Detector(config)
        
        # Test with invalid image data
        invalid_inputs = [
            None,
            "not_an_image",
            torch.tensor([]),  # Empty tensor
            np.array([]),  # Empty array
        ]
        
        for invalid_input in invalid_inputs:
            try:
                detections = detector.detect_objects(invalid_input)
                # Should either handle gracefully or raise appropriate exception
                self.assertIsInstance(detections, list)
            except (ValueError, TypeError, AttributeError):
                # These are acceptable exceptions for invalid input
                pass


class TestPerformance(unittest.TestCase):
    """Performance and benchmarking tests."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.config = YOLOv9Config(
            device="cpu",
            tensorrt_optimization=False,
            half_precision=False
        )
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_inference_speed(self, mock_yolo):
        """Test inference speed measurement."""
        mock_yolo.return_value = MockYOLOModel()
        
        detector = YOLOv9Detector(self.config)
        
        # Create test image
        test_image = Image.new('RGB', (640, 640), color='red')
        
        # Measure inference time
        import time
        start_time = time.time()
        
        for _ in range(10):
            detections = detector.detect_objects(test_image)
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        
        # Should complete reasonably quickly (less than 1 second per inference on CPU)
        self.assertLess(avg_time, 1.0)
        
        # Get metrics
        metrics = detector.get_metrics()
        self.assertIsInstance(metrics, InferenceMetrics)
    
    @patch('video_ad_placement.object_detection.yolov9_detector.ULTRALYTICS_AVAILABLE', True)
    @patch('video_ad_placement.object_detection.yolov9_detector.YOLO')
    def test_memory_usage(self, mock_yolo):
        """Test memory usage monitoring."""
        mock_yolo.return_value = MockYOLOModel()
        
        detector = YOLOv9Detector(self.config)
        
        # Process multiple images to check for memory leaks
        for i in range(20):
            test_image = Image.new('RGB', (640, 640), color='red')
            detections = detector.detect_objects(test_image)
            
            # Cleanup periodically
            if i % 10 == 0:
                detector._cleanup_cache()
        
        # Should not accumulate excessive memory
        # This is more of a smoke test
        self.assertTrue(True)  # If we get here without OOM, test passes


if __name__ == '__main__':
    # Configure test environment
    os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', 'src')
    
    # Run tests
    unittest.main(verbosity=2) 