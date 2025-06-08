"""
YOLOv9 Object Detection System - Comprehensive Example

This example demonstrates the usage of the YOLOv9-based object detection system
for video advertisement placement scenarios, including:
- Real-time object detection
- Video sequence processing
- Training custom models
- Performance benchmarking
- Different quality profiles
- Integration with tracking systems
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import warnings

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from video_ad_placement.object_detection.yolov9_detector import (
        YOLOv9Detector, YOLOv9Config
    )
    from video_ad_placement.object_detection.detection_models import (
        Detection, DetectionClass, TrainingConfig
    )
    from video_ad_placement.object_detection.training.trainer import YOLOv9Trainer
    
    # Import benchmarking and evaluation components when available
    try:
        from video_ad_placement.object_detection.evaluation.benchmarking import (
            ObjectDetectionBenchmark
        )
        BENCHMARKING_AVAILABLE = True
    except ImportError:
        BENCHMARKING_AVAILABLE = False
        warnings.warn("Benchmarking module not available")
    
    # Import training components when available
    try:
        from video_ad_placement.object_detection.training.dataset import (
            YouTubeDatasetCreator
        )
        DATASET_CREATION_AVAILABLE = True
    except ImportError:
        DATASET_CREATION_AVAILABLE = False
        warnings.warn("Dataset creation module not available")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the object detection module is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_image(width: int = 640, height: int = 640, 
                       scenario: str = "living_room") -> np.ndarray:
    """Create a sample image for testing different scenarios."""
    
    if scenario == "living_room":
        # Create a simple living room scene
        image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light background
        
        # Add furniture rectangles (simplified)
        # Sofa
        cv2.rectangle(image, (50, 300), (300, 450), (139, 69, 19), -1)
        
        # TV
        cv2.rectangle(image, (400, 200), (550, 350), (50, 50, 50), -1)
        
        # Coffee table
        cv2.rectangle(image, (100, 460), (250, 500), (101, 67, 33), -1)
        
        # Add some texture/noise
        noise = np.random.randint(-30, 30, (height, width, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
    elif scenario == "office":
        # Create an office scene
        image = np.ones((height, width, 3), dtype=np.uint8) * 180  # Office background
        
        # Desk
        cv2.rectangle(image, (100, 400), (500, 500), (139, 69, 19), -1)
        
        # Monitor
        cv2.rectangle(image, (200, 250), (400, 390), (30, 30, 30), -1)
        
        # Chair
        cv2.rectangle(image, (150, 350), (250, 500), (80, 80, 80), -1)
        
        # Add texture
        noise = np.random.randint(-20, 20, (height, width, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
    else:  # Random/complex scene
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    return image


def visualize_detections(image: np.ndarray, detections: List[Detection], 
                        save_path: str = None) -> np.ndarray:
    """Visualize detections on an image."""
    
    # Convert to PIL for easier drawing
    if len(image.shape) == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert back to numpy for OpenCV drawing
    vis_image = np.array(pil_image)
    
    # Color map for different classes
    colors = {
        'PERSON': (0, 255, 0),
        'HAND': (255, 0, 0),
        'ARM': (255, 100, 0),
        'FACE': (0, 255, 255),
        'FURNITURE_CHAIR': (128, 0, 128),
        'FURNITURE_TABLE': (255, 0, 255),
        'FURNITURE_SOFA': (128, 128, 0),
        'ELECTRONIC_TV': (0, 128, 255),
        'ELECTRONIC_LAPTOP': (255, 128, 0),
    }
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Get color for class
        color = colors.get(detection.class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Background for text
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(vis_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Visualization saved to {save_path}")
    
    return vis_image


def example_1_basic_detection():
    """Example 1: Basic object detection on a single image."""
    
    logger.info("=== Example 1: Basic Object Detection ===")
    
    # Create configuration
    config = YOLOv9Config(
        model_path="yolov9n.pt",  # Use nano model for faster demo
        device="cpu",  # Use CPU for compatibility
        confidence_threshold=0.25,
        tensorrt_optimization=False  # Disable for demo
    )
    
    try:
        # Initialize detector
        detector = YOLOv9Detector(config)
        
        # Create sample image
        sample_image = create_sample_image(scenario="living_room")
        
        # Run detection
        start_time = time.time()
        detections = detector.detect_objects(sample_image)
        inference_time = time.time() - start_time
        
        logger.info(f"Detected {len(detections)} objects in {inference_time:.3f}s")
        
        # Print detection results
        for i, detection in enumerate(detections):
            logger.info(f"Detection {i+1}: {detection.class_name} "
                       f"(confidence: {detection.confidence:.3f}, "
                       f"bbox: {detection.bbox})")
        
        # Visualize results
        os.makedirs("outputs", exist_ok=True)
        vis_image = visualize_detections(
            sample_image, detections, 
            save_path="outputs/basic_detection_example.jpg"
        )
        
        # Get performance metrics
        metrics = detector.get_metrics()
        logger.info(f"Performance metrics: {metrics.to_dict()}")
        
    except Exception as e:
        logger.error(f"Basic detection example failed: {e}")
        # Create a mock example for demonstration
        logger.info("Running mock detection for demonstration...")
        
        # Mock detections
        mock_detections = [
            Detection(
                bbox=(100, 300, 300, 450),
                confidence=0.85,
                class_id=4,
                class_name="FURNITURE_SOFA"
            ),
            Detection(
                bbox=(400, 200, 550, 350),
                confidence=0.72,
                class_id=9,
                class_name="ELECTRONIC_TV"
            )
        ]
        
        sample_image = create_sample_image(scenario="living_room")
        vis_image = visualize_detections(
            sample_image, mock_detections,
            save_path="outputs/mock_detection_example.jpg"
        )
        
        logger.info(f"Mock detection created {len(mock_detections)} detections")


def example_2_quality_profiles():
    """Example 2: Demonstrate different quality profiles."""
    
    logger.info("=== Example 2: Quality Profiles Comparison ===")
    
    # Load quality profiles from configuration
    config_path = Path(__file__).parent.parent / "configs" / "yolov9_object_detection.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        quality_profiles = full_config.get('quality_profiles', {})
    else:
        # Define quality profiles if config not available
        quality_profiles = {
            'fast': {
                'model': {'model_size': 'yolov9n', 'input_size': [416, 416]},
                'inference': {'confidence_threshold': 0.3},
                'hardware': {'half_precision': True}
            },
            'balanced': {
                'model': {'model_size': 'yolov9s', 'input_size': [640, 640]},
                'inference': {'confidence_threshold': 0.25},
                'hardware': {'half_precision': True}
            },
            'high_quality': {
                'model': {'model_size': 'yolov9c', 'input_size': [832, 832]},
                'inference': {'confidence_threshold': 0.2},
                'hardware': {'half_precision': False}
            }
        }
    
    # Test each quality profile
    results = {}
    sample_image = create_sample_image(scenario="office")
    
    for profile_name, profile_config in quality_profiles.items():
        logger.info(f"Testing {profile_name} profile...")
        
        try:
            # Flatten configuration
            flat_config = {}
            for section, params in profile_config.items():
                flat_config.update(params)
            
            # Add required defaults
            flat_config.update({
                'device': 'cpu',
                'tensorrt_optimization': False
            })
            
            config = YOLOv9Config(**flat_config)
            
            # Simulate detection (since actual models might not be available)
            start_time = time.time()
            
            # Mock timing based on profile complexity
            if profile_name == 'fast':
                time.sleep(0.1)  # Simulate fast inference
                mock_detections = 3
            elif profile_name == 'balanced':
                time.sleep(0.2)  # Simulate medium inference
                mock_detections = 5
            else:  # high_quality
                time.sleep(0.4)  # Simulate slow but accurate inference
                mock_detections = 7
            
            inference_time = time.time() - start_time
            
            results[profile_name] = {
                'inference_time': inference_time,
                'detection_count': mock_detections,
                'input_size': flat_config.get('input_size', [640, 640]),
                'confidence_threshold': flat_config.get('confidence_threshold', 0.25)
            }
            
            logger.info(f"{profile_name}: {mock_detections} detections in {inference_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Failed to test {profile_name} profile: {e}")
    
    # Display comparison
    logger.info("\n=== Quality Profile Comparison ===")
    for profile, result in results.items():
        logger.info(f"{profile:12} | "
                   f"Time: {result['inference_time']:.3f}s | "
                   f"Detections: {result['detection_count']} | "
                   f"Input: {result['input_size']} | "
                   f"Threshold: {result['confidence_threshold']}")


def example_3_video_processing():
    """Example 3: Video sequence processing with temporal optimization."""
    
    logger.info("=== Example 3: Video Sequence Processing ===")
    
    # Create a mock video sequence
    frames = []
    for i in range(10):
        # Create frames with slight variations
        if i < 5:
            frame = create_sample_image(scenario="living_room")
        else:
            frame = create_sample_image(scenario="office")
        
        # Add frame number for identification
        cv2.putText(frame, f"Frame {i+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    # Save frames as a video file for demonstration
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    video_path = output_dir / "sample_video.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 2.0, (640, 640))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    logger.info(f"Sample video created: {video_path}")
    
    # Process video sequence
    try:
        config = YOLOv9Config(
            model_path="yolov9n.pt",
            device="cpu",
            batch_size=4,
            temporal_smoothing=True,
            tensorrt_optimization=False
        )
        
        detector = YOLOv9Detector(config)
        
        # Process video sequence
        logger.info("Processing video sequence...")
        
        frame_detections = []
        detection_generator = detector.detect_sequence(str(video_path))
        
        for frame_idx, detections in enumerate(detection_generator):
            frame_detections.append(detections)
            logger.info(f"Frame {frame_idx + 1}: {len(detections)} detections")
        
        # Analyze temporal consistency
        detection_counts = [len(dets) for dets in frame_detections]
        avg_detections = np.mean(detection_counts)
        std_detections = np.std(detection_counts)
        
        logger.info(f"Temporal analysis:")
        logger.info(f"  Average detections per frame: {avg_detections:.2f}")
        logger.info(f"  Standard deviation: {std_detections:.2f}")
        logger.info(f"  Temporal consistency: {'Good' if std_detections < 2 else 'Variable'}")
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        
        # Mock video processing results
        logger.info("Simulating video processing results...")
        mock_results = [
            [Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="PERSON")],
            [Detection(bbox=(105, 105, 205, 205), confidence=0.82, class_id=0, class_name="PERSON")],
            [Detection(bbox=(110, 100, 210, 200), confidence=0.79, class_id=0, class_name="PERSON")],
        ]
        
        for i, detections in enumerate(mock_results):
            logger.info(f"Mock Frame {i+1}: {len(detections)} detections")


def example_4_training_setup():
    """Example 4: Demonstrate training setup and configuration."""
    
    logger.info("=== Example 4: Training Setup Demonstration ===")
    
    # Create training configuration
    training_config = TrainingConfig(
        # Model configuration
        model_size="yolov9n",  # Start with nano for faster training
        input_size=(640, 640),
        num_classes=15,
        
        # Training parameters
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0005,
        
        # Data paths (would be real paths in practice)
        train_data_path="data/train",
        val_data_path="data/val",
        test_data_path="data/test",
        
        # Augmentation
        mosaic_prob=1.0,
        mixup_prob=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        
        # Advertisement-specific weights
        focus_on_person_detection=True,
        furniture_detection_weight=1.5,
        hand_tracking_weight=2.0,
        
        # Optimization
        optimizer="AdamW",
        scheduler="cosine",
        amp=True,
        
        # Experiment tracking
        project_name="video_ad_placement",
        experiment_name="yolov9_demo_training",
        
        # Hardware
        device="cpu",
        workers=4
    )
    
    logger.info("Training Configuration:")
    config_dict = training_config.to_dict()
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Validate configuration
    try:
        training_config.validate()
        logger.info("✓ Training configuration is valid")
    except ValueError as e:
        logger.error(f"✗ Training configuration invalid: {e}")
    
    # Demonstrate trainer initialization
    try:
        trainer = YOLOv9Trainer(training_config)
        logger.info("✓ Trainer initialized successfully")
        
        # Show training directory structure
        train_dir = trainer.train_dir
        logger.info(f"Training directory: {train_dir}")
        
        # Clean up
        trainer.cleanup()
        
    except Exception as e:
        logger.error(f"✗ Trainer initialization failed: {e}")
    
    # Show data preparation example
    logger.info("\nData Preparation Example:")
    logger.info("1. Collect YouTube videos using YouTubeDatasetCreator")
    logger.info("2. Extract frames at 2-5 FPS")
    logger.info("3. Auto-annotate using pretrained model")
    logger.info("4. Manual review and correction")
    logger.info("5. Split into train/val/test sets")
    logger.info("6. Apply advertisement-specific augmentations")


def example_5_benchmarking():
    """Example 5: Performance benchmarking demonstration."""
    
    logger.info("=== Example 5: Performance Benchmarking ===")
    
    if not BENCHMARKING_AVAILABLE:
        logger.warning("Benchmarking module not available, showing mock results")
        
        # Mock benchmarking results
        benchmark_results = {
            'yolov9n': {
                'inference_time_ms': 15.2,
                'fps': 65.8,
                'memory_mb': 245,
                'map50': 0.68,
                'map50_95': 0.42
            },
            'yolov9s': {
                'inference_time_ms': 28.5,
                'fps': 35.1,
                'memory_mb': 456,
                'map50': 0.73,
                'map50_95': 0.48
            },
            'yolov9c': {
                'inference_time_ms': 45.3,
                'fps': 22.1,
                'memory_mb': 678,
                'map50': 0.78,
                'map50_95': 0.52
            }
        }
        
        logger.info("\nMock Benchmark Results:")
        logger.info("Model    | Inference(ms) | FPS  | Memory(MB) | mAP@0.5 | mAP@0.5:0.95")
        logger.info("-" * 70)
        
        for model, results in benchmark_results.items():
            logger.info(f"{model:8} | {results['inference_time_ms']:11.1f} | "
                       f"{results['fps']:4.1f} | {results['memory_mb']:9} | "
                       f"{results['map50']:7.3f} | {results['map50_95']:11.3f}")
    
    else:
        # Real benchmarking would go here
        logger.info("Running real performance benchmarks...")
        
        try:
            benchmark = ObjectDetectionBenchmark()
            
            # Define test configurations
            test_configs = [
                {'model_size': 'yolov9n', 'input_size': [416, 416]},
                {'model_size': 'yolov9s', 'input_size': [640, 640]},
                {'model_size': 'yolov9c', 'input_size': [640, 640]}
            ]
            
            for config_dict in test_configs:
                config = YOLOv9Config(**config_dict)
                results = benchmark.run_performance_benchmark(config)
                
                logger.info(f"Results for {config_dict['model_size']}:")
                for metric, value in results.items():
                    logger.info(f"  {metric}: {value}")
        
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")


def example_6_integration_demo():
    """Example 6: Integration with tracking system demonstration."""
    
    logger.info("=== Example 6: Integration with Tracking System ===")
    
    # Mock tracking integration
    logger.info("Demonstrating integration with tracking system...")
    
    # Simulate detection results from multiple frames
    tracking_detections = [
        # Frame 1
        [
            Detection(bbox=(100, 100, 200, 200), confidence=0.85, class_id=0, 
                     class_name="PERSON", frame_id=0),
            Detection(bbox=(300, 150, 450, 350), confidence=0.72, class_id=4, 
                     class_name="FURNITURE_CHAIR", frame_id=0)
        ],
        # Frame 2
        [
            Detection(bbox=(105, 105, 205, 205), confidence=0.87, class_id=0, 
                     class_name="PERSON", frame_id=1),
            Detection(bbox=(302, 152, 452, 352), confidence=0.74, class_id=4, 
                     class_name="FURNITURE_CHAIR", frame_id=1)
        ],
        # Frame 3
        [
            Detection(bbox=(110, 100, 210, 200), confidence=0.83, class_id=0, 
                     class_name="PERSON", frame_id=2),
            Detection(bbox=(305, 148, 455, 348), confidence=0.76, class_id=4, 
                     class_name="FURNITURE_CHAIR", frame_id=2)
        ]
    ]
    
    # Simulate tracking ID assignment
    def assign_tracking_ids(detections_sequence):
        """Simple tracking ID assignment based on IoU overlap."""
        
        tracked_objects = {}
        next_id = 1
        
        for frame_detections in detections_sequence:
            for detection in frame_detections:
                best_match_id = None
                best_iou = 0.0
                
                # Find best matching tracked object
                for track_id, tracked_detection in tracked_objects.items():
                    if tracked_detection.class_id == detection.class_id:
                        iou = detection.iou(tracked_detection)
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            best_match_id = track_id
                
                if best_match_id:
                    # Update existing track
                    detection.track_id = best_match_id
                    tracked_objects[best_match_id] = detection
                else:
                    # Create new track
                    detection.track_id = next_id
                    tracked_objects[next_id] = detection
                    next_id += 1
        
        return tracked_objects
    
    # Apply tracking
    tracked_objects = assign_tracking_ids(tracking_detections)
    
    logger.info("Tracking results:")
    for frame_idx, detections in enumerate(tracking_detections):
        logger.info(f"Frame {frame_idx + 1}:")
        for detection in detections:
            logger.info(f"  Track ID {detection.track_id}: {detection.class_name} "
                       f"(conf: {detection.confidence:.3f})")
    
    # Advertisement placement analysis
    logger.info("\nAdvertisement Placement Analysis:")
    
    # Analyze placement suitability
    placement_scores = {}
    for track_id, detection in tracked_objects.items():
        # Simple scoring based on class and confidence
        base_score = detection.confidence
        
        if detection.class_name == "PERSON":
            score = base_score * 1.2  # People are good for ads
        elif "FURNITURE" in detection.class_name:
            score = base_score * 1.1  # Furniture provides context
        else:
            score = base_score
        
        placement_scores[track_id] = score
    
    # Sort by placement suitability
    sorted_placements = sorted(placement_scores.items(), 
                              key=lambda x: x[1], reverse=True)
    
    logger.info("Best advertisement placement targets:")
    for track_id, score in sorted_placements[:3]:
        detection = tracked_objects[track_id]
        logger.info(f"  Track {track_id}: {detection.class_name} "
                   f"(placement score: {score:.3f})")


def main():
    """Run all examples."""
    
    logger.info("YOLOv9 Object Detection System - Comprehensive Example")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    try:
        # Run all examples
        example_1_basic_detection()
        print()
        
        example_2_quality_profiles()
        print()
        
        example_3_video_processing()
        print()
        
        example_4_training_setup()
        print()
        
        example_5_benchmarking()
        print()
        
        example_6_integration_demo()
        print()
        
        logger.info("All examples completed successfully!")
        logger.info("Check the 'outputs' directory for generated files.")
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Examples failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 