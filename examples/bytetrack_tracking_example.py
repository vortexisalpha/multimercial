"""
Comprehensive ByteTrack Multi-Object Tracking Example

This example demonstrates the complete ByteTracker system for video advertisement
placement scenarios, including detection integration, tracking evaluation,
and visualization.
"""

import cv2
import numpy as np
import torch
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from video_ad_placement.object_detection.yolov9_detector import YOLOv9Detector
from video_ad_placement.object_detection.detection_models import Detection, DetectionClass
from video_ad_placement.object_detection.tracking.byte_tracker import ByteTracker, TrackingConfig
from video_ad_placement.object_detection.tracking.track_models import Track, MotionModel
from video_ad_placement.object_detection.tracking.motion_models import MotionParameters
from video_ad_placement.object_detection.tracking.evaluation import TrackingEvaluator
from video_ad_placement.object_detection.tracking.visualization import TrackingVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockVideoSource:
    """Mock video source for demonstration."""
    
    def __init__(self, width: int = 1280, height: int = 720, num_frames: int = 100):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frame_count = 0
        
        # Simulate moving objects
        self.objects = [
            {
                'start_pos': (100, 100),
                'velocity': (2, 1),
                'size': (80, 120),  # Person-like
                'class': DetectionClass.PERSON,
                'confidence': 0.85
            },
            {
                'start_pos': (300, 200),
                'velocity': (-1, 0.5),
                'size': (60, 60),
                'class': DetectionClass.FURNITURE_CHAIR,
                'confidence': 0.75
            },
            {
                'start_pos': (500, 150),
                'velocity': (1, -0.5),
                'size': (40, 80),
                'class': DetectionClass.HAND,
                'confidence': 0.70
            }
        ]
    
    def get_frame(self) -> np.ndarray:
        """Generate a mock frame with moving objects."""
        if self.frame_count >= self.num_frames:
            return None
        
        # Create frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame.fill(50)  # Dark gray background
        
        # Add some noise for realism
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Draw objects
        for obj_id, obj in enumerate(self.objects):
            # Calculate current position
            curr_x = obj['start_pos'][0] + obj['velocity'][0] * self.frame_count
            curr_y = obj['start_pos'][1] + obj['velocity'][1] * self.frame_count
            
            # Wrap around screen edges
            curr_x = curr_x % (self.width - obj['size'][0])
            curr_y = curr_y % (self.height - obj['size'][1])
            
            # Draw object as colored rectangle
            color = (0, 255, 0) if obj['class'] == DetectionClass.PERSON else (255, 0, 0)
            if obj['class'] == DetectionClass.HAND:
                color = (0, 0, 255)
            
            x1, y1 = int(curr_x), int(curr_y)
            x2, y2 = int(curr_x + obj['size'][0]), int(curr_y + obj['size'][1])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        self.frame_count += 1
        return frame
    
    def get_mock_detections(self) -> List[Detection]:
        """Generate mock detections for current frame."""
        detections = []
        
        for obj_id, obj in enumerate(self.objects):
            # Calculate current position
            curr_x = obj['start_pos'][0] + obj['velocity'][0] * (self.frame_count - 1)
            curr_y = obj['start_pos'][1] + obj['velocity'][1] * (self.frame_count - 1)
            
            # Wrap around screen edges
            curr_x = curr_x % (self.width - obj['size'][0])
            curr_y = curr_y % (self.height - obj['size'][1])
            
            # Add some noise to detection
            noise_x = np.random.normal(0, 2)
            noise_y = np.random.normal(0, 2)
            
            x1 = curr_x + noise_x
            y1 = curr_y + noise_y
            x2 = x1 + obj['size'][0]
            y2 = y1 + obj['size'][1]
            
            # Occasionally miss detections or add false positives
            if np.random.random() > 0.1:  # 90% detection rate
                # Add confidence noise
                conf_noise = np.random.normal(0, 0.05)
                confidence = max(0.1, min(0.99, obj['confidence'] + conf_noise))
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=obj['class'].value,
                    class_name=obj['class'].name,
                    frame_id=self.frame_count - 1
                )
                detections.append(detection)
        
        # Occasionally add false positive
        if np.random.random() > 0.95:
            fp_x = np.random.uniform(0, self.width - 50)
            fp_y = np.random.uniform(0, self.height - 50)
            
            fp_detection = Detection(
                bbox=(fp_x, fp_y, fp_x + 50, fp_y + 50),
                confidence=0.3,
                class_id=DetectionClass.ELECTRONICS_TV.value,
                class_name=DetectionClass.ELECTRONICS_TV.name,
                frame_id=self.frame_count - 1
            )
            detections.append(fp_detection)
        
        return detections


def create_tracking_config() -> TrackingConfig:
    """Create optimized tracking configuration for advertisement scenarios."""
    
    motion_params = MotionParameters(
        process_noise_scale=1.0,
        measurement_noise_scale=0.1,
        position_std=1.0,
        velocity_std=0.1
    )
    
    config = TrackingConfig(
        # Detection thresholds optimized for advertisement content
        high_threshold=0.6,
        low_threshold=0.2,
        match_threshold=0.7,
        second_match_threshold=0.4,
        
        # Track management for video content
        track_buffer_size=30,  # 1 second at 30fps
        min_box_area=100.0,
        aspect_ratio_thresh=10.0,  # Allow tall/wide objects
        
        # Frame rate settings
        frame_rate=30,
        new_track_thresh=0.6,
        
        # Motion model configuration
        motion_model=MotionModel.ADAPTIVE,
        motion_parameters=motion_params,
        
        # Re-identification for person tracking
        use_reid=True,
        reid_weight=0.2,
        appearance_thresh=0.3,
        
        # Camera motion compensation
        use_camera_motion_compensation=True,
        
        # Advertisement-specific optimizations
        prioritize_person_tracks=True,
        furniture_stability_bonus=1.5,
        
        # Performance settings
        max_tracks=50,
        cleanup_interval=30,
        
        # Occlusion handling
        occlusion_detection=True,
        occlusion_threshold=0.6,
        max_occlusion_frames=15,
        
        # Temporal consistency
        temporal_smoothing=True,
        smoothing_alpha=0.8,
        interpolation_enabled=True
    )
    
    return config


def demonstrate_basic_tracking():
    """Demonstrate basic ByteTracker functionality."""
    print("\n" + "="*60)
    print("BASIC BYTETRACKER DEMONSTRATION")
    print("="*60)
    
    # Create tracking configuration
    config = create_tracking_config()
    
    # Initialize tracker
    tracker = ByteTracker(config)
    
    # Create mock video source
    video_source = MockVideoSource(width=1280, height=720, num_frames=50)
    
    # Initialize visualizer
    visualizer = TrackingVisualizer(frame_size=(1280, 720))
    
    # Track objects through video
    all_tracks = []
    all_frames = []
    
    print("Processing frames...")
    for frame_id in range(50):
        # Get frame and detections
        frame = video_source.get_frame()
        if frame is None:
            break
        
        detections = video_source.get_mock_detections()
        
        # Update tracker
        tracks = tracker.update(detections, frame_id, frame)
        
        # Store for analysis
        all_tracks.append(tracks.copy())
        all_frames.append(frame.copy())
        
        # Print frame statistics
        if frame_id % 10 == 0:
            stats = tracker.get_tracking_statistics()
            print(f"Frame {frame_id}: {len(tracks)} active tracks, "
                  f"{stats['total_tracks_created']} total created")
    
    # Print final statistics
    final_stats = tracker.get_tracking_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total tracks created: {final_stats['total_tracks_created']}")
    print(f"  Active tracks: {final_stats['active_tracks']}")
    print(f"  Lost tracks: {final_stats['lost_tracks']}")
    print(f"  Average processing time: {final_stats['average_processing_time']:.4f}s")
    print(f"  Quality distribution: {final_stats['track_quality_distribution']}")
    
    return tracker, all_tracks, all_frames, visualizer


def demonstrate_evaluation():
    """Demonstrate tracking evaluation with ground truth."""
    print("\n" + "="*60)
    print("TRACKING EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create tracking system
    config = create_tracking_config()
    tracker = ByteTracker(config)
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # Create video source
    video_source = MockVideoSource(width=1280, height=720, num_frames=30)
    
    # Process video with evaluation
    print("Processing frames with evaluation...")
    frame_metrics = []
    
    for frame_id in range(30):
        frame = video_source.get_frame()
        if frame is None:
            break
        
        # Get detections (these serve as both input and ground truth for demo)
        detections = video_source.get_mock_detections()
        
        # Add ground truth track IDs
        for i, det in enumerate(detections):
            det.track_id = i  # Simple ground truth assignment
        
        # Update tracker
        predicted_tracks = tracker.update(detections, frame_id, frame)
        
        # Evaluate frame
        metrics = evaluator.evaluate_frame(predicted_tracks, detections, frame_id)
        frame_metrics.append(metrics)
        
        if frame_id % 10 == 0:
            print(f"Frame {frame_id}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")
    
    # Compute summary metrics
    summary_metrics = evaluator.compute_summary_metrics()
    mot_metrics = evaluator.compute_mot_metrics()
    
    print(f"\nSummary Metrics:")
    print(f"  MOTA: {summary_metrics.mota:.3f}")
    print(f"  MOTP: {summary_metrics.motp:.3f}")
    print(f"  Precision: {summary_metrics.precision:.3f}")
    print(f"  Recall: {summary_metrics.recall:.3f}")
    print(f"  F1 Score: {summary_metrics.f1_score:.3f}")
    print(f"  ID Switches: {summary_metrics.id_switches}")
    print(f"  Fragmentations: {summary_metrics.fragments}")
    print(f"  Track Stability: {summary_metrics.track_stability:.3f}")
    
    print(f"\nMOT Challenge Metrics:")
    print(f"  MT (Mostly Tracked): {mot_metrics.mt}")
    print(f"  PT (Partially Tracked): {mot_metrics.pt}")
    print(f"  ML (Mostly Lost): {mot_metrics.ml}")
    
    return frame_metrics, summary_metrics, evaluator


def demonstrate_visualization(tracker, all_tracks, all_frames, visualizer):
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create output directory
    output_dir = Path("tracking_output")
    output_dir.mkdir(exist_ok=True)
    
    # Visualize a few sample frames
    print("Creating sample frame visualizations...")
    sample_frames = [0, 10, 20, 30, 40]
    
    for frame_idx in sample_frames:
        if frame_idx < len(all_frames):
            frame = all_frames[frame_idx]
            tracks = all_tracks[frame_idx]
            
            # Create visualization
            vis_frame = visualizer.visualize_frame(
                frame, tracks,
                show_trails=True,
                show_metadata=True
            )
            
            # Save frame
            output_path = output_dir / f"frame_{frame_idx:03d}.jpg"
            cv2.imwrite(str(output_path), vis_frame)
    
    print(f"Sample frames saved to: {output_dir}")
    
    # Create trajectory analysis
    print("Creating trajectory analysis...")
    all_tracks_flat = []
    for tracks in all_tracks:
        all_tracks_flat.extend(tracks)
    
    # Remove duplicates (same track across frames)
    unique_tracks = {}
    for track in all_tracks_flat:
        if track.track_id not in unique_tracks:
            unique_tracks[track.track_id] = track
        else:
            # Merge trajectory histories
            existing_track = unique_tracks[track.track_id]
            existing_track.bbox_history.extend(track.bbox_history)
            existing_track.confidence_history.extend(track.confidence_history)
    
    trajectory_fig = visualizer.plot_trajectory_analysis(
        list(unique_tracks.values()),
        save_path=str(output_dir / "trajectory_analysis.png")
    )
    plt.close(trajectory_fig)
    
    # Create track heatmap
    print("Creating track heatmap...")
    heatmap = visualizer.create_track_heatmap(
        list(unique_tracks.values()),
        frame_size=(1280, 720),
        save_path=str(output_dir / "track_heatmap.jpg")
    )
    
    # Save tracking summary
    summary_data = tracker.get_tracking_statistics()
    visualizer.save_tracking_summary(
        list(unique_tracks.values()),
        summary_data,
        str(output_dir)
    )
    
    print("Visualization outputs saved!")
    return output_dir


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between different configurations."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Test different configurations
    configs = {
        'Fast': TrackingConfig(
            high_threshold=0.7,
            low_threshold=0.3,
            match_threshold=0.6,
            temporal_smoothing=False,
            use_reid=False,
            use_camera_motion_compensation=False
        ),
        'Balanced': create_tracking_config(),
        'Accurate': TrackingConfig(
            high_threshold=0.5,
            low_threshold=0.1,
            match_threshold=0.8,
            second_match_threshold=0.6,
            temporal_smoothing=True,
            use_reid=True,
            track_buffer_size=50,
            smoothing_alpha=0.9
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        tracker = ByteTracker(config)
        video_source = MockVideoSource(width=1280, height=720, num_frames=30)
        
        start_time = time.time()
        
        for frame_id in range(30):
            frame = video_source.get_frame()
            if frame is None:
                break
            
            detections = video_source.get_mock_detections()
            tracks = tracker.update(detections, frame_id, frame)
        
        end_time = time.time()
        stats = tracker.get_tracking_statistics()
        
        results[config_name] = {
            'processing_time': end_time - start_time,
            'avg_frame_time': stats['average_processing_time'],
            'total_tracks': stats['total_tracks_created'],
            'active_tracks': stats['active_tracks'],
            'quality_dist': stats['track_quality_distribution']
        }
        
        print(f"  Total time: {results[config_name]['processing_time']:.3f}s")
        print(f"  Avg frame time: {results[config_name]['avg_frame_time']:.4f}s")
        print(f"  Total tracks: {results[config_name]['total_tracks']}")
    
    # Print comparison
    print(f"\nPerformance Comparison:")
    print(f"{'Config':<12} {'Total Time':<12} {'Frame Time':<12} {'Tracks':<8}")
    print("-" * 50)
    
    for config_name, result in results.items():
        print(f"{config_name:<12} {result['processing_time']:<12.3f} "
              f"{result['avg_frame_time']:<12.4f} {result['total_tracks']:<8}")
    
    return results


def demonstrate_integration_with_yolov9():
    """Demonstrate integration with YOLOv9 detector."""
    print("\n" + "="*60)
    print("YOLOV9 INTEGRATION DEMONSTRATION")
    print("="*60)
    
    try:
        # Note: This requires a trained YOLOv9 model
        # For demo purposes, we'll simulate the integration
        
        print("Simulating YOLOv9 + ByteTracker integration...")
        
        # Create tracking configuration optimized for YOLOv9 outputs
        config = TrackingConfig(
            high_threshold=0.6,  # YOLOv9 confidence threshold
            low_threshold=0.2,
            match_threshold=0.7,
            use_reid=True,
            prioritize_person_tracks=True,
            furniture_stability_bonus=1.3
        )
        
        tracker = ByteTracker(config)
        video_source = MockVideoSource(width=1920, height=1080, num_frames=20)
        
        print("Processing video with simulated YOLOv9 detections...")
        
        for frame_id in range(20):
            frame = video_source.get_frame()
            if frame is None:
                break
            
            # Simulate YOLOv9 detection (in practice, would be:)
            # detections = yolov9_detector.detect_objects(frame)
            detections = video_source.get_mock_detections()
            
            # Update tracker
            tracks = tracker.update(detections, frame_id, frame)
            
            # Process tracks for advertisement placement
            ad_relevant_tracks = []
            for track in tracks:
                if track.metadata.ad_relevance_score > 0.6:
                    ad_relevant_tracks.append(track)
            
            if frame_id % 5 == 0:
                print(f"Frame {frame_id}: {len(tracks)} total tracks, "
                      f"{len(ad_relevant_tracks)} ad-relevant")
        
        final_stats = tracker.get_tracking_statistics()
        print(f"\nIntegration Results:")
        print(f"  Total tracks processed: {final_stats['total_tracks_created']}")
        print(f"  Average processing time: {final_stats['average_processing_time']:.4f}s")
        print(f"  Final active tracks: {final_stats['active_tracks']}")
        
        return tracker
        
    except Exception as e:
        print(f"Integration demo failed: {e}")
        print("This requires proper YOLOv9 model setup.")
        return None


def main():
    """Run all ByteTracker demonstrations."""
    print("ByteTracker Multi-Object Tracking System Demonstration")
    print("=" * 60)
    
    try:
        # Basic tracking demonstration
        tracker, all_tracks, all_frames, visualizer = demonstrate_basic_tracking()
        
        # Evaluation demonstration
        frame_metrics, summary_metrics, evaluator = demonstrate_evaluation()
        
        # Visualization demonstration
        output_dir = demonstrate_visualization(tracker, all_tracks, all_frames, visualizer)
        
        # Performance comparison
        performance_results = demonstrate_performance_comparison()
        
        # YOLOv9 integration
        integrated_tracker = demonstrate_integration_with_yolov9()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print(f"Output files saved to: {output_dir}")
        print("\nKey Features Demonstrated:")
        print("✓ ByteTracker algorithm with two-stage association")
        print("✓ Advanced motion models and Kalman filtering")
        print("✓ Occlusion handling and re-identification")
        print("✓ Temporal consistency and smoothing")
        print("✓ Comprehensive evaluation metrics")
        print("✓ Rich visualization capabilities")
        print("✓ Performance optimization")
        print("✓ Advertisement-specific optimizations")
        print("✓ Integration with detection systems")
        
        print(f"\nFinal Performance Summary:")
        print(f"  MOTA Score: {summary_metrics.mota:.3f}")
        print(f"  Track Stability: {summary_metrics.track_stability:.3f}")
        print(f"  Ad Relevance Score: {summary_metrics.advertisement_relevance_score:.3f}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 