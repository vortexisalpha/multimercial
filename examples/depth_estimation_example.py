#!/usr/bin/env python3
"""
Comprehensive Example: AI-Powered Video Advertisement Placement with Marigold Depth Estimation

This example demonstrates the complete usage of the Marigold depth estimation system
including temporal consistency, video processing, visualization, and performance monitoring.
"""

import os
import sys
import time
from pathlib import Path
import logging
import argparse

import torch
import numpy as np
import cv2
from PIL import Image

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ad_placement.depth_estimation import (
    MarigoldDepthEstimator,
    DepthEstimationConfig,
    PrecisionMode,
    QualityMode,
    TemporalSmoothingMethod
)

from video_ad_placement.depth_estimation.utils import (
    DepthVisualization,
    VideoDepthProcessor,
    DepthMetricsCollector,
    VideoProcessingConfig
)

from video_ad_placement.depth_estimation.temporal_consistency import (
    TemporalConsistencyProcessor,
    TemporalFilterConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_video(output_path: str, duration_seconds: int = 10, fps: int = 30, resolution: tuple = (720, 1280)):
    """Create a test video for depth estimation."""
    height, width = resolution
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    logger.info(f"Creating test video: {total_frames} frames at {fps} FPS")
    
    for frame_idx in range(total_frames):
        # Create a simple animated scene
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving gradient background
        for y in range(height):
            for x in range(width):
                # Create moving pattern
                val = int(128 + 127 * np.sin((x + frame_idx * 2) * 0.01) * np.cos((y + frame_idx) * 0.01))
                frame[y, x] = [val, val // 2, val // 3]
        
        # Add moving objects
        # Circle moving horizontally
        circle_x = int((width / 2) + (width / 4) * np.sin(frame_idx * 0.1))
        circle_y = height // 3
        cv2.circle(frame, (circle_x, circle_y), 50, (255, 255, 255), -1)
        
        # Rectangle moving vertically
        rect_x = width * 2 // 3
        rect_y = int((height / 2) + (height / 4) * np.cos(frame_idx * 0.08))
        cv2.rectangle(frame, (rect_x - 40, rect_y - 30), (rect_x + 40, rect_y + 30), (0, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Test video created: {output_path}")


def example_single_frame_depth_estimation():
    """Example: Single frame depth estimation."""
    logger.info("=== Single Frame Depth Estimation Example ===")
    
    # Configuration
    config = {
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'quality_mode': QualityMode.BALANCED,
        'precision_mode': PrecisionMode.FP16,
        'batch_size': 4,
        'max_resolution': (720, 1280),
        'enable_temporal_consistency': False,
        'log_gpu_memory': True
    }
    
    try:
        # Initialize depth estimator
        estimator = MarigoldDepthEstimator(config)
        
        # Create test image
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Add some structure to the image
        cv2.rectangle(test_image, (200, 200), (600, 500), (255, 255, 255), -1)
        cv2.circle(test_image, (640, 360), 100, (128, 128, 128), -1)
        
        logger.info("Estimating depth for single frame...")
        start_time = time.time()
        
        # Estimate depth
        depth_map = estimator.estimate_depth(test_image)
        
        processing_time = time.time() - start_time
        logger.info(f"Depth estimation completed in {processing_time:.2f}s")
        
        # Get metrics
        metrics = estimator.get_metrics()
        logger.info(f"Processing metrics: {metrics}")
        
        # Visualize results
        visualizer = DepthVisualization()
        
        # Save visualizations
        os.makedirs("output/single_frame", exist_ok=True)
        
        # Original image
        cv2.imwrite("output/single_frame/original.jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        
        # Depth visualization
        visualizer.save_depth_visualization(
            depth_map, 
            "output/single_frame/depth.png",
            title="Estimated Depth Map"
        )
        
        # Side-by-side comparison
        side_by_side = visualizer.create_side_by_side(test_image, depth_map)
        cv2.imwrite("output/single_frame/comparison.jpg", cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        
        logger.info("Single frame results saved to output/single_frame/")
        
    except Exception as e:
        logger.error(f"Single frame example failed: {e}")
        # Use dummy depth map for demonstration
        depth_map = torch.randn(720, 1280)
        logger.info("Using dummy depth map for visualization demo")
        
        visualizer = DepthVisualization()
        os.makedirs("output/single_frame", exist_ok=True)
        visualizer.save_depth_visualization(depth_map, "output/single_frame/dummy_depth.png")


def example_video_depth_estimation():
    """Example: Video depth estimation with temporal consistency."""
    logger.info("=== Video Depth Estimation Example ===")
    
    # Create test video
    test_video_path = "output/test_video.mp4"
    os.makedirs("output", exist_ok=True)
    create_test_video(test_video_path, duration_seconds=5, fps=10)  # Short video for demo
    
    # Configuration for video processing
    depth_config = {
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'quality_mode': QualityMode.FAST,  # Fast mode for demo
        'precision_mode': PrecisionMode.FP16,
        'batch_size': 2,
        'max_resolution': (360, 640),  # Smaller resolution for demo
        'enable_temporal_consistency': True,
        'temporal_smoothing_method': TemporalSmoothingMethod.EXPONENTIAL,
        'temporal_alpha': 0.3,
        'adaptive_batch_size': True
    }
    
    video_config = VideoProcessingConfig(
        output_fps=10,
        visualization_mode="side_by_side",
        depth_colormap="plasma",
        save_individual_frames=True,
        frame_output_dir="output/video_frames"
    )
    
    try:
        # Initialize components
        estimator = MarigoldDepthEstimator(depth_config)
        video_processor = VideoDepthProcessor(video_config)
        metrics_collector = DepthMetricsCollector()
        
        logger.info("Processing video with depth estimation...")
        
        # Process video
        output_video_path = "output/depth_video.mp4"
        stats = video_processor.process_video_with_depth(
            test_video_path,
            estimator,
            output_video_path,
            end_frame=30  # Process first 30 frames for demo
        )
        
        logger.info(f"Video processing completed: {stats}")
        
        # Collect detailed metrics
        for frame_stats in stats['depth_stats']:
            metrics_collector.collect_frame_metrics(
                frame_stats['frame_idx'],
                torch.tensor(frame_stats['mean_depth']),  # Simplified for demo
                0.1  # Mock processing time
            )
        
        # Generate metrics report
        sequence_metrics = metrics_collector.compute_sequence_metrics()
        report = metrics_collector.get_summary_report()
        
        logger.info("Metrics Report:")
        logger.info(report)
        
        # Save metrics
        metrics_collector.save_metrics("output/video_metrics.json")
        metrics_collector.create_metrics_visualization("output/metrics_plot.png")
        
        logger.info("Video processing results saved to output/")
        
    except Exception as e:
        logger.error(f"Video processing example failed: {e}")
        logger.info("This is expected if Marigold dependencies are not installed")


def example_temporal_consistency():
    """Example: Temporal consistency processing."""
    logger.info("=== Temporal Consistency Example ===")
    
    # Configuration
    temporal_config = TemporalFilterConfig(
        window_size=5,
        alpha=0.3,
        bilateral_sigma_intensity=0.1,
        motion_threshold=0.2
    )
    
    # Initialize processor
    processor = TemporalConsistencyProcessor(temporal_config)
    
    # Simulate sequence of depth maps and frames
    sequence_length = 20
    depth_maps = []
    frames = []
    
    logger.info(f"Processing temporal sequence of {sequence_length} frames...")
    
    for i in range(sequence_length):
        # Create synthetic depth map with some temporal correlation
        base_depth = torch.randn(100, 100)
        if i > 0:
            # Add temporal correlation
            base_depth = 0.7 * depth_maps[-1] + 0.3 * base_depth
        
        # Create corresponding frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Process with temporal consistency
        consistent_depth = processor.process_frame(base_depth, frame, i)
        
        depth_maps.append(consistent_depth)
        frames.append(frame)
    
    # Analyze temporal consistency
    consistency_score = processor.compute_temporal_consistency_score()
    stats = processor.get_processing_stats()
    
    logger.info(f"Temporal consistency score: {consistency_score:.3f}")
    logger.info(f"Processing stats: {stats}")
    
    # Visualize temporal evolution
    os.makedirs("output/temporal", exist_ok=True)
    
    visualizer = DepthVisualization()
    for i, depth_map in enumerate(depth_maps[::5]):  # Sample every 5th frame
        visualizer.save_depth_visualization(
            depth_map,
            f"output/temporal/depth_frame_{i:03d}.png",
            title=f"Frame {i * 5} - Temporally Consistent Depth"
        )
    
    logger.info("Temporal consistency results saved to output/temporal/")


def example_performance_benchmark():
    """Example: Performance benchmarking."""
    logger.info("=== Performance Benchmark Example ===")
    
    # Test different configurations
    configs = [
        ('Fast', {'quality_mode': QualityMode.FAST, 'batch_size': 8}),
        ('Balanced', {'quality_mode': QualityMode.BALANCED, 'batch_size': 4}),
        ('High Quality', {'quality_mode': QualityMode.HIGH_QUALITY, 'batch_size': 2}),
    ]
    
    results = []
    test_frames = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    for config_name, config_params in configs:
        logger.info(f"Benchmarking {config_name} configuration...")
        
        try:
            config = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'max_resolution': (360, 640),
                'enable_temporal_consistency': False,
                'benchmark_mode': True,
                **config_params
            }
            
            estimator = MarigoldDepthEstimator(config)
            
            # Warmup
            estimator.estimate_depth(test_frames[0])
            
            # Benchmark
            start_time = time.time()
            for frame in test_frames:
                depth_map = estimator.estimate_depth(frame)
            
            total_time = time.time() - start_time
            fps = len(test_frames) / total_time
            
            metrics = estimator.get_metrics()
            
            result = {
                'config': config_name,
                'total_time': total_time,
                'fps': fps,
                'avg_processing_time': total_time / len(test_frames),
                'gpu_memory': metrics.gpu_memory_peak if hasattr(metrics, 'gpu_memory_peak') else 0
            }
            
            results.append(result)
            logger.info(f"{config_name}: {fps:.2f} FPS, {result['avg_processing_time']:.3f}s per frame")
            
        except Exception as e:
            logger.error(f"Benchmark for {config_name} failed: {e}")
            results.append({'config': config_name, 'error': str(e)})
    
    # Save benchmark results
    import json
    with open("output/benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Benchmark results saved to output/benchmark_results.json")


def example_quality_assessment():
    """Example: Depth quality assessment and metrics."""
    logger.info("=== Quality Assessment Example ===")
    
    from video_ad_placement.depth_estimation import compute_depth_quality_metrics
    
    # Create test depth maps
    high_quality_depth = torch.randn(200, 200) * 5 + 10  # Realistic depth range
    noisy_depth = high_quality_depth + torch.randn(200, 200) * 2  # Add noise
    
    # Assess quality
    hq_metrics = compute_depth_quality_metrics(high_quality_depth)
    noisy_metrics = compute_depth_quality_metrics(noisy_depth, high_quality_depth)
    
    logger.info("High Quality Depth Metrics:")
    for key, value in hq_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Noisy Depth Metrics (vs reference):")
    for key, value in noisy_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Visualize quality comparison
    visualizer = DepthVisualization()
    os.makedirs("output/quality", exist_ok=True)
    
    visualizer.save_depth_visualization(
        high_quality_depth,
        "output/quality/high_quality.png",
        title="High Quality Depth"
    )
    
    visualizer.save_depth_visualization(
        noisy_depth,
        "output/quality/noisy_depth.png",
        title="Noisy Depth"
    )
    
    # Error map
    error_map = torch.abs(high_quality_depth - noisy_depth)
    visualizer.save_depth_visualization(
        error_map,
        "output/quality/error_map.png",
        title="Absolute Error Map"
    )
    
    logger.info("Quality assessment results saved to output/quality/")


def main():
    """Main example runner."""
    parser = argparse.ArgumentParser(description="Marigold Depth Estimation Examples")
    parser.add_argument('--example', choices=[
        'single_frame', 'video', 'temporal', 'benchmark', 'quality', 'all'
    ], default='all', help='Which example to run')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting Marigold Depth Estimation Examples")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    examples = {
        'single_frame': example_single_frame_depth_estimation,
        'video': example_video_depth_estimation,
        'temporal': example_temporal_consistency,
        'benchmark': example_performance_benchmark,
        'quality': example_quality_assessment
    }
    
    if args.example == 'all':
        for name, func in examples.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {name} example")
            logger.info(f"{'='*50}")
            try:
                func()
            except Exception as e:
                logger.error(f"Example {name} failed: {e}")
                logger.info("Continuing with next example...")
    else:
        if args.example in examples:
            examples[args.example]()
        else:
            logger.error(f"Unknown example: {args.example}")
    
    logger.info("\nAll examples completed! Check the output/ directory for results.")
    logger.info("\nNote: Some examples may fail if Marigold dependencies are not installed.")
    logger.info("Install with: pip install diffusers transformers")


if __name__ == "__main__":
    main() 