"""
Comprehensive Example: Ensemble Depth Estimation with Marigold and Depth Pro

This example demonstrates how to use the ensemble depth estimation system
combining Marigold and Apple's Depth Pro models with various fusion methods,
benchmarking, and performance optimization.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

# Add source directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_ad_placement.depth_estimation.marigold_estimator import MarigoldDepthEstimator
from src.video_ad_placement.depth_estimation.ensemble_estimator import (
    EnsembleDepthEstimator,
    EnsembleConfig,
    FusionMethod
)
from src.video_ad_placement.depth_estimation.benchmarking import (
    DepthEstimationBenchmark,
    BenchmarkConfig,
    BenchmarkMode
)
from src.video_ad_placement.depth_estimation.utils import (
    create_depth_visualization,
    save_depth_prediction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDepthProEstimator:
    """Mock Depth Pro estimator for demonstration purposes."""
    
    def __init__(self, config=None):
        """Initialize mock estimator."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        logger.info("MockDepthProEstimator initialized (replace with real implementation)")
    
    def estimate_depth(self, frames: torch.Tensor):
        """Mock depth estimation with confidence."""
        batch_size, channels, height, width = frames.shape
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Generate realistic depth pattern
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing='ij'
        )
        
        # Create depth pattern with perspective effect
        depth_base = 2.0 + 3.0 * y_coords  # Closer at top, farther at bottom
        depth_noise = torch.randn(height, width) * 0.1
        depth = depth_base + depth_noise
        
        # Expand to batch
        depth = depth.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Generate confidence based on gradient
        gradient_x = torch.diff(depth, dim=-1, prepend=depth[..., :1])
        gradient_y = torch.diff(depth, dim=-2, prepend=depth[..., :1, :])
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        
        # Higher confidence in smoother regions
        confidence = torch.exp(-gradient_magnitude * 5).clamp(0.3, 0.95)
        
        return depth.to(self.device), confidence.to(self.device)


def create_test_images(num_images: int = 5) -> List[torch.Tensor]:
    """Create test images for demonstration."""
    logger.info(f"Creating {num_images} test images")
    
    test_images = []
    
    for i in range(num_images):
        # Create diverse test patterns
        if i == 0:
            # Simple gradient
            image = torch.zeros(3, 480, 640)
            for c in range(3):
                image[c] = torch.linspace(0, 1, 640).unsqueeze(0).repeat(480, 1)
        
        elif i == 1:
            # Checkerboard pattern
            image = torch.zeros(3, 480, 640)
            for y in range(0, 480, 32):
                for x in range(0, 640, 32):
                    if (y // 32 + x // 32) % 2 == 0:
                        image[:, y:y+32, x:x+32] = 1.0
        
        elif i == 2:
            # Random texture
            image = torch.rand(3, 480, 640)
        
        elif i == 3:
            # Geometric shapes
            image = torch.zeros(3, 480, 640)
            # Circle
            center_y, center_x = 240, 320
            y_coords, x_coords = torch.meshgrid(
                torch.arange(480), torch.arange(640), indexing='ij'
            )
            distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            circle_mask = distance < 100
            image[:, circle_mask] = 0.8
        
        else:
            # Complex scene simulation
            image = torch.randn(3, 480, 640) * 0.1 + 0.5
            image = torch.clamp(image, 0, 1)
        
        test_images.append(image.unsqueeze(0))  # Add batch dimension
    
    return test_images


def demonstrate_fusion_methods():
    """Demonstrate different fusion methods."""
    logger.info("=== Demonstrating Fusion Methods ===")
    
    # Initialize estimators (using mock for Depth Pro)
    marigold_config = OmegaConf.create({
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'num_inference_steps': 4,
        'ensemble_size': 3,
        'precision': 'fp16'
    })
    
    marigold_estimator = MarigoldDepthEstimator(marigold_config)
    depth_pro_estimator = MockDepthProEstimator()
    
    # Test different fusion methods
    fusion_methods = [
        FusionMethod.WEIGHTED_AVERAGE,
        FusionMethod.CONFIDENCE_BASED,
        FusionMethod.ADAPTIVE_SELECTION,
        FusionMethod.QUALITY_AWARE
    ]
    
    test_image = create_test_images(1)[0]
    results = {}
    
    for fusion_method in fusion_methods:
        logger.info(f"Testing fusion method: {fusion_method.value}")
        
        # Configure ensemble
        ensemble_config = EnsembleConfig(
            fusion_method=fusion_method,
            parallel_inference=True,
            quality_assessment=True
        )
        
        ensemble = EnsembleDepthEstimator(
            marigold_estimator,
            depth_pro_estimator,
            ensemble_config
        )
        
        # Run estimation
        start_time = time.time()
        depth_result = ensemble.estimate_depth_ensemble(test_image)
        inference_time = time.time() - start_time
        
        # Get metrics
        metrics = ensemble.get_metrics()
        
        results[fusion_method.value] = {
            'depth': depth_result,
            'inference_time': inference_time,
            'metrics': metrics
        }
        
        logger.info(f"  - Inference time: {inference_time:.3f}s")
        logger.info(f"  - Fusion weights: {metrics.fusion_weights}")
        logger.info(f"  - Total time: {metrics.total_time:.3f}s")
        
        ensemble.cleanup()
    
    return results


def demonstrate_quality_profiles():
    """Demonstrate different quality profiles."""
    logger.info("=== Demonstrating Quality Profiles ===")
    
    # Load configuration with quality profiles
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="ensemble_depth_estimation")
    
    profiles = ['fast', 'balanced', 'high_quality']
    test_image = create_test_images(1)[0]
    
    results = {}
    
    for profile in profiles:
        logger.info(f"Testing quality profile: {profile}")
        
        # Get profile configuration
        profile_cfg = cfg.quality_profiles[profile]
        
        # Initialize estimators with profile settings
        marigold_config = OmegaConf.create(profile_cfg.marigold)
        marigold_estimator = MarigoldDepthEstimator(marigold_config)
        depth_pro_estimator = MockDepthProEstimator(profile_cfg.depth_pro)
        
        # Configure ensemble
        ensemble_config = EnsembleConfig(**profile_cfg.ensemble)
        ensemble = EnsembleDepthEstimator(
            marigold_estimator,
            depth_pro_estimator,
            ensemble_config
        )
        
        # Benchmark
        start_time = time.time()
        depth_result = ensemble.estimate_depth_ensemble(test_image)
        inference_time = time.time() - start_time
        
        metrics = ensemble.get_metrics()
        
        results[profile] = {
            'depth': depth_result,
            'inference_time': inference_time,
            'metrics': metrics
        }
        
        logger.info(f"  - Inference time: {inference_time:.3f}s")
        logger.info(f"  - FPS equivalent: {1.0/inference_time:.1f}")
        
        ensemble.cleanup()
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing different configurations."""
    logger.info("=== Running Comprehensive Benchmark ===")
    
    # Initialize estimators
    marigold_config = OmegaConf.create({
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'num_inference_steps': 4,
        'ensemble_size': 3
    })
    
    marigold_estimator = MarigoldDepthEstimator(marigold_config)
    depth_pro_estimator = MockDepthProEstimator()
    
    # Benchmark configuration
    benchmark_config = BenchmarkConfig(
        mode=BenchmarkMode.COMPREHENSIVE,
        iterations=20,
        warmup_iterations=3,
        test_resolutions=[(240, 320), (480, 640)],
        batch_sizes=[1, 2, 4],
        output_dir="benchmark_results",
        generate_plots=True
    )
    
    benchmark = DepthEstimationBenchmark(benchmark_config)
    
    # Test different ensemble configurations
    test_configs = [
        {
            'name': 'fast_ensemble',
            'config': EnsembleConfig(
                fusion_method=FusionMethod.WEIGHTED_AVERAGE,
                parallel_inference=True,
                quality_assessment=False
            )
        },
        {
            'name': 'quality_ensemble',
            'config': EnsembleConfig(
                fusion_method=FusionMethod.CONFIDENCE_BASED,
                parallel_inference=True,
                quality_assessment=True,
                scene_analysis=True
            )
        },
        {
            'name': 'adaptive_ensemble',
            'config': EnsembleConfig(
                fusion_method=FusionMethod.ADAPTIVE_SELECTION,
                parallel_inference=True,
                scene_analysis=True
            )
        }
    ]
    
    benchmark_results = {}
    
    for test_config in test_configs:
        logger.info(f"Benchmarking: {test_config['name']}")
        
        ensemble = EnsembleDepthEstimator(
            marigold_estimator,
            depth_pro_estimator,
            test_config['config']
        )
        
        result = benchmark.benchmark_model(ensemble, test_config['name'])
        benchmark_results[test_config['name']] = result
        
        logger.info(f"  - Average FPS: {result.fps:.2f}")
        logger.info(f"  - Peak GPU Memory: {result.peak_gpu_memory:.2f} GB")
        logger.info(f"  - CPU Utilization: {result.cpu_utilization:.1f}%")
        
        ensemble.cleanup()
    
    # Generate comprehensive report
    report_path = benchmark.generate_report("json")
    benchmark.generate_plots()
    
    logger.info(f"Benchmark report saved to: {report_path}")
    
    return benchmark_results


def demonstrate_video_processing():
    """Demonstrate video sequence processing with temporal consistency."""
    logger.info("=== Demonstrating Video Processing ===")
    
    # Initialize estimators
    marigold_config = OmegaConf.create({
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'num_inference_steps': 4,
        'ensemble_size': 3,
        'enable_temporal_consistency': True
    })
    
    marigold_estimator = MarigoldDepthEstimator(marigold_config)
    depth_pro_estimator = MockDepthProEstimator()
    
    # Configure ensemble for video
    ensemble_config = EnsembleConfig(
        fusion_method=FusionMethod.CONFIDENCE_BASED,
        parallel_inference=True,
        quality_assessment=True
    )
    
    ensemble = EnsembleDepthEstimator(
        marigold_estimator,
        depth_pro_estimator,
        ensemble_config
    )
    
    # Create video sequence (simulate moving camera)
    sequence_length = 10
    video_frames = []
    
    for frame_idx in range(sequence_length):
        # Create frame with slight camera movement
        offset_x = int(frame_idx * 5)  # Horizontal pan
        base_image = torch.zeros(3, 480, 640)
        
        # Add moving objects
        for c in range(3):
            y_coords, x_coords = torch.meshgrid(
                torch.arange(480), torch.arange(640), indexing='ij'
            )
            pattern = torch.sin((x_coords + offset_x) * 0.02) * torch.sin(y_coords * 0.02)
            base_image[c] = (pattern + 1) / 2  # Normalize to [0, 1]
        
        video_frames.append(base_image.unsqueeze(0))
    
    # Process video sequence
    depth_sequence = []
    processing_times = []
    
    logger.info(f"Processing {sequence_length} frames...")
    
    for i, frame in enumerate(video_frames):
        start_time = time.time()
        depth = ensemble.estimate_depth_ensemble(frame)
        processing_time = time.time() - start_time
        
        depth_sequence.append(depth)
        processing_times.append(processing_time)
        
        logger.info(f"Frame {i+1}/{sequence_length}: {processing_time:.3f}s")
    
    # Analyze temporal consistency
    temporal_diffs = []
    for i in range(1, len(depth_sequence)):
        diff = torch.mean(torch.abs(depth_sequence[i] - depth_sequence[i-1]))
        temporal_diffs.append(diff.item())
    
    avg_temporal_diff = np.mean(temporal_diffs)
    avg_processing_time = np.mean(processing_times)
    
    logger.info(f"Average processing time: {avg_processing_time:.3f}s")
    logger.info(f"Effective FPS: {1.0/avg_processing_time:.1f}")
    logger.info(f"Average temporal difference: {avg_temporal_diff:.4f}")
    
    ensemble.cleanup()
    
    return {
        'depth_sequence': depth_sequence,
        'processing_times': processing_times,
        'temporal_consistency': avg_temporal_diff
    }


def demonstrate_multi_resolution_processing():
    """Demonstrate processing at different resolutions."""
    logger.info("=== Demonstrating Multi-Resolution Processing ===")
    
    # Initialize estimators
    marigold_estimator = MarigoldDepthEstimator(OmegaConf.create({
        'model_name': 'prs-eth/marigold-lcm-v1-0',
        'num_inference_steps': 4,
        'ensemble_size': 3
    }))
    depth_pro_estimator = MockDepthProEstimator()
    
    ensemble_config = EnsembleConfig(
        fusion_method=FusionMethod.CONFIDENCE_BASED,
        parallel_inference=True
    )
    
    ensemble = EnsembleDepthEstimator(
        marigold_estimator,
        depth_pro_estimator,
        ensemble_config
    )
    
    # Test different resolutions
    resolutions = [
        (240, 320),   # QVGA
        (480, 640),   # VGA
        (720, 1280),  # HD
        (1080, 1920)  # Full HD
    ]
    
    results = {}
    
    for height, width in resolutions:
        logger.info(f"Testing resolution: {height}x{width}")
        
        # Create test image at specific resolution
        test_image = torch.rand(1, 3, height, width)
        
        # Measure processing time and memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        depth = ensemble.estimate_depth_ensemble(test_image)
        processing_time = time.time() - start_time
        
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        results[f"{height}x{width}"] = {
            'processing_time': processing_time,
            'peak_memory': peak_memory,
            'fps': 1.0 / processing_time,
            'depth_shape': depth.shape
        }
        
        logger.info(f"  - Processing time: {processing_time:.3f}s")
        logger.info(f"  - FPS: {1.0/processing_time:.2f}")
        logger.info(f"  - Peak memory: {peak_memory:.2f} GB")
    
    ensemble.cleanup()
    
    return results


def create_visualization_report(results: Dict[str, Any], output_dir: str = "outputs"):
    """Create visualization report of all results."""
    logger.info("=== Creating Visualization Report ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble Depth Estimation Results Summary', fontsize=16)
    
    # Plot 1: Fusion method comparison
    if 'fusion_methods' in results:
        fusion_data = results['fusion_methods']
        methods = list(fusion_data.keys())
        times = [fusion_data[method]['inference_time'] for method in methods]
        
        axes[0, 0].bar(methods, times)
        axes[0, 0].set_title('Fusion Method Performance')
        axes[0, 0].set_ylabel('Inference Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Quality profiles comparison
    if 'quality_profiles' in results:
        profile_data = results['quality_profiles']
        profiles = list(profile_data.keys())
        fps_values = [1.0/profile_data[profile]['inference_time'] for profile in profiles]
        
        axes[0, 1].bar(profiles, fps_values)
        axes[0, 1].set_title('Quality Profile FPS')
        axes[0, 1].set_ylabel('FPS')
    
    # Plot 3: Resolution scaling
    if 'multi_resolution' in results:
        res_data = results['multi_resolution']
        resolutions = list(res_data.keys())
        times = [res_data[res]['processing_time'] for res in resolutions]
        
        axes[1, 0].plot(range(len(resolutions)), times, 'o-')
        axes[1, 0].set_title('Resolution vs Processing Time')
        axes[1, 0].set_ylabel('Processing Time (s)')
        axes[1, 0].set_xticks(range(len(resolutions)))
        axes[1, 0].set_xticklabels(resolutions, rotation=45)
    
    # Plot 4: Video processing temporal consistency
    if 'video_processing' in results:
        video_data = results['video_processing']
        frame_times = video_data['processing_times']
        
        axes[1, 1].plot(frame_times, 'o-')
        axes[1, 1].set_title('Video Frame Processing Times')
        axes[1, 1].set_ylabel('Processing Time (s)')
        axes[1, 1].set_xlabel('Frame Number')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ensemble_results_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization report saved to: {output_path}")


def main():
    """Main demonstration function."""
    logger.info("Starting Ensemble Depth Estimation Demonstration")
    
    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    Path("benchmark_results").mkdir(exist_ok=True)
    
    # Collect all results
    all_results = {}
    
    try:
        # 1. Demonstrate fusion methods
        fusion_results = demonstrate_fusion_methods()
        all_results['fusion_methods'] = fusion_results
        
        # 2. Demonstrate quality profiles
        profile_results = demonstrate_quality_profiles()
        all_results['quality_profiles'] = profile_results
        
        # 3. Run comprehensive benchmark
        benchmark_results = run_comprehensive_benchmark()
        all_results['benchmark'] = benchmark_results
        
        # 4. Demonstrate video processing
        video_results = demonstrate_video_processing()
        all_results['video_processing'] = video_results
        
        # 5. Demonstrate multi-resolution processing
        resolution_results = demonstrate_multi_resolution_processing()
        all_results['multi_resolution'] = resolution_results
        
        # 6. Create visualization report
        create_visualization_report(all_results)
        
        logger.info("=== Demonstration Complete ===")
        logger.info("Summary of results:")
        
        # Print summary statistics
        if 'fusion_methods' in all_results:
            fastest_fusion = min(all_results['fusion_methods'].items(),
                               key=lambda x: x[1]['inference_time'])
            logger.info(f"Fastest fusion method: {fastest_fusion[0]} ({fastest_fusion[1]['inference_time']:.3f}s)")
        
        if 'quality_profiles' in all_results:
            fastest_profile = min(all_results['quality_profiles'].items(),
                                key=lambda x: x[1]['inference_time'])
            logger.info(f"Fastest quality profile: {fastest_profile[0]} ({fastest_profile[1]['inference_time']:.3f}s)")
        
        if 'video_processing' in all_results:
            avg_fps = 1.0 / np.mean(all_results['video_processing']['processing_times'])
            logger.info(f"Video processing average FPS: {avg_fps:.2f}")
        
        logger.info("Check 'outputs/' and 'benchmark_results/' directories for detailed results")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 