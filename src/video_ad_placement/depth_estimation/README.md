# Advanced Marigold Depth Estimation Module

A comprehensive depth estimation system using the Marigold diffusion model with advanced temporal consistency, memory optimization, and production-ready features.

## Overview

This module provides state-of-the-art depth estimation capabilities using the Marigold diffusion model, enhanced with:

- **Temporal Consistency**: Advanced optical flow-based temporal smoothing
- **Memory Optimization**: Adaptive batch sizing and GPU memory management
- **Quality Modes**: Configurable quality/speed trade-offs
- **Comprehensive Metrics**: Performance monitoring and quality assessment
- **Production Ready**: Error handling, checkpointing, and scalability

## Features

### Core Capabilities

- **Marigold Integration**: Direct integration with Hugging Face Marigold models
- **Batch Processing**: Efficient processing of multiple frames
- **Multiple Precision**: FP32, FP16, and mixed precision support
- **Adaptive Resolution**: Automatic resolution scaling based on hardware

### Temporal Consistency

- **Optical Flow Warping**: Dense optical flow for depth map warping
- **Multiple Filters**: Exponential, bilateral, and Kalman temporal filters
- **Occlusion Detection**: Automatic handling of occluded regions
- **Motion Compensation**: Advanced motion-aware temporal smoothing

### Memory Management

- **GPU Memory Monitoring**: Real-time memory usage tracking
- **Adaptive Batch Sizing**: Automatic batch size adjustment
- **Memory Cleanup**: Automatic garbage collection and cache clearing
- **Memory Context**: Safe memory management with context managers

### Quality Assessment

- **Depth Metrics**: Edge density, gradient magnitude, temporal consistency
- **Quality Scoring**: Automatic quality assessment and reporting
- **Benchmark Mode**: Detailed performance profiling
- **Validation Support**: Comparison with ground truth data

## Installation

### Dependencies

```bash
# Core dependencies (required)
pip install torch torchvision numpy opencv-python Pillow

# Marigold dependencies (required for inference)
pip install diffusers transformers accelerate safetensors

# Optional optimization dependencies
pip install xformers  # For memory efficiency

# Video processing dependencies
pip install moviepy imageio av ffmpeg-python

# Visualization dependencies
pip install matplotlib plotly

# Development dependencies
pip install pytest pytest-cov black isort
```

### Installation from source

```bash
git clone <repository-url>
cd multimercial
pip install -e .
```

## Quick Start

### Basic Usage

```python
from video_ad_placement.depth_estimation import MarigoldDepthEstimator
import numpy as np

# Initialize estimator
config = {
    'quality_mode': 'balanced',
    'device': 'cuda',  # or 'cpu'
    'enable_temporal_consistency': True
}

estimator = MarigoldDepthEstimator(config)

# Estimate depth for single image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
depth_map = estimator.estimate_depth(image)

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
```

### Video Processing

```python
from video_ad_placement.depth_estimation.utils import VideoDepthProcessor, VideoProcessingConfig

# Configure video processing
video_config = VideoProcessingConfig(
    visualization_mode="side_by_side",
    save_individual_frames=True,
    output_fps=30
)

processor = VideoDepthProcessor(video_config)

# Process video with depth estimation
stats = processor.process_video_with_depth(
    video_path="input_video.mp4",
    depth_estimator=estimator,
    output_path="output_with_depth.mp4"
)

print(f"Processed {stats['total_frames']} frames in {stats['processing_time']:.2f}s")
```

### Temporal Consistency

```python
from video_ad_placement.depth_estimation.temporal_consistency import (
    TemporalConsistencyProcessor,
    TemporalFilterConfig
)

# Configure temporal processing
temporal_config = TemporalFilterConfig(
    window_size=5,
    alpha=0.3,
    motion_threshold=0.1
)

processor = TemporalConsistencyProcessor(temporal_config)

# Process sequence with temporal consistency
for frame_idx, (depth, frame) in enumerate(zip(depth_sequence, frame_sequence)):
    consistent_depth = processor.process_frame(depth, frame, frame_idx)
```

## Configuration

### Quality Modes

| Mode | Speed | Quality | Ensemble Size | Inference Steps | Use Case |
|------|-------|---------|---------------|-----------------|----------|
| fast | ⭐⭐⭐⭐⭐ | ⭐⭐ | 3 | 4 | Real-time processing |
| balanced | ⭐⭐⭐ | ⭐⭐⭐⭐ | 5 | 10 | General use |
| high_quality | ⭐⭐ | ⭐⭐⭐⭐⭐ | 8 | 15 | High-quality output |
| ultra | ⭐ | ⭐⭐⭐⭐⭐ | 10 | 20 | Maximum quality |

### Configuration File

Create a configuration file `configs/depth_estimation.yaml`:

```yaml
model:
  name: "prs-eth/marigold-lcm-v1-0"
  device: "auto"

quality:
  mode: "balanced"
  precision: "fp16"

temporal:
  enabled: true
  method: "optical_flow"
  alpha: 0.3

memory:
  max_gpu_memory_gb: 8.0
  adaptive_batch_size: true
```

Load configuration with Hydra:

```python
from hydra import compose, initialize
from omegaconf import DictConfig

with initialize(config_path="configs"):
    cfg = compose(config_name="depth_estimation")
    estimator = MarigoldDepthEstimator(cfg)
```

## Advanced Features

### Custom Temporal Filters

```python
from video_ad_placement.depth_estimation.temporal_consistency import TemporalFilter

class CustomTemporalFilter(TemporalFilter):
    def filter(self, current_depth, previous_depths, **kwargs):
        # Custom filtering logic
        return filtered_depth
    
    def reset(self):
        # Reset filter state
        pass

# Use custom filter
processor.temporal_filter = CustomTemporalFilter()
```

### GPU Memory Management

```python
from video_ad_placement.depth_estimation import GPUMemoryManager

memory_manager = GPUMemoryManager(memory_limit_gb=6.0)

# Monitor memory usage
usage = memory_manager.get_memory_usage()
print(f"GPU Memory: {usage['allocated']:.2f}GB / {usage['total']:.2f}GB")

# Automatic memory management
with memory_manager.memory_context():
    depth_map = estimator.estimate_depth(large_batch)
```

### Quality Metrics

```python
from video_ad_placement.depth_estimation import compute_depth_quality_metrics

# Compute quality metrics
metrics = compute_depth_quality_metrics(depth_map, reference_depth)

print(f"Edge density: {metrics['edge_density']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Correlation: {metrics['correlation']:.3f}")
```

### Visualization

```python
from video_ad_placement.depth_estimation.utils import DepthVisualization

visualizer = DepthVisualization(colormap="plasma")

# Create visualizations
depth_colored = visualizer.depth_to_colormap(depth_map)
side_by_side = visualizer.create_side_by_side(image, depth_map)
overlay = visualizer.create_overlay(image, depth_map, alpha=0.4)

# Save high-quality visualization
visualizer.save_depth_visualization(
    depth_map, 
    "depth_visualization.png",
    title="Estimated Depth Map"
)

# Create interactive 3D visualization
fig = visualizer.create_3d_visualization(depth_map)
fig.show()
```

## Performance Optimization

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU Memory | 4GB | 8GB | 16GB+ |
| System RAM | 8GB | 16GB | 32GB+ |
| CUDA Compute | 6.0+ | 7.5+ | 8.0+ |

### Optimization Tips

1. **Use FP16 precision** for 2x memory savings with minimal quality loss
2. **Enable xformers** for memory-efficient attention computation
3. **Adaptive batch sizing** automatically optimizes memory usage
4. **Attention slicing** reduces memory requirements for large images
5. **CPU offloading** for systems with limited GPU memory

### Benchmarking

```python
# Enable benchmark mode
config = {
    'benchmark_mode': True,
    'log_gpu_memory': True
}

estimator = MarigoldDepthEstimator(config)

# Process test batch
test_images = [create_test_image() for _ in range(10)]
depths = [estimator.estimate_depth(img) for img in test_images]

# Get performance metrics
metrics = estimator.get_metrics()
print(f"Average FPS: {metrics.average_fps:.2f}")
print(f"Peak GPU Memory: {metrics.gpu_memory_peak:.2f}GB")
```

## Error Handling

### Common Issues

1. **CUDA Out of Memory**
   - Enable adaptive batch sizing
   - Reduce ensemble size or inference steps
   - Use FP16 precision
   - Enable attention slicing

2. **Model Loading Errors**
   - Check internet connection
   - Verify model name
   - Clear Hugging Face cache

3. **Temporal Consistency Issues**
   - Adjust temporal alpha parameter
   - Check optical flow computation
   - Reduce temporal window size

### Graceful Degradation

```python
config = {
    'error_handling': {
        'max_retries': 3,
        'fallback_quality': 'fast',
        'continue_on_error': True,
        'reduce_batch_on_oom': True
    }
}
```

## API Reference

### MarigoldDepthEstimator

Main class for depth estimation with temporal consistency.

#### Methods

- `estimate_depth(frames)` - Estimate depth for single or batch of frames
- `estimate_depth_sequence(video_path)` - Process video sequence with temporal consistency
- `get_metrics()` - Get processing performance metrics
- `reset_state()` - Reset temporal consistency state
- `save_checkpoint(path)` - Save processing checkpoint
- `load_checkpoint(path)` - Load processing checkpoint

#### Configuration

- `quality_mode` - Quality/speed trade-off (fast, balanced, high_quality, ultra)
- `precision_mode` - Numerical precision (fp32, fp16, mixed)
- `enable_temporal_consistency` - Enable temporal smoothing
- `temporal_smoothing_method` - Temporal consistency method
- `adaptive_batch_size` - Enable adaptive batch sizing

### TemporalConsistencyProcessor

Advanced temporal consistency processing with multiple filter options.

#### Methods

- `process_frame(depth, frame, frame_idx)` - Process single frame with temporal consistency
- `compute_temporal_consistency_score()` - Compute temporal consistency metric
- `reset()` - Reset processor state
- `get_processing_stats()` - Get processing statistics

### DepthVisualization

Comprehensive depth map visualization utilities.

#### Methods

- `depth_to_colormap(depth_map)` - Convert depth to color visualization
- `create_side_by_side(image, depth)` - Create side-by-side comparison
- `create_overlay(image, depth)` - Create overlay visualization
- `save_depth_visualization(depth, path)` - Save high-quality visualization
- `create_3d_visualization(depth)` - Create interactive 3D visualization

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/test_depth_estimation.py -v

# Run specific test categories
pytest tests/test_depth_estimation.py::TestMarigoldDepthEstimator -v
pytest tests/test_depth_estimation.py::TestTemporalConsistency -v

# Run with coverage
pytest tests/test_depth_estimation.py --cov=src/video_ad_placement/depth_estimation
```

## Examples

See `examples/depth_estimation_example.py` for comprehensive usage examples:

```bash
# Run all examples
python examples/depth_estimation_example.py --example all

# Run specific example
python examples/depth_estimation_example.py --example single_frame
python examples/depth_estimation_example.py --example video
python examples/depth_estimation_example.py --example benchmark
```

## Integration

### With Advertisement Placement System

```python
from video_ad_placement.depth_estimation import MarigoldDepthEstimator
from video_ad_placement.core.scene_analyzer import SceneAnalyzer

# Initialize components
depth_estimator = MarigoldDepthEstimator(config)
scene_analyzer = SceneAnalyzer()

# Analyze scene with depth information
depth_map = depth_estimator.estimate_depth(frame)
scene_info = scene_analyzer.analyze_with_depth(frame, depth_map)

# Use depth for advertisement placement
placement_areas = scene_analyzer.find_placement_areas(
    frame, depth_map, min_depth=0.3, max_depth=0.8
)
```

### With 3D Reconstruction

```python
from video_ad_placement.depth_estimation import MarigoldDepthEstimator
from video_ad_placement.core.reconstruction_3d import PointCloudGenerator

depth_estimator = MarigoldDepthEstimator(config)
point_cloud_gen = PointCloudGenerator()

# Generate point cloud from depth
depth_map = depth_estimator.estimate_depth(frame)
point_cloud = point_cloud_gen.depth_to_point_cloud(depth_map, camera_params)
```

## Contributing

1. Follow the existing code style and structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Use type hints and docstrings
5. Run tests and linting before submitting

## License

This module is part of the AI-Powered Video Advertisement Placement System.
See the main project LICENSE file for details.

## Citation

If you use this depth estimation module in your research, please cite:

```bibtex
@software{marigold_depth_estimator,
  title={Advanced Marigold Depth Estimation Module},
  author={AI Video Advertisement Placement Team},
  year={2024},
  url={https://github.com/your-repo/multimercial}
}
```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review existing GitHub issues
- Create a new issue with detailed description and reproduction steps

---

**Note**: This module requires the Marigold model dependencies (`diffusers`, `transformers`) for full functionality. The module will gracefully handle missing dependencies with appropriate error messages. 