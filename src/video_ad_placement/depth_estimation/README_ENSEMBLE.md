# Ensemble Depth Estimation System

## Overview

This ensemble depth estimation system combines Apple's Depth Pro model with the Marigold diffusion-based depth estimator to provide state-of-the-art depth estimation with robust performance across diverse scenes and conditions.

## Key Features

### 🔥 **Advanced Ensemble Fusion**
- **Multiple Fusion Methods**: Weighted average, confidence-based, learned fusion, adaptive selection, and quality-aware fusion
- **Learned Fusion Network**: Neural network-based fusion with pixel-wise and global weight prediction
- **Scene Complexity Analysis**: Automatic model selection based on scene characteristics
- **Quality Assessment**: Comprehensive quality metrics for prediction validation

### ⚡ **High-Performance Optimization** 
- **TensorRT Integration**: Real-time inference optimization for production deployment
- **Multi-GPU Support**: Parallel processing across multiple GPUs with data/model parallelism
- **Memory Efficiency**: Advanced memory management with gradient checkpointing and cache optimization
- **Parallel Inference**: Concurrent execution of both models for maximum throughput

### 📊 **Comprehensive Benchmarking**
- **Multi-dimensional Testing**: Performance, accuracy, memory, latency, and throughput benchmarks
- **Hardware Monitoring**: Real-time GPU/CPU utilization and memory usage tracking
- **Ground Truth Validation**: Standard depth estimation metrics (MAE, RMSE, δ-accuracy, SILog)
- **Automated Reporting**: Detailed reports with visualizations and performance analysis

### 🎯 **Production-Ready Features**
- **Metric Scale Recovery**: Geometric and learned approaches for absolute depth estimation
- **Temporal Consistency**: Advanced temporal filtering for video sequences
- **Error Handling**: Robust fallback mechanisms and graceful degradation
- **Configuration Management**: Flexible YAML-based configuration with quality profiles

## Quick Start

### Installation

```bash
# Install core dependencies
pip install torch torchvision transformers diffusers

# Install ensemble-specific dependencies
pip install psutil seaborn plotly line_profiler
pip install xformers accelerate  # Optional: for optimization

# Install TensorRT (optional, for production)
# Follow NVIDIA TensorRT installation guide
```

### Basic Usage

```python
from src.video_ad_placement.depth_estimation import (
    MarigoldDepthEstimator,
    EnsembleDepthEstimator,
    EnsembleConfig,
    FusionMethod
)
from src.video_ad_placement.depth_estimation.depth_pro_estimator import DepthProEstimator
from omegaconf import OmegaConf

# Initialize individual estimators
marigold_config = OmegaConf.create({
    'model_name': 'prs-eth/marigold-lcm-v1-0',
    'num_inference_steps': 4,
    'ensemble_size': 3,
    'precision': 'fp16'
})

marigold_estimator = MarigoldDepthEstimator(marigold_config)
depth_pro_estimator = DepthProEstimator({})

# Configure ensemble
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.CONFIDENCE_BASED,
    parallel_inference=True,
    quality_assessment=True,
    scene_analysis=True
)

# Create ensemble
ensemble = EnsembleDepthEstimator(
    marigold_estimator,
    depth_pro_estimator,
    ensemble_config
)

# Estimate depth
import torch
input_image = torch.randn(1, 3, 480, 640)  # Replace with real image
depth_map = ensemble.estimate_depth_ensemble(input_image)

print(f"Output shape: {depth_map.shape}")
print(f"Depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")

# Get performance metrics
metrics = ensemble.get_metrics()
print(f"Inference time: {metrics.total_time:.3f}s")
print(f"Fusion weights: {metrics.fusion_weights}")
```

### Configuration Profiles

The system includes predefined quality profiles for different use cases:

```python
# Load configuration with profiles
from hydra import compose, initialize

with initialize(config_path="configs", version_base=None):
    cfg = compose(config_name="ensemble_depth_estimation")

# Available profiles: fast, balanced, high_quality, ultra
profile = cfg.quality_profiles.balanced
```

#### Profile Comparison

| Profile | Speed | Quality | Memory | Use Case |
|---------|-------|---------|---------|----------|
| **Fast** | ⚡⚡⚡⚡ | ⭐⭐ | 🟢 Low | Real-time applications |
| **Balanced** | ⚡⚡⚡ | ⭐⭐⭐ | 🟡 Medium | General purpose |
| **High Quality** | ⚡⚡ | ⭐⭐⭐⭐ | 🟠 High | Content creation |
| **Ultra** | ⚡ | ⭐⭐⭐⭐⭐ | 🔴 Very High | Research & analysis |

## Fusion Methods

### 1. Weighted Average
Simple linear combination of predictions with fixed weights.

```python
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.WEIGHTED_AVERAGE,
    marigold_weight=0.6,
    depth_pro_weight=0.4
)
```

### 2. Confidence-Based Fusion
Adaptive weighting based on model confidence scores.

```python
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.CONFIDENCE_BASED,
    confidence_threshold=0.7
)
```

### 3. Learned Fusion
Neural network-based fusion with trainable weights.

```python
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.LEARNED_FUSION,
    learned_fusion=True
)
```

### 4. Adaptive Selection
Scene complexity-based model selection.

```python
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.ADAPTIVE_SELECTION,
    scene_analysis=True,
    complexity_threshold=0.5
)
```

### 5. Quality-Aware Fusion
Quality assessment-based adaptive weighting.

```python
ensemble_config = EnsembleConfig(
    fusion_method=FusionMethod.QUALITY_AWARE,
    quality_assessment=True,
    outlier_rejection=True
)
```

## Benchmarking

### Quick Benchmark

```python
from src.video_ad_placement.depth_estimation.benchmarking import (
    DepthEstimationBenchmark,
    BenchmarkConfig,
    BenchmarkMode
)

# Configure benchmark
benchmark_config = BenchmarkConfig(
    mode=BenchmarkMode.COMPREHENSIVE,
    iterations=50,
    test_resolutions=[(480, 640), (720, 1280)],
    batch_sizes=[1, 2, 4, 8],
    output_dir="benchmark_results"
)

# Run benchmark
benchmark = DepthEstimationBenchmark(benchmark_config)
results = benchmark.benchmark_model(ensemble, "ensemble_test")

print(f"Average FPS: {results.fps:.2f}")
print(f"Peak GPU Memory: {results.peak_gpu_memory:.2f} GB")
```

### Comprehensive Benchmark

```python
# Benchmark multiple ensemble configurations
test_scenarios = [
    {'fusion_method': 'weighted_average', 'parallel_inference': False},
    {'fusion_method': 'confidence_based', 'parallel_inference': True},
    {'fusion_method': 'adaptive_selection', 'parallel_inference': True},
    {'fusion_method': 'learned_fusion', 'parallel_inference': True}
]

ensemble_results = benchmark.benchmark_ensemble(ensemble, test_scenarios)

# Generate comprehensive report
report_path = benchmark.generate_report("json")
benchmark.generate_plots()
```

## Advanced Features

### Temporal Consistency for Video

```python
# Enable temporal consistency
marigold_config = OmegaConf.create({
    'enable_temporal_consistency': True,
    'temporal_filter_type': 'exponential',
    'temporal_alpha': 0.8
})

# Process video sequence
video_frames = [...]  # List of video frames
depth_sequence = []

for frame in video_frames:
    depth = ensemble.estimate_depth_ensemble(frame)
    depth_sequence.append(depth)
```

### Multi-GPU Processing

```python
# Configure multi-GPU ensemble
ensemble_config = EnsembleConfig(
    multi_gpu=True,
    gpu_devices=[0, 1, 2, 3],
    data_parallel=True,
    parallel_inference=True
)
```

### Custom Quality Assessment

```python
from src.video_ad_placement.depth_estimation.ensemble_estimator import QualityAssessmentModule

quality_assessor = QualityAssessmentModule(ensemble_config)

# Assess prediction quality
quality_metrics = quality_assessor.assess_quality(
    depth_map=depth_prediction,
    confidence_map=confidence_scores,
    original_image=input_image
)

print(f"Overall quality: {quality_metrics['overall_quality']:.3f}")
print(f"Edge preservation: {quality_metrics['edge_preservation']:.3f}")
```

## Performance Optimization

### TensorRT Optimization

```python
# Enable TensorRT for Depth Pro
depth_pro_config = {
    'use_tensorrt': True,
    'tensorrt_workspace': 4096,  # MB
    'precision': 'fp16',
    'optimize_for_latency': True
}

depth_pro_estimator = DepthProEstimator(depth_pro_config)
```

### Memory Optimization

```python
# Memory-efficient configuration
ensemble_config = EnsembleConfig(
    memory_efficient=True,
    cache_predictions=False,
    outlier_rejection=True,  # Reduces memory spikes
    consistency_check=True   # Post-processing optimization
)
```

### Batch Processing

```python
# Optimize batch size for your hardware
batch_sizes = [1, 2, 4, 8, 16]
optimal_batch_size = 4  # Determined through benchmarking

# Process large datasets
for i in range(0, len(dataset), optimal_batch_size):
    batch = dataset[i:i+optimal_batch_size]
    depths = ensemble.estimate_depth_ensemble(batch)
```

## Integration Examples

### Video Advertisement Placement

```python
# Integration with main video processing pipeline
from src.video_ad_placement.core.scene_analyzer import SceneAnalyzer
from src.video_ad_placement.placement.ad_placement_engine import AdPlacementEngine

# Initialize ensemble for scene analysis
ensemble = EnsembleDepthEstimator(marigold_estimator, depth_pro_estimator, config)

# Scene analysis with depth
scene_analyzer = SceneAnalyzer(depth_estimator=ensemble)
placement_engine = AdPlacementEngine()

# Process video for ad placement
video_frames = load_video("input.mp4")
placements = []

for frame in video_frames:
    # Get depth and scene understanding
    depth = ensemble.estimate_depth_ensemble(frame)
    scene_info = scene_analyzer.analyze_scene(frame, depth)
    
    # Find optimal ad placement
    placement = placement_engine.find_placement(frame, depth, scene_info)
    placements.append(placement)
```

### Real-time Streaming

```python
import asyncio
from src.video_ad_placement.streaming import StreamProcessor

async def process_stream():
    # Configure for low-latency streaming
    fast_config = EnsembleConfig(
        fusion_method=FusionMethod.WEIGHTED_AVERAGE,
        parallel_inference=True,
        quality_assessment=False,  # Disable for speed
        scene_analysis=False
    )
    
    ensemble = EnsembleDepthEstimator(marigold_estimator, depth_pro_estimator, fast_config)
    
    # Process real-time stream
    async for frame in stream_source:
        depth = ensemble.estimate_depth_ensemble(frame)
        yield process_frame_with_depth(frame, depth)

# Run streaming pipeline
asyncio.run(process_stream())
```

## Configuration Reference

### Complete Configuration Example

```yaml
# ensemble_depth_estimation.yaml
marigold:
  model_name: "prs-eth/marigold-lcm-v1-0"
  precision: "fp16"
  batch_size: 4
  num_inference_steps: 5
  ensemble_size: 5
  processing_resolution: 768
  enable_temporal_consistency: true
  temporal_filter_type: "exponential"
  temporal_alpha: 0.8

depth_pro:
  model_name: "apple/DepthPro"
  precision: "fp16"
  batch_size: 8
  use_tensorrt: true
  tensorrt_workspace: 4096
  enable_xformers: true
  metric_scale: "auto"

ensemble:
  fusion_method: "confidence_based"
  learned_fusion: true
  adaptive_weights: true
  marigold_weight: 0.6
  depth_pro_weight: 0.4
  confidence_threshold: 0.7
  scene_analysis: true
  quality_assessment: true
  parallel_inference: true
  memory_efficient: true

hardware:
  device: "cuda"
  mixed_precision: true
  compile_model: false
  multi_gpu: false
  memory_fraction: 0.9

tensorrt:
  enabled: true
  fp16_mode: true
  workspace_size: 4096
  max_batch_size: 16
  optimize_for_latency: true

quality_profiles:
  fast:
    marigold:
      num_inference_steps: 1
      ensemble_size: 1
      processing_resolution: 384
    ensemble:
      fusion_method: "weighted_average"
      quality_assessment: false
  
  balanced:
    marigold:
      num_inference_steps: 4
      ensemble_size: 3
      processing_resolution: 512
    ensemble:
      fusion_method: "confidence_based"
      quality_assessment: true
```

## Troubleshooting

### Common Issues

#### Memory Issues
```python
# Reduce memory usage
ensemble_config = EnsembleConfig(
    memory_efficient=True,
    cache_predictions=False
)

# Monitor memory usage
metrics = ensemble.get_metrics()
print(f"Memory usage: {metrics.memory_usage:.2f} GB")
```

#### Performance Issues
```python
# Profile performance
from src.video_ad_placement.depth_estimation.benchmarking import HardwareMonitor

monitor = HardwareMonitor(config)
monitor.start_monitoring()

# Your inference code here
depth = ensemble.estimate_depth_ensemble(frame)

monitor.stop_monitoring()
summary = monitor.get_summary()
print(f"GPU utilization: {summary['gpu_utilization_avg']:.1f}%")
```

#### Model Loading Issues
```python
# Fallback configuration
try:
    ensemble = EnsembleDepthEstimator(marigold_estimator, depth_pro_estimator, config)
except Exception as e:
    logger.warning(f"Failed to load ensemble: {e}")
    # Use single model fallback
    single_estimator = marigold_estimator
```

### Performance Tips

1. **Use appropriate batch sizes**: Start with batch_size=4 and adjust based on GPU memory
2. **Enable TensorRT**: Significant speedup for Depth Pro model in production
3. **Choose fusion method wisely**: Weighted average is fastest, learned fusion is most accurate
4. **Monitor hardware utilization**: Use benchmarking tools to identify bottlenecks
5. **Profile regularly**: Performance characteristics may change with different scene types

## API Reference

### EnsembleDepthEstimator

Main class for ensemble depth estimation.

#### Methods

- `estimate_depth_ensemble(frames)`: Estimate depth using ensemble
- `get_metrics()`: Get performance metrics
- `benchmark(test_data, iterations)`: Run performance benchmark
- `cleanup()`: Clean up resources

#### Configuration

- `fusion_method`: Fusion strategy (weighted_average, confidence_based, etc.)
- `parallel_inference`: Enable parallel model execution
- `quality_assessment`: Enable quality validation
- `scene_analysis`: Enable scene complexity analysis

### BenchmarkConfig

Configuration for comprehensive benchmarking.

#### Parameters

- `mode`: Benchmark mode (performance, accuracy, memory, comprehensive)
- `iterations`: Number of benchmark iterations
- `test_resolutions`: List of resolutions to test
- `batch_sizes`: List of batch sizes to test
- `output_dir`: Directory for benchmark results

## Contributing

### Adding New Fusion Methods

1. Add new method to `FusionMethod` enum
2. Implement fusion logic in `EnsembleDepthEstimator._fuse_predictions()`
3. Add configuration options to `EnsembleConfig`
4. Update tests and documentation

### Adding New Benchmarks

1. Extend `BenchmarkMode` enum
2. Implement benchmark logic in `DepthEstimationBenchmark`
3. Add visualization support
4. Update configuration schema

## License

This ensemble depth estimation system is part of the AI-Powered Video Advertisement Placement project. Please refer to the main project license for usage terms.

## Citation

If you use this ensemble system in your research, please cite:

```bibtex
@software{ensemble_depth_estimation,
  title={Ensemble Depth Estimation with Marigold and Depth Pro},
  author={AI Video Advertisement Team},
  year={2024},
  url={https://github.com/your-repo/multimercial}
}
```

## Acknowledgments

- Apple for the Depth Pro model
- Marigold team for the diffusion-based depth estimation
- NVIDIA for TensorRT optimization tools
- Hugging Face for model hosting and transformers library 