"""
Comprehensive Benchmarking and Profiling for Depth Estimation Ensemble

This module provides advanced benchmarking, profiling, and validation tools
for depth estimation systems with ground truth evaluation capabilities.
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import psutil

try:
    import nvgpu
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)


class BenchmarkMode(Enum):
    """Benchmark modes for different testing scenarios."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COMPREHENSIVE = "comprehensive"


class HardwareProfile(Enum):
    """Hardware profiles for testing."""
    EDGE_DEVICE = "edge_device"
    WORKSTATION = "workstation"
    SERVER = "server"
    CLOUD_GPU = "cloud_gpu"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    
    # Test configuration
    mode: BenchmarkMode = BenchmarkMode.COMPREHENSIVE
    iterations: int = 100
    warmup_iterations: int = 10
    
    # Data configuration
    test_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (240, 320), (480, 640), (720, 1280), (1080, 1920)
    ])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    test_data_path: Optional[str] = None
    ground_truth_path: Optional[str] = None
    
    # Hardware profiling
    hardware_profile: HardwareProfile = HardwareProfile.WORKSTATION
    monitor_gpu: bool = True
    monitor_cpu: bool = True
    monitor_memory: bool = True
    
    # Output configuration
    output_dir: str = "benchmark_results"
    save_detailed_results: bool = True
    generate_plots: bool = True
    save_predictions: bool = False
    
    # Validation configuration
    ground_truth_validation: bool = False
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'mae': 0.5,
        'rmse': 1.0,
        'relative_error': 0.2,
        'edge_preservation': 0.7
    })


@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""
    
    # Performance metrics
    avg_inference_time: float = 0.0
    std_inference_time: float = 0.0
    fps: float = 0.0
    throughput: float = 0.0
    
    # Memory metrics
    peak_gpu_memory: float = 0.0
    avg_gpu_memory: float = 0.0
    peak_cpu_memory: float = 0.0
    avg_cpu_memory: float = 0.0
    
    # Quality metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Hardware utilization
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Model-specific metrics
    model_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Configuration used
    test_config: Dict[str, Any] = field(default_factory=dict)


class HardwareMonitor:
    """Monitor hardware utilization during benchmarking."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitoring = False
        self.metrics = {
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_utilization': [],
            'ram_usage': []
        }
        
    def start_monitoring(self):
        """Start hardware monitoring."""
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        
    def stop_monitoring(self):
        """Stop hardware monitoring."""
        self.monitoring = False
        
    def collect_metrics(self):
        """Collect current hardware metrics."""
        if not self.monitoring:
            return
        
        # GPU metrics
        if torch.cuda.is_available() and self.config.monitor_gpu:
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.metrics['gpu_memory'].append(gpu_memory)
            
            if GPU_MONITORING_AVAILABLE:
                try:
                    gpu_info = nvgpu.gpu_info()
                    if gpu_info:
                        self.metrics['gpu_utilization'].append(gpu_info[0]['utilization'])
                except Exception:
                    pass
        
        # CPU metrics
        if self.config.monitor_cpu:
            cpu_percent = psutil.cpu_percent()
            self.metrics['cpu_utilization'].append(cpu_percent)
            
        # Memory metrics
        if self.config.monitor_memory:
            memory_info = psutil.virtual_memory()
            ram_usage = memory_info.used / 1e9  # GB
            self.metrics['ram_usage'].append(ram_usage)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of monitoring results."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[f'{metric_name}_avg'] = np.mean(values)
                summary[f'{metric_name}_peak'] = np.max(values)
                summary[f'{metric_name}_std'] = np.std(values)
        
        return summary


class AccuracyValidator:
    """Validate depth estimation accuracy against ground truth."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
    def compute_depth_metrics(self, predicted: torch.Tensor, 
                            ground_truth: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute accuracy metrics between predicted and ground truth depth."""
        try:
            # Ensure same shape
            if predicted.shape != ground_truth.shape:
                predicted = F.interpolate(
                    predicted, size=ground_truth.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Apply mask if provided
            if mask is not None:
                predicted = predicted[mask]
                ground_truth = ground_truth[mask]
            else:
                # Use valid depth mask
                valid_mask = (ground_truth > 0) & (ground_truth < 1000)
                predicted = predicted[valid_mask]
                ground_truth = ground_truth[valid_mask]
            
            if len(predicted) == 0:
                return {'error': 'No valid depth values'}
            
            # Compute standard depth metrics
            metrics = {}
            
            # Mean Absolute Error
            mae = torch.mean(torch.abs(predicted - ground_truth))
            metrics['mae'] = mae.item()
            
            # Root Mean Square Error
            mse = torch.mean((predicted - ground_truth) ** 2)
            rmse = torch.sqrt(mse)
            metrics['rmse'] = rmse.item()
            
            # Relative Error
            rel_error = torch.mean(torch.abs(predicted - ground_truth) / ground_truth)
            metrics['relative_error'] = rel_error.item()
            
            # Threshold Accuracies (δ < 1.25, 1.25², 1.25³)
            ratio = torch.max(predicted / ground_truth, ground_truth / predicted)
            metrics['delta_1'] = (ratio < 1.25).float().mean().item()
            metrics['delta_2'] = (ratio < 1.25**2).float().mean().item()
            metrics['delta_3'] = (ratio < 1.25**3).float().mean().item()
            
            # Correlation
            correlation = torch.corrcoef(torch.stack([predicted.flatten(), ground_truth.flatten()]))[0, 1]
            metrics['correlation'] = correlation.item() if not torch.isnan(correlation) else 0.0
            
            # Scale-Invariant Log Error (SILog)
            log_pred = torch.log(predicted + 1e-8)
            log_gt = torch.log(ground_truth + 1e-8)
            log_diff = log_pred - log_gt
            silog = torch.sqrt(torch.mean(log_diff**2) - torch.mean(log_diff)**2)
            metrics['silog'] = silog.item()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Accuracy computation failed: {e}")
            return {'error': str(e)}
    
    def compute_edge_preservation(self, predicted: torch.Tensor, 
                                 ground_truth: torch.Tensor,
                                 original_image: Optional[torch.Tensor] = None) -> float:
        """Compute edge preservation score."""
        try:
            # Compute edges for predicted depth
            pred_edges = self._compute_edges(predicted)
            gt_edges = self._compute_edges(ground_truth)
            
            # Compute correlation between edge maps
            pred_flat = pred_edges.flatten()
            gt_flat = gt_edges.flatten()
            
            correlation = F.cosine_similarity(pred_flat, gt_flat, dim=0)
            
            return correlation.item()
            
        except Exception as e:
            logger.warning(f"Edge preservation computation failed: {e}")
            return 0.0
    
    def _compute_edges(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel operator."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth_map, sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, sobel_y, padding=1)
        
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges


class DatasetLoader:
    """Load and manage benchmark datasets."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.test_data = []
        self.ground_truth = []
        
    def load_synthetic_data(self, num_samples: int = 100) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Generate synthetic test data."""
        test_samples = []
        
        for i in range(num_samples):
            # Generate random image
            height, width = np.random.choice([(480, 640), (720, 1280)])
            image = torch.randn(1, 3, height, width)
            
            # Generate synthetic ground truth depth
            if self.config.ground_truth_validation:
                # Simple depth pattern
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(0, 1, height),
                    torch.linspace(0, 1, width),
                    indexing='ij'
                )
                depth_gt = 5.0 + 3.0 * (y_coords + 0.5 * torch.sin(x_coords * 4))
                depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
            else:
                depth_gt = None
            
            test_samples.append((image, depth_gt))
        
        return test_samples
    
    def load_real_dataset(self, dataset_path: str) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Load real dataset from path."""
        # Placeholder for real dataset loading
        logger.warning("Real dataset loading not implemented, using synthetic data")
        return self.load_synthetic_data()


class DepthEstimationBenchmark:
    """
    Comprehensive benchmarking suite for depth estimation models.
    
    Features:
    - Performance benchmarking across different configurations
    - Accuracy validation against ground truth
    - Hardware utilization monitoring
    - Memory profiling and optimization analysis
    - Detailed reporting and visualization
    """
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite."""
        self.config = config
        
        # Initialize components
        self.hardware_monitor = HardwareMonitor(config)
        self.accuracy_validator = AccuracyValidator(config)
        self.dataset_loader = DatasetLoader(config)
        
        # Results storage
        self.results = {}
        self.detailed_results = []
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Benchmark initialized: {config.mode.value} mode")
    
    def benchmark_model(self, model, model_name: str) -> BenchmarkResult:
        """Benchmark a single depth estimation model."""
        logger.info(f"Benchmarking {model_name}")
        
        results = BenchmarkResult()
        results.test_config = {
            'model_name': model_name,
            'mode': self.config.mode.value,
            'iterations': self.config.iterations
        }
        
        # Load test data
        test_data = self.dataset_loader.load_synthetic_data(self.config.iterations)
        
        if self.config.mode == BenchmarkMode.PERFORMANCE:
            results = self._benchmark_performance(model, test_data, results)
        elif self.config.mode == BenchmarkMode.ACCURACY:
            results = self._benchmark_accuracy(model, test_data, results)
        elif self.config.mode == BenchmarkMode.MEMORY:
            results = self._benchmark_memory(model, test_data, results)
        elif self.config.mode == BenchmarkMode.LATENCY:
            results = self._benchmark_latency(model, test_data, results)
        elif self.config.mode == BenchmarkMode.THROUGHPUT:
            results = self._benchmark_throughput(model, test_data, results)
        else:  # COMPREHENSIVE
            results = self._benchmark_comprehensive(model, test_data, results)
        
        self.results[model_name] = results
        return results
    
    def benchmark_ensemble(self, ensemble_estimator, test_scenarios: Optional[List[Dict]] = None) -> Dict[str, BenchmarkResult]:
        """Benchmark ensemble depth estimator with various configurations."""
        logger.info("Benchmarking ensemble estimator")
        
        if test_scenarios is None:
            test_scenarios = [
                {'fusion_method': 'weighted_average', 'parallel_inference': False},
                {'fusion_method': 'confidence_based', 'parallel_inference': True},
                {'fusion_method': 'adaptive_selection', 'parallel_inference': True},
                {'fusion_method': 'learned_fusion', 'parallel_inference': True}
            ]
        
        ensemble_results = {}
        
        for i, scenario in enumerate(test_scenarios):
            scenario_name = f"ensemble_scenario_{i+1}"
            logger.info(f"Testing scenario: {scenario}")
            
            # Update ensemble configuration
            for key, value in scenario.items():
                if hasattr(ensemble_estimator.config, key):
                    setattr(ensemble_estimator.config, key, value)
            
            # Benchmark scenario
            result = self.benchmark_model(ensemble_estimator, scenario_name)
            result.test_config.update(scenario)
            ensemble_results[scenario_name] = result
        
        return ensemble_results
    
    def _benchmark_performance(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark model performance (speed/throughput)."""
        inference_times = []
        
        # Warmup
        warmup_data = test_data[:self.config.warmup_iterations]
        for image, _ in warmup_data:
            try:
                if hasattr(model, 'estimate_depth_ensemble'):
                    _ = model.estimate_depth_ensemble(image)
                else:
                    _ = model.estimate_depth(image)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Actual benchmark
        self.hardware_monitor.start_monitoring()
        
        for image, _ in tqdm(test_data[:self.config.iterations], desc="Performance benchmark"):
            start_time = time.time()
            
            try:
                if hasattr(model, 'estimate_depth_ensemble'):
                    _ = model.estimate_depth_ensemble(image)
                else:
                    _ = model.estimate_depth(image)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            except Exception as e:
                logger.warning(f"Inference failed: {e}")
                continue
            
            self.hardware_monitor.collect_metrics()
        
        self.hardware_monitor.stop_monitoring()
        
        # Compute performance metrics
        if inference_times:
            result.avg_inference_time = np.mean(inference_times)
            result.std_inference_time = np.std(inference_times)
            result.fps = 1.0 / result.avg_inference_time
            result.throughput = len(inference_times) / sum(inference_times)
        
        # Add hardware metrics
        hw_summary = self.hardware_monitor.get_summary()
        result.peak_gpu_memory = hw_summary.get('gpu_memory_peak', 0)
        result.avg_gpu_memory = hw_summary.get('gpu_memory_avg', 0)
        result.gpu_utilization = hw_summary.get('gpu_utilization_avg', 0)
        result.cpu_utilization = hw_summary.get('cpu_utilization_avg', 0)
        
        return result
    
    def _benchmark_accuracy(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark model accuracy against ground truth."""
        if not self.config.ground_truth_validation:
            logger.warning("Ground truth validation disabled, skipping accuracy benchmark")
            return result
        
        accuracy_metrics = {
            'mae': [],
            'rmse': [],
            'relative_error': [],
            'delta_1': [],
            'correlation': [],
            'edge_preservation': []
        }
        
        for image, ground_truth in tqdm(test_data[:self.config.iterations], desc="Accuracy benchmark"):
            if ground_truth is None:
                continue
            
            try:
                # Get prediction
                if hasattr(model, 'estimate_depth_ensemble'):
                    predicted = model.estimate_depth_ensemble(image)
                else:
                    predicted = model.estimate_depth(image)
                
                # Compute metrics
                metrics = self.accuracy_validator.compute_depth_metrics(predicted, ground_truth)
                
                for metric_name in accuracy_metrics.keys():
                    if metric_name in metrics:
                        accuracy_metrics[metric_name].append(metrics[metric_name])
                
                # Edge preservation
                edge_score = self.accuracy_validator.compute_edge_preservation(predicted, ground_truth)
                accuracy_metrics['edge_preservation'].append(edge_score)
                
            except Exception as e:
                logger.warning(f"Accuracy computation failed: {e}")
                continue
        
        # Compute average metrics
        for metric_name, values in accuracy_metrics.items():
            if values:
                result.accuracy_metrics[f'{metric_name}_mean'] = np.mean(values)
                result.accuracy_metrics[f'{metric_name}_std'] = np.std(values)
        
        return result
    
    def _benchmark_memory(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark memory usage across different batch sizes."""
        memory_results = {}
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create batch
            batch_data = []
            for i in range(min(batch_size, len(test_data))):
                batch_data.append(test_data[i][0])
            
            if len(batch_data) < batch_size:
                # Repeat data if needed
                while len(batch_data) < batch_size:
                    batch_data.extend(batch_data[:min(batch_size - len(batch_data), len(batch_data))])
                batch_data = batch_data[:batch_size]
            
            # Stack into batch tensor
            batch_tensor = torch.cat(batch_data, dim=0)
            
            try:
                # Reset memory stats
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Run inference
                if hasattr(model, 'estimate_depth_ensemble'):
                    _ = model.estimate_depth_ensemble(batch_tensor)
                else:
                    _ = model.estimate_depth(batch_tensor)
                
                # Collect memory stats
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    memory_results[f'batch_{batch_size}'] = peak_memory
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at batch size {batch_size}")
                    memory_results[f'batch_{batch_size}'] = float('inf')
                else:
                    raise
            except Exception as e:
                logger.warning(f"Memory benchmark failed for batch size {batch_size}: {e}")
                memory_results[f'batch_{batch_size}'] = None
        
        result.model_breakdown['memory_usage'] = memory_results
        
        # Find optimal batch size
        valid_results = {k: v for k, v in memory_results.items() if v is not None and v != float('inf')}
        if valid_results:
            result.peak_gpu_memory = max(valid_results.values())
        
        return result
    
    def _benchmark_latency(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark latency across different input resolutions."""
        latency_results = {}
        
        for resolution in self.config.test_resolutions:
            height, width = resolution
            logger.info(f"Testing resolution: {height}x{width}")
            
            # Create test input at specific resolution
            test_input = torch.randn(1, 3, height, width)
            
            # Warmup
            for _ in range(5):
                try:
                    if hasattr(model, 'estimate_depth_ensemble'):
                        _ = model.estimate_depth_ensemble(test_input)
                    else:
                        _ = model.estimate_depth(test_input)
                except Exception:
                    pass
            
            # Measure latency
            latencies = []
            for _ in range(20):  # Multiple runs for stable measurement
                start_time = time.perf_counter()
                
                try:
                    if hasattr(model, 'estimate_depth_ensemble'):
                        _ = model.estimate_depth_ensemble(test_input)
                    else:
                        _ = model.estimate_depth(test_input)
                    
                    latency = time.perf_counter() - start_time
                    latencies.append(latency)
                    
                except Exception as e:
                    logger.warning(f"Latency test failed for {resolution}: {e}")
                    break
            
            if latencies:
                latency_results[f'{height}x{width}'] = {
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies)
                }
        
        result.model_breakdown['latency'] = latency_results
        
        return result
    
    def _benchmark_throughput(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark throughput with continuous processing."""
        # Prepare continuous data stream
        num_samples = min(self.config.iterations, len(test_data))
        test_images = [data[0] for data in test_data[:num_samples]]
        
        # Measure throughput
        start_time = time.time()
        processed_count = 0
        
        self.hardware_monitor.start_monitoring()
        
        for image in tqdm(test_images, desc="Throughput benchmark"):
            try:
                if hasattr(model, 'estimate_depth_ensemble'):
                    _ = model.estimate_depth_ensemble(image)
                else:
                    _ = model.estimate_depth(image)
                
                processed_count += 1
                self.hardware_monitor.collect_metrics()
                
            except Exception as e:
                logger.warning(f"Throughput test iteration failed: {e}")
                continue
        
        total_time = time.time() - start_time
        self.hardware_monitor.stop_monitoring()
        
        if total_time > 0:
            result.throughput = processed_count / total_time
            result.fps = result.throughput  # Same for single frame processing
        
        # Hardware utilization
        hw_summary = self.hardware_monitor.get_summary()
        result.gpu_utilization = hw_summary.get('gpu_utilization_avg', 0)
        result.cpu_utilization = hw_summary.get('cpu_utilization_avg', 0)
        
        return result
    
    def _benchmark_comprehensive(self, model, test_data: List, result: BenchmarkResult) -> BenchmarkResult:
        """Run comprehensive benchmark covering all aspects."""
        logger.info("Running comprehensive benchmark")
        
        # Performance benchmark
        perf_result = self._benchmark_performance(model, test_data, BenchmarkResult())
        result.avg_inference_time = perf_result.avg_inference_time
        result.fps = perf_result.fps
        result.throughput = perf_result.throughput
        
        # Memory benchmark
        mem_result = self._benchmark_memory(model, test_data, BenchmarkResult())
        result.peak_gpu_memory = mem_result.peak_gpu_memory
        result.model_breakdown.update(mem_result.model_breakdown)
        
        # Latency benchmark
        lat_result = self._benchmark_latency(model, test_data, BenchmarkResult())
        if 'latency' not in result.model_breakdown:
            result.model_breakdown['latency'] = {}
        result.model_breakdown['latency'].update(lat_result.model_breakdown.get('latency', {}))
        
        # Accuracy benchmark (if enabled)
        if self.config.ground_truth_validation:
            acc_result = self._benchmark_accuracy(model, test_data, BenchmarkResult())
            result.accuracy_metrics.update(acc_result.accuracy_metrics)
        
        return result
    
    def generate_report(self, output_format: str = "json") -> str:
        """Generate comprehensive benchmark report."""
        report_path = Path(self.config.output_dir) / f"benchmark_report.{output_format}"
        
        if output_format == "json":
            # Convert results to JSON-serializable format
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = {
                    'performance': {
                        'avg_inference_time': result.avg_inference_time,
                        'fps': result.fps,
                        'throughput': result.throughput
                    },
                    'memory': {
                        'peak_gpu_memory': result.peak_gpu_memory,
                        'avg_gpu_memory': result.avg_gpu_memory
                    },
                    'accuracy': result.accuracy_metrics,
                    'hardware': {
                        'gpu_utilization': result.gpu_utilization,
                        'cpu_utilization': result.cpu_utilization
                    },
                    'model_breakdown': result.model_breakdown,
                    'test_config': result.test_config
                }
            
            with open(report_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        logger.info(f"Benchmark report saved: {report_path}")
        return str(report_path)
    
    def generate_plots(self):
        """Generate visualization plots for benchmark results."""
        if not self.config.generate_plots:
            return
        
        try:
            self._plot_performance_comparison()
            self._plot_memory_usage()
            self._plot_accuracy_metrics()
            self._plot_latency_vs_resolution()
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across models."""
        if not self.results:
            return
        
        models = list(self.results.keys())
        fps_values = [self.results[model].fps for model in models]
        memory_values = [self.results[model].peak_gpu_memory for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # FPS comparison
        ax1.bar(models, fps_values)
        ax1.set_title('FPS Comparison')
        ax1.set_ylabel('FPS')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        ax2.bar(models, memory_values)
        ax2.set_title('Peak GPU Memory Usage')
        ax2.set_ylabel('Memory (GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self):
        """Plot memory usage across batch sizes."""
        for model_name, result in self.results.items():
            memory_data = result.model_breakdown.get('memory_usage', {})
            if not memory_data:
                continue
            
            batch_sizes = []
            memory_values = []
            
            for key, value in memory_data.items():
                if value is not None and value != float('inf'):
                    batch_size = int(key.split('_')[1])
                    batch_sizes.append(batch_size)
                    memory_values.append(value)
            
            if batch_sizes:
                plt.figure(figsize=(8, 6))
                plt.plot(batch_sizes, memory_values, marker='o', label=model_name)
                plt.xlabel('Batch Size')
                plt.ylabel('Peak GPU Memory (GB)')
                plt.title(f'Memory Usage vs Batch Size - {model_name}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(Path(self.config.output_dir) / f'memory_usage_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_accuracy_metrics(self):
        """Plot accuracy metrics comparison."""
        if not any(result.accuracy_metrics for result in self.results.values()):
            return
        
        # Collect metrics
        metrics_data = {}
        models = []
        
        for model_name, result in self.results.items():
            if result.accuracy_metrics:
                models.append(model_name)
                for metric_name, value in result.accuracy_metrics.items():
                    if metric_name.endswith('_mean'):
                        base_name = metric_name.replace('_mean', '')
                        if base_name not in metrics_data:
                            metrics_data[base_name] = []
                        metrics_data[base_name].append(value)
        
        if not metrics_data:
            return
        
        # Create subplot for each metric
        num_metrics = len(metrics_data)
        fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i < len(axes):
                axes[i].bar(models, values)
                axes[i].set_title(f'{metric_name.upper()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / 'accuracy_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_vs_resolution(self):
        """Plot latency vs input resolution."""
        for model_name, result in self.results.items():
            latency_data = result.model_breakdown.get('latency', {})
            if not latency_data:
                continue
            
            resolutions = []
            latencies = []
            
            for res_str, lat_data in latency_data.items():
                if isinstance(lat_data, dict) and 'mean' in lat_data:
                    resolutions.append(res_str)
                    latencies.append(lat_data['mean'] * 1000)  # Convert to ms
            
            if resolutions:
                plt.figure(figsize=(10, 6))
                plt.plot(resolutions, latencies, marker='o', linewidth=2, markersize=8)
                plt.xlabel('Input Resolution')
                plt.ylabel('Latency (ms)')
                plt.title(f'Latency vs Input Resolution - {model_name}')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.savefig(Path(self.config.output_dir) / f'latency_vs_resolution_{model_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def cleanup(self):
        """Cleanup benchmark resources."""
        self.results.clear()
        self.detailed_results.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Benchmark cleanup completed") 