"""
Comprehensive Test Suite for Ensemble Depth Estimation

This module provides extensive testing for the ensemble depth estimation system,
including unit tests, integration tests, performance tests, and validation tests.
"""

import os
import sys
import time
import logging
import warnings
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pytest
from PIL import Image
from omegaconf import DictConfig, OmegaConf

# Add source directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_ad_placement.depth_estimation.ensemble_estimator import (
    EnsembleDepthEstimator, 
    EnsembleConfig, 
    EnsembleMetrics,
    FusionMethod,
    SceneComplexity,
    SceneComplexityAnalyzer,
    LearnedFusionNetwork,
    QualityAssessmentModule
)
from src.video_ad_placement.depth_estimation.benchmarking import (
    DepthEstimationBenchmark,
    BenchmarkConfig,
    BenchmarkMode,
    HardwareMonitor,
    AccuracyValidator
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMarigoldEstimator:
    """Mock Marigold estimator for testing."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
    def estimate_depth(self, frames: torch.Tensor) -> torch.Tensor:
        """Mock depth estimation."""
        batch_size, channels, height, width = frames.shape
        # Return mock depth map
        depth = torch.randn(batch_size, 1, height, width)
        return torch.abs(depth) + 0.1  # Ensure positive depth


class MockDepthProEstimator:
    """Mock Depth Pro estimator for testing."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        
    def estimate_depth(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mock depth estimation with confidence."""
        batch_size, channels, height, width = frames.shape
        # Return mock depth map and confidence
        depth = torch.randn(batch_size, 1, height, width)
        confidence = torch.rand(batch_size, 1, height, width)
        return torch.abs(depth) + 0.1, confidence


class TestEnsembleConfig(unittest.TestCase):
    """Test EnsembleConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnsembleConfig()
        
        self.assertEqual(config.fusion_method, FusionMethod.CONFIDENCE_BASED)
        self.assertTrue(config.learned_fusion)
        self.assertTrue(config.adaptive_weights)
        self.assertEqual(config.marigold_weight, 0.6)
        self.assertEqual(config.depth_pro_weight, 0.4)
        self.assertEqual(config.confidence_threshold, 0.7)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnsembleConfig(
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            marigold_weight=0.8,
            depth_pro_weight=0.2,
            parallel_inference=False
        )
        
        self.assertEqual(config.fusion_method, FusionMethod.WEIGHTED_AVERAGE)
        self.assertEqual(config.marigold_weight, 0.8)
        self.assertEqual(config.depth_pro_weight, 0.2)
        self.assertFalse(config.parallel_inference)


class TestSceneComplexityAnalyzer(unittest.TestCase):
    """Test scene complexity analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EnsembleConfig()
        self.analyzer = SceneComplexityAnalyzer(self.config)
    
    def test_analyze_simple_scene(self):
        """Test analysis of simple scene."""
        # Create simple uniform image
        image = torch.ones(3, 100, 100) * 0.5
        
        complexity_score, complexity_level = self.analyzer.analyze_complexity(image)
        
        self.assertIsInstance(complexity_score, float)
        self.assertIn(complexity_level, [
            SceneComplexity.SIMPLE, 
            SceneComplexity.MODERATE, 
            SceneComplexity.COMPLEX, 
            SceneComplexity.VERY_COMPLEX
        ])
        self.assertLessEqual(complexity_score, 1.0)
        self.assertGreaterEqual(complexity_score, 0.0)
    
    def test_analyze_complex_scene(self):
        """Test analysis of complex scene."""
        # Create complex random image
        image = torch.randn(3, 100, 100)
        
        complexity_score, complexity_level = self.analyzer.analyze_complexity(image)
        
        self.assertIsInstance(complexity_score, float)
        self.assertIsInstance(complexity_level, SceneComplexity)
    
    def test_edge_density_computation(self):
        """Test edge density computation."""
        # Create image with edges
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[40:60, :] = 255  # Horizontal edge
        
        edge_density = self.analyzer._compute_edge_density(gray_image)
        
        self.assertIsInstance(edge_density, float)
        self.assertGreater(edge_density, 0.0)
    
    def test_texture_variance_computation(self):
        """Test texture variance computation."""
        # Create textured image
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        texture_variance = self.analyzer._compute_texture_variance(gray_image)
        
        self.assertIsInstance(texture_variance, float)
        self.assertGreaterEqual(texture_variance, 0.0)


class TestLearnedFusionNetwork(unittest.TestCase):
    """Test learned fusion network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion_net = LearnedFusionNetwork()
        self.batch_size = 2
        self.height = 64
        self.width = 64
    
    def test_forward_pass(self):
        """Test forward pass of fusion network."""
        # Create test inputs
        depth1 = torch.randn(self.batch_size, 1, self.height, self.width)
        depth2 = torch.randn(self.batch_size, 1, self.height, self.width)
        conf1 = torch.rand(self.batch_size, 1, self.height, self.width)
        conf2 = torch.rand(self.batch_size, 1, self.height, self.width)
        
        # Run forward pass
        fused_depth, fused_conf, global_weights = self.fusion_net(
            depth1, depth2, conf1, conf2
        )
        
        # Check output shapes
        self.assertEqual(fused_depth.shape, (self.batch_size, 1, self.height, self.width))
        self.assertEqual(fused_conf.shape, (self.batch_size, 1, self.height, self.width))
        self.assertEqual(global_weights.shape, (self.batch_size, 2))
        
        # Check value ranges
        self.assertTrue(torch.all(fused_conf >= 0))
        self.assertTrue(torch.all(fused_conf <= 1))
        self.assertTrue(torch.allclose(global_weights.sum(dim=1), torch.ones(self.batch_size)))
    
    def test_gradient_features(self):
        """Test gradient feature computation."""
        depth = torch.randn(self.batch_size, 1, self.height, self.width)
        
        gradient_features = self.fusion_net._compute_gradient_features(depth)
        
        self.assertEqual(gradient_features.shape, depth.shape)
        self.assertTrue(torch.all(gradient_features >= 0))


class TestQualityAssessmentModule(unittest.TestCase):
    """Test quality assessment module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EnsembleConfig()
        self.quality_assessor = QualityAssessmentModule(self.config)
    
    def test_assess_quality(self):
        """Test quality assessment."""
        depth_map = torch.rand(1, 1, 100, 100) * 10 + 0.1
        confidence_map = torch.rand(1, 1, 100, 100)
        original_image = torch.rand(1, 3, 100, 100)
        
        quality_metrics = self.quality_assessor.assess_quality(
            depth_map, confidence_map, original_image
        )
        
        # Check required metrics
        required_metrics = [
            'depth_coverage', 'depth_variance', 'depth_range',
            'confidence_mean', 'confidence_std', 'high_confidence_ratio',
            'edge_preservation', 'smoothness', 'overall_quality'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, quality_metrics)
            self.assertIsInstance(quality_metrics[metric], float)
    
    def test_edge_preservation(self):
        """Test edge preservation computation."""
        # Create depth map with similar structure to image
        depth_map = torch.rand(1, 1, 50, 50)
        image = torch.rand(1, 3, 50, 50)
        
        edge_score = self.quality_assessor.compute_edge_preservation(depth_map, image)
        
        self.assertIsInstance(edge_score, float)
        self.assertGreaterEqual(edge_score, -1.0)
        self.assertLessEqual(edge_score, 1.0)
    
    def test_smoothness_computation(self):
        """Test smoothness computation."""
        # Create smooth depth map
        depth_map = torch.ones(1, 1, 50, 50) * 5.0
        
        smoothness = self.quality_assessor._compute_smoothness(depth_map)
        
        self.assertIsInstance(smoothness, float)
        self.assertGreater(smoothness, 0.0)


class TestEnsembleDepthEstimator(unittest.TestCase):
    """Test ensemble depth estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.marigold_estimator = MockMarigoldEstimator()
        self.depth_pro_estimator = MockDepthProEstimator()
        self.config = EnsembleConfig()
        
        self.ensemble = EnsembleDepthEstimator(
            self.marigold_estimator,
            self.depth_pro_estimator,
            self.config
        )
    
    def test_initialization(self):
        """Test ensemble estimator initialization."""
        self.assertIsNotNone(self.ensemble.marigold_estimator)
        self.assertIsNotNone(self.ensemble.depth_pro_estimator)
        self.assertIsNotNone(self.ensemble.scene_analyzer)
        self.assertIsNotNone(self.ensemble.quality_assessor)
        self.assertIsInstance(self.ensemble.config, EnsembleConfig)
    
    def test_preprocessing(self):
        """Test frame preprocessing."""
        # Test single PIL image
        pil_image = Image.new('RGB', (640, 480), color='red')
        processed = self.ensemble._preprocess_frames(pil_image)
        
        self.assertEqual(processed.shape, (1, 3, 480, 640))
        self.assertTrue(torch.all(processed >= 0))
        self.assertTrue(torch.all(processed <= 1))
        
        # Test numpy array
        np_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        processed = self.ensemble._preprocess_frames(np_image)
        
        self.assertEqual(processed.shape, (1, 3, 480, 640))
        
        # Test torch tensor
        torch_image = torch.randn(1, 3, 480, 640)
        processed = self.ensemble._preprocess_frames(torch_image)
        
        self.assertEqual(processed.shape, (1, 3, 480, 640))
    
    def test_sequential_inference(self):
        """Test sequential inference."""
        frames = torch.randn(2, 3, 100, 100)
        
        predictions = self.ensemble._sequential_inference(frames)
        
        self.assertIn('marigold', predictions)
        self.assertIn('depth_pro', predictions)
        
        # Check marigold prediction
        marigold_depth, marigold_conf = predictions['marigold']
        self.assertEqual(marigold_depth.shape, (2, 1, 100, 100))
        self.assertEqual(marigold_conf.shape, (2, 1, 100, 100))
        
        # Check depth pro prediction
        depth_pro_depth, depth_pro_conf = predictions['depth_pro']
        self.assertEqual(depth_pro_depth.shape, (2, 1, 100, 100))
        self.assertEqual(depth_pro_conf.shape, (2, 1, 100, 100))
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_parallel_inference(self, mock_executor):
        """Test parallel inference."""
        # Mock the executor
        mock_future = Mock()
        mock_future.result.return_value = (torch.randn(2, 1, 100, 100), torch.rand(2, 1, 100, 100))
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        frames = torch.randn(2, 3, 100, 100)
        
        predictions = self.ensemble._parallel_inference(frames)
        
        # Verify executor was used
        self.assertTrue(mock_executor.called)
    
    def test_weighted_average_fusion(self):
        """Test weighted average fusion."""
        depth1 = torch.ones(2, 1, 50, 50) * 5.0
        depth2 = torch.ones(2, 1, 50, 50) * 3.0
        
        fused = self.ensemble._weighted_average_fusion(depth1, depth2)
        
        expected = (self.config.marigold_weight * depth1 + 
                   self.config.depth_pro_weight * depth2) / (
                   self.config.marigold_weight + self.config.depth_pro_weight)
        
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
    
    def test_confidence_based_fusion(self):
        """Test confidence-based fusion."""
        depth1 = torch.ones(2, 1, 50, 50) * 5.0
        depth2 = torch.ones(2, 1, 50, 50) * 3.0
        conf1 = torch.ones(2, 1, 50, 50) * 0.8
        conf2 = torch.ones(2, 1, 50, 50) * 0.6
        
        fused = self.ensemble._confidence_based_fusion(depth1, depth2, conf1, conf2)
        
        self.assertEqual(fused.shape, (2, 1, 50, 50))
        self.assertTrue(torch.all(fused > 0))
    
    def test_adaptive_selection_fusion(self):
        """Test adaptive selection fusion."""
        depth1 = torch.ones(2, 1, 50, 50) * 5.0
        depth2 = torch.ones(2, 1, 50, 50) * 3.0
        frames = torch.randn(2, 3, 50, 50)
        
        fused = self.ensemble._adaptive_selection_fusion(depth1, depth2, frames)
        
        self.assertEqual(fused.shape, (2, 1, 50, 50))
        # Should select either depth1 or depth2 for each sample
        self.assertTrue(torch.all(torch.logical_or(
            torch.allclose(fused[0], depth1[0]),
            torch.allclose(fused[0], depth2[0])
        )))
    
    def test_ensemble_estimation(self):
        """Test end-to-end ensemble estimation."""
        frames = torch.randn(2, 3, 100, 100)
        
        depth_prediction = self.ensemble.estimate_depth_ensemble(frames)
        
        self.assertEqual(depth_prediction.shape, (2, 1, 100, 100))
        self.assertTrue(torch.all(depth_prediction > 0))
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        frames = torch.randn(1, 3, 100, 100)
        
        # Run estimation to collect metrics
        _ = self.ensemble.estimate_depth_ensemble(frames)
        
        metrics = self.ensemble.get_metrics()
        
        self.assertIsInstance(metrics, EnsembleMetrics)
        self.assertGreater(metrics.total_time, 0)
        self.assertIn('marigold', metrics.fusion_weights)
        self.assertIn('depth_pro', metrics.fusion_weights)
    
    def test_fallback_estimation(self):
        """Test fallback estimation when models fail."""
        frames = torch.randn(1, 3, 100, 100)
        
        # Mock failures
        self.ensemble.marigold_estimator.estimate_depth = Mock(side_effect=Exception("Model failed"))
        self.ensemble.depth_pro_estimator.estimate_depth = Mock(side_effect=Exception("Model failed"))
        
        depth_prediction = self.ensemble._fallback_estimation(frames)
        
        self.assertEqual(depth_prediction.shape, (1, 1, 100, 100))
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        # Should not raise any exceptions
        self.ensemble.cleanup()


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BenchmarkConfig(
            iterations=5,
            warmup_iterations=1,
            output_dir=tempfile.mkdtemp()
        )
        self.benchmark = DepthEstimationBenchmark(self.config)
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        self.assertIsInstance(self.benchmark.config, BenchmarkConfig)
        self.assertIsNotNone(self.benchmark.hardware_monitor)
        self.assertIsNotNone(self.benchmark.accuracy_validator)
        self.assertIsNotNone(self.benchmark.dataset_loader)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        test_data = self.benchmark.dataset_loader.load_synthetic_data(10)
        
        self.assertEqual(len(test_data), 10)
        
        for image, depth_gt in test_data:
            self.assertIsInstance(image, torch.Tensor)
            self.assertEqual(image.dim(), 4)  # BCHW format
            self.assertEqual(image.shape[1], 3)  # RGB channels
    
    def test_accuracy_metrics_computation(self):
        """Test accuracy metrics computation."""
        predicted = torch.rand(1, 1, 100, 100) * 10 + 0.1
        ground_truth = torch.rand(1, 1, 100, 100) * 10 + 0.1
        
        metrics = self.benchmark.accuracy_validator.compute_depth_metrics(
            predicted, ground_truth
        )
        
        required_metrics = ['mae', 'rmse', 'relative_error', 'delta_1', 'correlation', 'silog']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
    
    def test_hardware_monitoring(self):
        """Test hardware monitoring."""
        monitor = self.benchmark.hardware_monitor
        
        monitor.start_monitoring()
        self.assertTrue(monitor.monitoring)
        
        monitor.collect_metrics()
        monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring)
        
        summary = monitor.get_summary()
        self.assertIsInstance(summary, dict)
    
    def test_model_benchmarking(self):
        """Test benchmarking a single model."""
        mock_estimator = MockMarigoldEstimator()
        
        result = self.benchmark.benchmark_model(mock_estimator, "test_model")
        
        self.assertIsNotNone(result)
        self.assertGreater(result.avg_inference_time, 0)
        self.assertGreater(result.fps, 0)
    
    def test_report_generation(self):
        """Test benchmark report generation."""
        # Run a quick benchmark
        mock_estimator = MockMarigoldEstimator()
        self.benchmark.benchmark_model(mock_estimator, "test_model")
        
        # Generate report
        report_path = self.benchmark.generate_report("json")
        
        self.assertTrue(os.path.exists(report_path))
        
        # Clean up
        os.remove(report_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.marigold_estimator = MockMarigoldEstimator()
        self.depth_pro_estimator = MockDepthProEstimator()
        
    def test_full_pipeline_integration(self):
        """Test complete pipeline integration."""
        # Create ensemble with different fusion methods
        fusion_methods = [
            FusionMethod.WEIGHTED_AVERAGE,
            FusionMethod.CONFIDENCE_BASED,
            FusionMethod.ADAPTIVE_SELECTION
        ]
        
        for fusion_method in fusion_methods:
            config = EnsembleConfig(fusion_method=fusion_method)
            ensemble = EnsembleDepthEstimator(
                self.marigold_estimator,
                self.depth_pro_estimator,
                config
            )
            
            # Test with different input types
            test_inputs = [
                torch.randn(1, 3, 240, 320),
                torch.randn(2, 3, 480, 640),
                [torch.randn(1, 3, 100, 100) for _ in range(3)]
            ]
            
            for test_input in test_inputs:
                try:
                    result = ensemble.estimate_depth_ensemble(test_input)
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertTrue(torch.all(result > 0))
                except Exception as e:
                    self.fail(f"Pipeline failed for {fusion_method} with input {type(test_input)}: {e}")
            
            ensemble.cleanup()
    
    def test_benchmarking_integration(self):
        """Test benchmarking integration with ensemble."""
        config = EnsembleConfig()
        ensemble = EnsembleDepthEstimator(
            self.marigold_estimator,
            self.depth_pro_estimator,
            config
        )
        
        # Create benchmark
        benchmark_config = BenchmarkConfig(
            iterations=3,
            warmup_iterations=1,
            output_dir=tempfile.mkdtemp()
        )
        benchmark = DepthEstimationBenchmark(benchmark_config)
        
        # Run benchmark
        result = benchmark.benchmark_model(ensemble, "ensemble_test")
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.avg_inference_time, float)
        self.assertGreater(result.avg_inference_time, 0)
        
        ensemble.cleanup()
    
    def test_error_handling_integration(self):
        """Test error handling across the system."""
        # Test with failing estimators
        failing_marigold = Mock()
        failing_marigold.estimate_depth = Mock(side_effect=RuntimeError("CUDA OOM"))
        failing_marigold.device = torch.device('cpu')
        
        working_depth_pro = MockDepthProEstimator()
        
        config = EnsembleConfig()
        ensemble = EnsembleDepthEstimator(
            failing_marigold,
            working_depth_pro,
            config
        )
        
        # Should fall back gracefully
        frames = torch.randn(1, 3, 100, 100)
        result = ensemble.estimate_depth_ensemble(frames)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 1, 100, 100))
        
        ensemble.cleanup()


class TestPerformance(unittest.TestCase):
    """Performance tests for the system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.marigold_estimator = MockMarigoldEstimator()
        self.depth_pro_estimator = MockDepthProEstimator()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of ensemble processing."""
        config = EnsembleConfig(memory_efficient=True)
        ensemble = EnsembleDepthEstimator(
            self.marigold_estimator,
            self.depth_pro_estimator,
            config
        )
        
        # Process multiple frames and check memory doesn't grow
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for _ in range(10):
            frames = torch.randn(2, 3, 480, 640)
            _ = ensemble.estimate_depth_ensemble(frames)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        self.assertLess(memory_growth, 1e9)  # Less than 1GB growth
        
        ensemble.cleanup()
    
    def test_parallel_vs_sequential_performance(self):
        """Test performance difference between parallel and sequential inference."""
        # Sequential config
        config_seq = EnsembleConfig(parallel_inference=False)
        ensemble_seq = EnsembleDepthEstimator(
            self.marigold_estimator,
            self.depth_pro_estimator,
            config_seq
        )
        
        # Parallel config
        config_par = EnsembleConfig(parallel_inference=True)
        ensemble_par = EnsembleDepthEstimator(
            self.marigold_estimator,
            self.depth_pro_estimator,
            config_par
        )
        
        frames = torch.randn(4, 3, 240, 320)
        
        # Time sequential
        start_time = time.time()
        _ = ensemble_seq.estimate_depth_ensemble(frames)
        seq_time = time.time() - start_time
        
        # Time parallel
        start_time = time.time()
        _ = ensemble_par.estimate_depth_ensemble(frames)
        par_time = time.time() - start_time
        
        # Parallel should be faster or at least comparable
        # (Note: With mocked estimators, the difference might be small)
        self.assertLessEqual(par_time, seq_time * 1.5)  # Allow some overhead
        
        ensemble_seq.cleanup()
        ensemble_par.cleanup()


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    logger.info("Starting comprehensive test suite for ensemble depth estimation")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnsembleConfig,
        TestSceneComplexityAnalyzer,
        TestLearnedFusionNetwork,
        TestQualityAssessmentModule,
        TestEnsembleDepthEstimator,
        TestBenchmarking,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"{test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"{test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 