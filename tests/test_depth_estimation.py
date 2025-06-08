"""
Unit Tests for Marigold Depth Estimation Module

Comprehensive test suite covering all aspects of the depth estimation system
including temporal consistency, memory management, and quality assessment.
"""

import pytest
import torch
import numpy as np
import cv2
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from video_ad_placement.depth_estimation import (
    MarigoldDepthEstimator,
    DepthEstimationConfig,
    DepthEstimationMetrics,
    PrecisionMode,
    QualityMode,
    TemporalSmoothingMethod,
    GPUMemoryManager,
    OpticalFlowProcessor,
    compute_depth_quality_metrics
)

from video_ad_placement.depth_estimation.temporal_consistency import (
    TemporalConsistencyProcessor,
    TemporalFilterConfig,
    OpticalFlowWarping,
    ExponentialTemporalFilter,
    BilateralTemporalFilter,
    KalmanTemporalFilter
)

from video_ad_placement.depth_estimation.utils import (
    DepthVisualization,
    VideoDepthProcessor,
    DepthMetricsCollector,
    VideoProcessingConfig
)


class TestDepthEstimationConfig:
    """Test depth estimation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DepthEstimationConfig()
        
        assert config.model_name == "prs-eth/marigold-lcm-v1-0"
        assert config.precision_mode == PrecisionMode.FP16
        assert config.quality_mode == QualityMode.BALANCED
        assert config.enable_temporal_consistency is True
        assert config.temporal_smoothing_method == TemporalSmoothingMethod.OPTICAL_FLOW
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DepthEstimationConfig(
            quality_mode=QualityMode.FAST,
            batch_size=8,
            max_resolution=(720, 1280),
            temporal_alpha=0.5
        )
        
        assert config.quality_mode == QualityMode.FAST
        assert config.batch_size == 8
        assert config.max_resolution == (720, 1280)
        assert config.temporal_alpha == 0.5


class TestGPUMemoryManager:
    """Test GPU memory management utilities."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        manager = GPUMemoryManager(memory_limit_gb=4.0)
        assert manager.memory_limit_bytes == 4.0 * 1024**3
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_usage_no_gpu(self, mock_cuda):
        """Test memory usage when no GPU is available."""
        manager = GPUMemoryManager()
        usage = manager.get_memory_usage()
        
        assert usage["allocated"] == 0
        assert usage["cached"] == 0
        assert usage["total"] == 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)
    @patch('torch.cuda.memory_reserved', return_value=2 * 1024**3)
    @patch('torch.cuda.get_device_properties')
    def test_memory_usage_with_gpu(self, mock_props, mock_reserved, mock_allocated, mock_cuda):
        """Test memory usage with GPU available."""
        # Mock GPU properties
        mock_props.return_value.total_memory = 8 * 1024**3
        
        manager = GPUMemoryManager()
        usage = manager.get_memory_usage()
        
        assert usage["allocated"] == 1.0  # 1GB
        assert usage["cached"] == 2.0     # 2GB
        assert usage["total"] == 8.0      # 8GB
    
    def test_batch_size_estimation(self):
        """Test batch size estimation."""
        manager = GPUMemoryManager()
        
        # Mock available memory
        with patch.object(manager, 'get_available_memory', return_value=4 * 1024**3):  # 4GB
            batch_size = manager.estimate_batch_size((1080, 1920), base_batch_size=8)
            assert batch_size >= 1
            assert batch_size <= 8
    
    def test_memory_context(self):
        """Test memory context manager."""
        manager = GPUMemoryManager()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=0), \
             patch('torch.cuda.empty_cache') as mock_empty:
            
            with manager.memory_context():
                pass
            
            mock_empty.assert_called_once()


class TestOpticalFlowProcessor:
    """Test optical flow processing."""
    
    def test_initialization(self):
        """Test optical flow processor initialization."""
        processor = OpticalFlowProcessor()
        assert processor is not None
    
    def test_compute_flow_with_valid_frames(self):
        """Test optical flow computation with valid frames."""
        processor = OpticalFlowProcessor()
        
        # Create test frames
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        flow = processor.compute_flow(frame1, frame2)
        
        # Flow may be None if OpenCV fails, but should not raise exception
        assert flow is None or isinstance(flow, np.ndarray)
    
    def test_warp_depth_no_flow(self):
        """Test depth warping with no flow."""
        processor = OpticalFlowProcessor()
        depth_map = torch.randn(100, 100)
        
        warped = processor.warp_depth(depth_map, None)
        
        assert torch.equal(warped, depth_map)
    
    def test_warp_depth_with_flow(self):
        """Test depth warping with flow."""
        processor = OpticalFlowProcessor()
        depth_map = torch.randn(100, 100)
        flow = np.random.randn(100, 100, 2).astype(np.float32)
        
        # Should not raise exception
        warped = processor.warp_depth(depth_map, flow)
        assert warped.shape == depth_map.shape


class TestMarigoldDepthEstimator:
    """Test Marigold depth estimator."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'model_name': 'test-model',
            'model_cache_dir': '/tmp/test_cache',
            'device': 'cpu',
            'quality_mode': 'fast',
            'batch_size': 2,
            'enable_temporal_consistency': False
        }
    
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', False)
    def test_initialization_no_dependencies(self, mock_config):
        """Test initialization when Marigold dependencies are not available."""
        with pytest.raises(ImportError):
            MarigoldDepthEstimator(mock_config)
    
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True)
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline')
    def test_initialization_success(self, mock_pipeline, mock_config):
        """Test successful initialization."""
        # Mock the pipeline
        mock_pipeline.from_pretrained.return_value = Mock()
        
        estimator = MarigoldDepthEstimator(mock_config)
        
        assert estimator.device.type == 'cpu'
        assert estimator.config.quality_mode == QualityMode.FAST
    
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True)
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline')
    def test_quality_configuration(self, mock_pipeline, mock_config):
        """Test quality mode configuration."""
        mock_pipeline.from_pretrained.return_value = Mock()
        
        mock_config['quality_mode'] = 'ultra'
        estimator = MarigoldDepthEstimator(mock_config)
        
        assert estimator.config.ensemble_size == 10
        assert estimator.config.num_inference_steps == 20
    
    def test_preprocess_frames_pil_image(self):
        """Test frame preprocessing with PIL Image."""
        with patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True), \
             patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline'):
            
            estimator = MarigoldDepthEstimator({'device': 'cpu'})
            
            # Create test PIL image
            test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            
            processed = estimator._preprocess_frames(test_image)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.ndim == 4  # BCHW
            assert processed.shape[0] == 1  # Batch size 1
    
    def test_preprocess_frames_numpy_array(self):
        """Test frame preprocessing with numpy array."""
        with patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True), \
             patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline'):
            
            estimator = MarigoldDepthEstimator({'device': 'cpu'})
            
            # Single frame
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            processed = estimator._preprocess_frames(test_frame)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.ndim == 4
    
    def test_preprocess_frames_torch_tensor(self):
        """Test frame preprocessing with torch tensor."""
        with patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True), \
             patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline'):
            
            estimator = MarigoldDepthEstimator({'device': 'cpu'})
            
            # Single frame tensor
            test_tensor = torch.randn(3, 100, 100)
            processed = estimator._preprocess_frames(test_tensor)
            
            assert isinstance(processed, torch.Tensor)
            assert processed.ndim == 4
    
    def test_postprocess_depth(self):
        """Test depth map postprocessing."""
        with patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True), \
             patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline'):
            
            config = {
                'device': 'cpu',
                'normalize_output': True,
                'depth_range': (0.0, 10.0),
                'output_dtype': torch.float32
            }
            estimator = MarigoldDepthEstimator(config)
            
            # Test depth maps
            depth_maps = torch.randn(2, 1, 100, 100) * 20 - 5  # Range -5 to 15
            
            processed = estimator._postprocess_depth(depth_maps)
            
            assert processed.dtype == torch.float32
            assert processed.min() >= 0.0
            assert processed.max() <= 1.0
    
    def test_reset_state(self):
        """Test state reset functionality."""
        with patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True), \
             patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline'):
            
            estimator = MarigoldDepthEstimator({'device': 'cpu'})
            
            # Set some state
            estimator.previous_depth = torch.randn(100, 100)
            estimator.depth_history = [torch.randn(100, 100) for _ in range(3)]
            
            estimator.reset_state()
            
            assert estimator.previous_depth is None
            assert len(estimator.depth_history) == 0
            assert len(estimator.flow_history) == 0


class TestTemporalConsistency:
    """Test temporal consistency processing."""
    
    def test_temporal_filter_config(self):
        """Test temporal filter configuration."""
        config = TemporalFilterConfig(
            window_size=10,
            alpha=0.5,
            bilateral_sigma_intensity=0.2
        )
        
        assert config.window_size == 10
        assert config.alpha == 0.5
        assert config.bilateral_sigma_intensity == 0.2
    
    def test_exponential_temporal_filter(self):
        """Test exponential temporal filter."""
        filter_obj = ExponentialTemporalFilter(alpha=0.3)
        
        # First frame
        depth1 = torch.randn(50, 50)
        result1 = filter_obj.filter(depth1, [])
        
        assert torch.equal(result1, depth1)
        
        # Second frame
        depth2 = torch.randn(50, 50)
        result2 = filter_obj.filter(depth2, [depth1])
        
        # Should be blend of previous and current
        assert not torch.equal(result2, depth2)
        assert not torch.equal(result2, depth1)
    
    def test_bilateral_temporal_filter(self):
        """Test bilateral temporal filter."""
        config = TemporalFilterConfig(window_size=3)
        filter_obj = BilateralTemporalFilter(config)
        
        depths = [torch.randn(20, 20) for _ in range(5)]
        
        for i, depth in enumerate(depths):
            result = filter_obj.filter(depth, depths[:i])
            assert result.shape == depth.shape
    
    def test_kalman_temporal_filter(self):
        """Test Kalman temporal filter."""
        config = TemporalFilterConfig()
        filter_obj = KalmanTemporalFilter(config)
        
        depth1 = torch.randn(30, 30)
        result1 = filter_obj.filter(depth1, [])
        
        assert torch.equal(result1, depth1)
        assert filter_obj.state is not None
        assert filter_obj.covariance is not None
        
        # Second frame
        depth2 = torch.randn(30, 30)
        result2 = filter_obj.filter(depth2, [depth1])
        
        assert result2.shape == depth2.shape
    
    def test_optical_flow_warping(self):
        """Test optical flow warping."""
        warping = OpticalFlowWarping()
        
        # Test dense flow computation
        frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        flow = warping.compute_dense_flow(frame1, frame2)
        
        # Flow might be None if OpenCV fails
        if flow is not None:
            assert flow.shape == (100, 100, 2)
    
    def test_depth_map_warping(self):
        """Test depth map warping with optical flow."""
        warping = OpticalFlowWarping()
        
        depth_map = torch.randn(50, 50)
        flow = np.random.randn(50, 50, 2).astype(np.float32)
        
        warped_depth, occlusion_mask = warping.warp_depth_map(depth_map, flow)
        
        assert warped_depth.shape == depth_map.shape
        assert occlusion_mask.shape == depth_map.shape
        assert occlusion_mask.dtype == torch.bool
    
    def test_temporal_consistency_processor(self):
        """Test temporal consistency processor."""
        config = TemporalFilterConfig(window_size=3)
        processor = TemporalConsistencyProcessor(config)
        
        # Process sequence of frames
        for i in range(5):
            depth = torch.randn(40, 40)
            frame = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
            
            result = processor.process_frame(depth, frame, i)
            
            assert result.shape == depth.shape
        
        # Check consistency score
        score = processor.compute_temporal_consistency_score()
        assert 0.0 <= score <= 1.0


class TestDepthVisualization:
    """Test depth visualization utilities."""
    
    def test_depth_visualization_initialization(self):
        """Test depth visualization initialization."""
        viz = DepthVisualization(colormap="viridis")
        assert viz.colormap == "viridis"
    
    def test_depth_to_colormap(self):
        """Test depth to colormap conversion."""
        viz = DepthVisualization()
        
        depth_map = torch.randn(100, 100)
        colored = viz.depth_to_colormap(depth_map)
        
        assert colored.shape == (100, 100, 3)
        assert colored.dtype == np.uint8
        assert colored.min() >= 0
        assert colored.max() <= 255
    
    def test_side_by_side_visualization(self):
        """Test side-by-side visualization."""
        viz = DepthVisualization()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = torch.randn(100, 100)
        
        combined = viz.create_side_by_side(image, depth_map)
        
        assert combined.shape == (100, 200, 3)  # Width doubled
    
    def test_overlay_visualization(self):
        """Test overlay visualization."""
        viz = DepthVisualization()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = torch.randn(100, 100)
        
        overlay = viz.create_overlay(image, depth_map, alpha=0.5)
        
        assert overlay.shape == image.shape
    
    def test_save_depth_visualization(self):
        """Test saving depth visualization."""
        viz = DepthVisualization()
        
        depth_map = torch.randn(50, 50)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            viz.save_depth_visualization(depth_map, tmp.name)
            
            # Check file was created
            assert os.path.exists(tmp.name)
            
            # Cleanup
            os.unlink(tmp.name)


class TestVideoDepthProcessor:
    """Test video depth processing."""
    
    def test_video_processor_initialization(self):
        """Test video processor initialization."""
        config = VideoProcessingConfig(
            output_fps=25,
            visualization_mode="side_by_side",
            depth_colormap="jet"
        )
        
        processor = VideoDepthProcessor(config)
        
        assert processor.config.output_fps == 25
        assert processor.config.visualization_mode == "side_by_side"
        assert processor.visualizer.colormap == "jet"


class TestDepthMetricsCollector:
    """Test depth metrics collection."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = DepthMetricsCollector()
        
        assert len(collector.frame_metrics) == 0
        assert len(collector.sequence_metrics) == 0
    
    def test_collect_frame_metrics(self):
        """Test frame metrics collection."""
        collector = DepthMetricsCollector()
        
        depth_map = torch.randn(100, 100)
        metrics = collector.collect_frame_metrics(
            frame_idx=0,
            depth_map=depth_map,
            processing_time=0.1,
            gpu_memory=2.5
        )
        
        assert 'frame_idx' in metrics
        assert 'processing_time' in metrics
        assert 'mean_depth' in metrics
        assert 'gpu_memory_gb' in metrics
        assert 'edge_density' in metrics
        
        assert len(collector.frame_metrics) == 1
    
    def test_compute_sequence_metrics(self):
        """Test sequence metrics computation."""
        collector = DepthMetricsCollector()
        
        # Collect metrics for multiple frames
        for i in range(10):
            depth_map = torch.randn(50, 50)
            collector.collect_frame_metrics(i, depth_map, 0.1 + i * 0.01)
        
        sequence_metrics = collector.compute_sequence_metrics()
        
        assert 'total_frames' in sequence_metrics
        assert 'average_fps' in sequence_metrics
        assert 'temporal_consistency' in sequence_metrics
        assert sequence_metrics['total_frames'] == 10
    
    def test_temporal_consistency_computation(self):
        """Test temporal consistency computation."""
        collector = DepthMetricsCollector()
        
        # Perfect consistency (constant depth)
        consistent_series = [1.0] * 10
        consistency = collector._compute_temporal_consistency(consistent_series)
        assert consistency == 1.0
        
        # Variable series
        variable_series = [1.0, 2.0, 1.5, 3.0, 2.5]
        consistency = collector._compute_temporal_consistency(variable_series)
        assert 0.0 <= consistency <= 1.0
    
    def test_save_metrics(self):
        """Test metrics saving."""
        collector = DepthMetricsCollector()
        
        # Add some metrics
        depth_map = torch.randn(30, 30)
        collector.collect_frame_metrics(0, depth_map, 0.1)
        collector.compute_sequence_metrics()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            collector.save_metrics(tmp.name)
            
            # Check file exists
            assert os.path.exists(tmp.name)
            
            # Check file content
            import json
            with open(tmp.name, 'r') as f:
                data = json.load(f)
            
            assert 'sequence_metrics' in data
            assert 'frame_metrics' in data
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_summary_report(self):
        """Test summary report generation."""
        collector = DepthMetricsCollector()
        
        # Add metrics
        for i in range(5):
            depth_map = torch.randn(25, 25)
            collector.collect_frame_metrics(i, depth_map, 0.1)
        
        report = collector.get_summary_report()
        
        assert "Depth Estimation Metrics Summary" in report
        assert "Total Frames Processed" in report
        assert "Average FPS" in report


class TestQualityMetrics:
    """Test depth quality assessment."""
    
    def test_quality_metrics_basic(self):
        """Test basic quality metrics computation."""
        depth_map = torch.randn(100, 100)
        
        metrics = compute_depth_quality_metrics(depth_map)
        
        assert 'mean_depth' in metrics
        assert 'std_depth' in metrics
        assert 'min_depth' in metrics
        assert 'max_depth' in metrics
        assert 'edge_density' in metrics
        assert 'edge_strength' in metrics
    
    def test_quality_metrics_with_reference(self):
        """Test quality metrics with reference depth."""
        depth_map = torch.randn(50, 50)
        reference_depth = depth_map + torch.randn(50, 50) * 0.1  # Add noise
        
        metrics = compute_depth_quality_metrics(depth_map, reference_depth)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'relative_error' in metrics
        
        # MSE should be positive
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_quality_metrics_batch(self):
        """Test quality metrics with batch dimension."""
        depth_batch = torch.randn(2, 1, 64, 64)  # Batch of 2
        
        metrics = compute_depth_quality_metrics(depth_batch)
        
        # Should handle batch dimension correctly
        assert all(key in metrics for key in ['mean_depth', 'std_depth', 'edge_density'])


# Integration tests
class TestIntegration:
    """Integration tests for the complete depth estimation pipeline."""
    
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MARIGOLD_AVAILABLE', True)
    @patch('video_ad_placement.depth_estimation.marigold_estimator.MarigoldDepthPipeline')
    def test_end_to_end_processing(self, mock_pipeline):
        """Test end-to-end depth estimation processing."""
        # Mock pipeline output
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [np.random.randn(100, 100) for _ in range(2)]
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        
        # Configuration
        config = {
            'device': 'cpu',
            'quality_mode': 'fast',
            'batch_size': 2,
            'enable_temporal_consistency': True,
            'temporal_smoothing_method': 'exponential'
        }
        
        # Initialize estimator
        estimator = MarigoldDepthEstimator(config)
        
        # Test single frame processing
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = estimator.estimate_depth(test_frame)
        
        assert isinstance(depth_map, torch.Tensor)
        assert depth_map.ndim >= 2
        
        # Test metrics
        metrics = estimator.get_metrics()
        assert isinstance(metrics, DepthEstimationMetrics)
        assert metrics.frames_processed > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 