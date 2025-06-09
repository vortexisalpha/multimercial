"""
Depth Estimation Package

Advanced depth estimation capabilities using Marigold diffusion model
with temporal consistency and optimization features.
"""

from .marigold_estimator import (
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

from .temporal_consistency import (
    TemporalConsistencyProcessor,
    OpticalFlowWarping,
    TemporalFilter
)

from .utils import (
    DepthVisualization,
    VideoDepthProcessor,
    DepthMetricsCollector
)

# Import ensemble depth estimation components
from .ensemble_estimator import (
    EnsembleDepthEstimator,
    EnsembleConfig, 
    EnsembleMetrics,
    FusionMethod,
    SceneComplexity,
    SceneComplexityAnalyzer,
    LearnedFusionNetwork,
    QualityAssessmentModule
)

# Import benchmarking components
from .benchmarking import (
    DepthEstimationBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkMode,
    HardwareProfile,
    HardwareMonitor,
    AccuracyValidator,
    DatasetLoader
)

from .depth_pro_estimator import (
    DepthProEstimator,
    DepthProConfig,
    DepthProMetrics,
    DepthProPrecision,
    DepthProOptimization
)

# Main classes
'MarigoldDepthEstimator',
'DepthEstimationConfig',
'DepthEstimationMetrics',

# Create alias for backwards compatibility
DepthEstimator = MarigoldDepthEstimator

__all__ = [
    # Main classes
    'DepthEstimator',
    'MarigoldDepthEstimator',
    'DepthProEstimator',
    'DepthEstimationConfig',
    'DepthEstimationMetrics',
    
    # Enums
    'PrecisionMode',
    'QualityMode', 
    'TemporalSmoothingMethod',
    
    # Utilities
    'GPUMemoryManager',
    'OpticalFlowProcessor',
    'TemporalConsistencyProcessor',
    'OpticalFlowWarping',
    'TemporalFilter',
    'DepthVisualization',
    'VideoDepthProcessor',
    'DepthMetricsCollector',
    
    # Functions
    'compute_depth_quality_metrics',
    
    # New ensemble exports
    "EnsembleDepthEstimator",
    "EnsembleConfig",
    "EnsembleMetrics", 
    "FusionMethod",
    "SceneComplexity",
    "SceneComplexityAnalyzer",
    "LearnedFusionNetwork",
    "QualityAssessmentModule",
    
    # New benchmarking exports
    "DepthEstimationBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkMode", 
    "HardwareProfile",
    "HardwareMonitor",
    "AccuracyValidator",
    "DatasetLoader"
]

__version__ = "1.0.0" 