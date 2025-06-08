"""
Video Advertisement Placement System

Advanced AI-powered system for intelligent video advertisement placement using
state-of-the-art depth estimation and object detection technologies.

Features:
- Depth estimation with Marigold and Depth Pro models
- YOLOv9-based object detection for advertisement placement
- Real-time inference optimization with TensorRT
- Temporal consistency and tracking integration
- Comprehensive benchmarking and evaluation
- Custom training pipelines for video content
"""

# Depth estimation components
from .depth_estimation import (
    # Main classes
    MarigoldDepthEstimator,
    DepthEstimationConfig,
    DepthEstimationMetrics,
    
    # Enums
    PrecisionMode,
    QualityMode,
    TemporalSmoothingMethod,
    
    # Utilities
    GPUMemoryManager,
    OpticalFlowProcessor,
    TemporalConsistencyProcessor,
    OpticalFlowWarping,
    TemporalFilter,
    DepthVisualization,
    VideoDepthProcessor,
    DepthMetricsCollector,
    
    # Ensemble depth estimation
    EnsembleDepthEstimator,
    EnsembleConfig,
    EnsembleMetrics,
    FusionMethod,
    SceneComplexity,
    SceneComplexityAnalyzer,
    LearnedFusionNetwork,
    QualityAssessmentModule,
    
    # Benchmarking
    DepthEstimationBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkMode,
    HardwareProfile,
    HardwareMonitor,
    AccuracyValidator,
    DatasetLoader,
    
    # Functions
    compute_depth_quality_metrics
)

# Object detection components
try:
    from .object_detection import (
        # Main classes
        YOLOv9Detector,
        YOLOv9Config,
        
        # Data models
        Detection,
        DetectionBatch,
        DetectionClass,
        SceneContext,
        InferenceMetrics,
        TrainingConfig,
        
        # Training components
        YOLOv9Trainer,
        TrainingMetrics,
        VideoAdDataset,
        YouTubeDatasetCreator,
        DatasetConfig,
        VideoAugmentationPipeline,
        AugmentationConfig,
        
        # Optimization components
        HyperparameterOptimizer,
        OptimizationConfig,
        TensorRTOptimizer,
        ModelQuantizer,
        MemoryManager,
        
        # Evaluation components
        ObjectDetectionMetrics,
        ValidationFramework,
        ObjectDetectionBenchmark as ODBenchmark,
        
        # Utils and integration
        VideoProcessor,
        NMSOptimizer,
        TrackerInterface,
        MLflowIntegration,
        ABTestingFramework
    )
    
    OBJECT_DETECTION_AVAILABLE = True
    
except ImportError:
    # Object detection components not available
    OBJECT_DETECTION_AVAILABLE = False

__all__ = [
    # Depth estimation exports
    'MarigoldDepthEstimator',
    'DepthEstimationConfig',
    'DepthEstimationMetrics',
    'PrecisionMode',
    'QualityMode',
    'TemporalSmoothingMethod',
    'GPUMemoryManager',
    'OpticalFlowProcessor',
    'TemporalConsistencyProcessor',
    'OpticalFlowWarping',
    'TemporalFilter',
    'DepthVisualization',
    'VideoDepthProcessor',
    'DepthMetricsCollector',
    'compute_depth_quality_metrics',
    
    # Ensemble depth estimation
    'EnsembleDepthEstimator',
    'EnsembleConfig',
    'EnsembleMetrics',
    'FusionMethod',
    'SceneComplexity',
    'SceneComplexityAnalyzer',
    'LearnedFusionNetwork',
    'QualityAssessmentModule',
    
    # Depth benchmarking
    'DepthEstimationBenchmark',
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkMode',
    'HardwareProfile',
    'HardwareMonitor',
    'AccuracyValidator',
    'DatasetLoader',
]

# Add object detection exports if available
if OBJECT_DETECTION_AVAILABLE:
    __all__.extend([
        # Core object detection
        'YOLOv9Detector',
        'YOLOv9Config',
        'Detection',
        'DetectionBatch',
        'DetectionClass',
        'SceneContext',
        'InferenceMetrics',
        'TrainingConfig',
        
        # Training
        'YOLOv9Trainer',
        'TrainingMetrics',
        'VideoAdDataset',
        'YouTubeDatasetCreator',
        'DatasetConfig',
        'VideoAugmentationPipeline',
        'AugmentationConfig',
        
        # Optimization
        'HyperparameterOptimizer',
        'OptimizationConfig',
        'TensorRTOptimizer',
        'ModelQuantizer',
        'MemoryManager',
        
        # Evaluation
        'ObjectDetectionMetrics',
        'ValidationFramework',
        'ODBenchmark',
        
        # Utils and integration
        'VideoProcessor',
        'NMSOptimizer',
        'TrackerInterface',
        'MLflowIntegration',
        'ABTestingFramework'
    ])

__version__ = "1.0.0"
__author__ = "Video Ad Placement Team"
__email__ = "contact@video-ad-placement.ai"
__description__ = "Advanced AI-powered video advertisement placement system"

# Module information
MODULES = {
    'depth_estimation': {
        'available': True,
        'description': 'Advanced depth estimation with Marigold and Depth Pro models',
        'components': [
            'MarigoldDepthEstimator',
            'EnsembleDepthEstimator', 
            'DepthEstimationBenchmark'
        ]
    },
    'object_detection': {
        'available': OBJECT_DETECTION_AVAILABLE,
        'description': 'YOLOv9-based object detection for video advertisement placement',
        'components': [
            'YOLOv9Detector',
            'YOLOv9Trainer',
            'ObjectDetectionBenchmark'
        ] if OBJECT_DETECTION_AVAILABLE else []
    }
}

def get_module_info(module_name: str = None):
    """Get information about available modules."""
    if module_name:
        return MODULES.get(module_name, {})
    return MODULES

def check_dependencies():
    """Check if all required dependencies are available."""
    import importlib
    
    required_packages = [
        'torch',
        'torchvision', 
        'transformers',
        'numpy',
        'opencv-python',
        'PIL',
        'scipy',
        'matplotlib'
    ]
    
    optional_packages = [
        'ultralytics',  # For YOLOv9
        'tensorrt',     # For TensorRT optimization
        'mlflow',       # For experiment tracking
        'optuna'        # For hyperparameter optimization
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    return {
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'all_dependencies_met': len(missing_required) == 0
    }

# Package initialization message
import logging
logger = logging.getLogger(__name__)

def _log_initialization():
    """Log package initialization information."""
    logger.info(f"Video Advertisement Placement System v{__version__} initialized")
    
    # Log available modules
    for module, info in MODULES.items():
        status = "✓ Available" if info['available'] else "✗ Not Available"
        logger.info(f"  {module}: {status}")
    
    # Check dependencies
    dep_check = check_dependencies()
    if not dep_check['all_dependencies_met']:
        logger.warning(f"Missing required dependencies: {dep_check['missing_required']}")
    
    if dep_check['missing_optional']:
        logger.info(f"Optional dependencies not available: {dep_check['missing_optional']}")

# Initialize on import
try:
    _log_initialization()
except Exception:
    # Silent fail for initialization logging
    pass 