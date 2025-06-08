"""
Training Module for YOLOv9 Object Detection

This module provides comprehensive training capabilities including custom datasets,
augmentation pipelines, hyperparameter optimization, and fine-tuning for video
advertisement placement scenarios.
"""

from .trainer import YOLOv9Trainer
from .dataset import VideoAdDataset, YouTubeDatasetCreator, DatasetConfig
from .augmentation import VideoAugmentationPipeline, AugmentationConfig
from .hyperparameter_tuning import HyperparameterOptimizer, OptimizationConfig

__all__ = [
    'YOLOv9Trainer',
    'VideoAdDataset',
    'YouTubeDatasetCreator', 
    'DatasetConfig',
    'VideoAugmentationPipeline',
    'AugmentationConfig',
    'HyperparameterOptimizer',
    'OptimizationConfig'
] 