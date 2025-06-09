"""
Configuration Management System

This module provides comprehensive configuration management with Hydra support,
environment-specific overrides, validation, and secrets management.
"""

from .config_manager import ConfigManager, ConfigVersion
from .hydra_config import (
    VideoProcessingConfig,
    APIConfig,
    DatabaseConfig,
    CloudConfig,
    MonitoringConfig,
    SecurityConfig,
    AppConfig,
    Environment
)
from .validators import ConfigValidator
from .secrets_manager import SecretsManager
from .environment import EnvironmentManager

__all__ = [
    'ConfigManager',
    'ConfigVersion',
    'VideoProcessingConfig',
    'APIConfig',
    'DatabaseConfig',
    'CloudConfig',
    'MonitoringConfig',
    'SecurityConfig',
    'AppConfig',
    'Environment',
    'ConfigValidator',
    'SecretsManager',
    'EnvironmentManager'
] 