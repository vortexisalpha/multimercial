"""
Environment Manager

This module provides environment-specific configuration management
and deployment settings for different environments.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .hydra_config import Environment, AppConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Information about the current environment."""
    name: Environment
    is_production: bool
    is_development: bool
    is_staging: bool
    is_testing: bool
    config_overrides: Dict[str, Any]
    required_secrets: List[str]
    optional_features: List[str]


class EnvironmentManager:
    """Manages environment-specific configuration and settings."""
    
    def __init__(self):
        self.current_environment = self._detect_environment()
        self.environment_configs = self._load_environment_configs()
        
        logger.info(f"Environment detected: {self.current_environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from various sources."""
        
        # Check environment variable first
        env_var = os.getenv("VADP_ENVIRONMENT", "").lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                logger.warning(f"Invalid environment value: {env_var}")
        
        # Check for common environment indicators
        if os.getenv("PRODUCTION") or os.getenv("PROD"):
            return Environment.PRODUCTION
        
        if os.getenv("STAGING") or os.getenv("STAGE"):
            return Environment.STAGING
        
        if os.getenv("TESTING") or os.getenv("TEST"):
            return Environment.TESTING
        
        # Check for development indicators
        if (os.getenv("DEBUG") or 
            os.getenv("DEVELOPMENT") or 
            os.getenv("DEV") or
            os.path.exists(".git")):  # Git repository indicates development
            return Environment.DEVELOPMENT
        
        # Default to development
        logger.info("No environment indicators found, defaulting to development")
        return Environment.DEVELOPMENT
    
    def _load_environment_configs(self) -> Dict[Environment, EnvironmentInfo]:
        """Load configuration for all environments."""
        configs = {}
        
        # Development environment
        configs[Environment.DEVELOPMENT] = EnvironmentInfo(
            name=Environment.DEVELOPMENT,
            is_production=False,
            is_development=True,
            is_staging=False,
            is_testing=False,
            config_overrides={
                "debug": True,
                "api.reload": True,
                "api.workers": 1,
                "monitoring.log_level": "DEBUG",
                "security.ssl_enabled": False,
                "video_processing.enable_profiling": True,
                "video_processing.save_debug_data": True,
                "cloud.provider": "local",
                "database.echo_sql": True
            },
            required_secrets=[],
            optional_features=[
                "hot_reload",
                "debug_endpoints",
                "detailed_logging",
                "development_tools"
            ]
        )
        
        # Staging environment
        configs[Environment.STAGING] = EnvironmentInfo(
            name=Environment.STAGING,
            is_production=False,
            is_development=False,
            is_staging=True,
            is_testing=False,
            config_overrides={
                "debug": False,
                "api.reload": False,
                "api.workers": 2,
                "monitoring.log_level": "INFO",
                "security.ssl_enabled": True,
                "video_processing.enable_profiling": True,
                "video_processing.quality_level": "medium",
                "cloud.enable_distributed": False,
                "database.echo_sql": False
            },
            required_secrets=[
                "jwt_secret_key",
                "database_password"
            ],
            optional_features=[
                "monitoring",
                "performance_testing",
                "integration_testing"
            ]
        )
        
        # Production environment
        configs[Environment.PRODUCTION] = EnvironmentInfo(
            name=Environment.PRODUCTION,
            is_production=True,
            is_development=False,
            is_staging=False,
            is_testing=False,
            config_overrides={
                "debug": False,
                "api.reload": False,
                "api.workers": 4,
                "monitoring.log_level": "INFO",
                "security.ssl_enabled": True,
                "video_processing.enable_profiling": False,
                "video_processing.save_debug_data": False,
                "video_processing.quality_level": "high",
                "cloud.enable_distributed": True,
                "cloud.auto_scaling": True,
                "database.echo_sql": False,
                "monitoring.prometheus_enabled": True,
                "monitoring.alerting_enabled": True
            },
            required_secrets=[
                "jwt_secret_key",
                "database_password",
                "redis_password",
                "cloud_access_key",
                "cloud_secret_key"
            ],
            optional_features=[
                "monitoring",
                "alerting",
                "auto_scaling",
                "distributed_processing",
                "advanced_analytics"
            ]
        )
        
        # Testing environment
        configs[Environment.TESTING] = EnvironmentInfo(
            name=Environment.TESTING,
            is_production=False,
            is_development=False,
            is_staging=False,
            is_testing=True,
            config_overrides={
                "debug": False,
                "testing": True,
                "api.workers": 1,
                "monitoring.log_level": "WARNING",
                "security.ssl_enabled": False,
                "video_processing.batch_size": 4,
                "video_processing.max_workers": 1,
                "database.name": "video_ad_placement_test",
                "database.min_connections": 1,
                "database.max_connections": 5,
                "cloud.provider": "local",
                "cleanup_interval": 10  # Faster cleanup for tests
            },
            required_secrets=[],
            optional_features=[
                "test_fixtures",
                "mock_services",
                "test_database"
            ]
        )
        
        return configs
    
    def get_current_environment(self) -> Environment:
        """Get the current environment."""
        return self.current_environment
    
    def get_environment_info(self, environment: Optional[Environment] = None) -> EnvironmentInfo:
        """Get information about an environment."""
        env = environment or self.current_environment
        return self.environment_configs[env]
    
    def get_config_overrides(self, environment: Optional[Environment] = None) -> Dict[str, Any]:
        """Get configuration overrides for an environment."""
        env = environment or self.current_environment
        return self.environment_configs[env].config_overrides.copy()
    
    def get_required_secrets(self, environment: Optional[Environment] = None) -> List[str]:
        """Get required secrets for an environment."""
        env = environment or self.current_environment
        return self.environment_configs[env].required_secrets.copy()
    
    def validate_environment_setup(self, environment: Optional[Environment] = None) -> Dict[str, Any]:
        """Validate that the environment is properly set up."""
        env = environment or self.current_environment
        env_info = self.environment_configs[env]
        
        validation_result = {
            "environment": env.value,
            "is_valid": True,
            "missing_secrets": [],
            "missing_features": [],
            "warnings": [],
            "errors": []
        }
        
        # Check required secrets
        for secret_name in env_info.required_secrets:
            # Check if secret exists in environment variables
            env_var_name = f"VADP_{secret_name.upper()}"
            if not os.getenv(env_var_name):
                validation_result["missing_secrets"].append(secret_name)
                validation_result["is_valid"] = False
        
        # Environment-specific validations
        if env == Environment.PRODUCTION:
            # Production-specific checks
            if os.getenv("DEBUG", "").lower() in ("true", "1", "yes"):
                validation_result["errors"].append("Debug mode enabled in production")
                validation_result["is_valid"] = False
            
            if not os.getenv("VADP_JWT_SECRET_KEY"):
                validation_result["errors"].append("JWT secret key not configured")
                validation_result["is_valid"] = False
            
            # Check for HTTPS configuration
            if not os.getenv("VADP_SSL_CERT_PATH"):
                validation_result["warnings"].append("SSL certificate not configured")
        
        elif env == Environment.DEVELOPMENT:
            # Development-specific checks
            if not os.path.exists(".git"):
                validation_result["warnings"].append("Not in a Git repository")
        
        elif env == Environment.TESTING:
            # Testing-specific checks
            test_db_name = os.getenv("VADP_DATABASE_NAME", "")
            if test_db_name and not test_db_name.endswith("_test"):
                validation_result["warnings"].append("Test database name should end with '_test'")
        
        # Check for optional features
        for feature in env_info.optional_features:
            feature_enabled = os.getenv(f"VADP_FEATURE_{feature.upper()}", "").lower()
            if feature_enabled not in ("true", "1", "yes", "enabled"):
                validation_result["missing_features"].append(feature)
        
        return validation_result
    
    def apply_environment_config(self, config: AppConfig, 
                                environment: Optional[Environment] = None) -> AppConfig:
        """Apply environment-specific configuration overrides."""
        env = environment or self.current_environment
        overrides = self.get_config_overrides(env)
        
        # Apply overrides to config
        for key, value in overrides.items():
            self._apply_config_override(config, key, value)
        
        # Set environment in config
        config.environment = env
        
        logger.info(f"Applied {len(overrides)} environment overrides for {env.value}")
        return config
    
    def _apply_config_override(self, config: AppConfig, key: str, value: Any):
        """Apply a single configuration override."""
        try:
            # Handle nested keys like "api.workers"
            if "." in key:
                parts = key.split(".")
                obj = config
                
                # Navigate to the parent object
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                
                # Set the final value
                setattr(obj, parts[-1], value)
            else:
                # Direct attribute
                setattr(config, key, value)
                
        except AttributeError as e:
            logger.warning(f"Failed to apply config override {key}={value}: {e}")
    
    def get_deployment_requirements(self, environment: Optional[Environment] = None) -> Dict[str, Any]:
        """Get deployment requirements for an environment."""
        env = environment or self.current_environment
        env_info = self.environment_configs[env]
        
        requirements = {
            "environment": env.value,
            "min_workers": 1,
            "max_workers": 1,
            "min_memory_mb": 1024,
            "min_disk_gb": 10,
            "requires_ssl": False,
            "requires_database": True,
            "requires_redis": False,
            "requires_gpu": False,
            "health_check_endpoint": "/api/v1/health",
            "metrics_endpoint": "/api/v1/metrics"
        }
        
        if env == Environment.PRODUCTION:
            requirements.update({
                "min_workers": 4,
                "max_workers": 16,
                "min_memory_mb": 8192,
                "min_disk_gb": 100,
                "requires_ssl": True,
                "requires_redis": True,
                "requires_gpu": True,
                "load_balancer": True,
                "auto_scaling": True,
                "backup_strategy": "daily",
                "monitoring_required": True
            })
        
        elif env == Environment.STAGING:
            requirements.update({
                "min_workers": 2,
                "max_workers": 4,
                "min_memory_mb": 4096,
                "min_disk_gb": 50,
                "requires_ssl": True,
                "requires_redis": True,
                "monitoring_required": True
            })
        
        elif env == Environment.DEVELOPMENT:
            requirements.update({
                "min_workers": 1,
                "max_workers": 2,
                "min_memory_mb": 2048,
                "min_disk_gb": 20,
                "hot_reload": True,
                "debug_tools": True
            })
        
        elif env == Environment.TESTING:
            requirements.update({
                "min_workers": 1,
                "max_workers": 1,
                "min_memory_mb": 1024,
                "min_disk_gb": 10,
                "test_database": True,
                "fast_startup": True
            })
        
        return requirements
    
    def switch_environment(self, new_environment: Environment) -> bool:
        """Switch to a different environment."""
        try:
            # Validate new environment
            validation = self.validate_environment_setup(new_environment)
            
            if not validation["is_valid"]:
                logger.error(f"Cannot switch to {new_environment.value}: {validation['errors']}")
                return False
            
            # Update current environment
            old_env = self.current_environment
            self.current_environment = new_environment
            
            # Update environment variable
            os.environ["VADP_ENVIRONMENT"] = new_environment.value
            
            logger.info(f"Switched environment from {old_env.value} to {new_environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch environment: {e}")
            return False
    
    def get_environment_variables(self, environment: Optional[Environment] = None) -> Dict[str, str]:
        """Get recommended environment variables for an environment."""
        env = environment or self.current_environment
        env_info = self.environment_configs[env]
        
        env_vars = {
            "VADP_ENVIRONMENT": env.value
        }
        
        # Add environment-specific variables
        if env == Environment.PRODUCTION:
            env_vars.update({
                "VADP_DEBUG": "false",
                "VADP_API_WORKERS": "4",
                "VADP_SSL_ENABLED": "true",
                "VADP_MONITORING_ENABLED": "true",
                "VADP_CLOUD_DISTRIBUTED": "true"
            })
        
        elif env == Environment.DEVELOPMENT:
            env_vars.update({
                "VADP_DEBUG": "true",
                "VADP_API_WORKERS": "1",
                "VADP_API_RELOAD": "true",
                "VADP_LOG_LEVEL": "DEBUG"
            })
        
        elif env == Environment.TESTING:
            env_vars.update({
                "VADP_TESTING": "true",
                "VADP_DATABASE_NAME": "video_ad_placement_test",
                "VADP_LOG_LEVEL": "WARNING"
            })
        
        return env_vars 