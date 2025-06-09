"""
Configuration Validators

This module provides comprehensive validation for configuration settings
with type checking, range validation, and dependency validation.
"""

import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import ipaddress

from .hydra_config import AppConfig, Environment, QualityLevel, ProcessingMode


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class ConfigValidator:
    """Comprehensive configuration validator."""
    
    def __init__(self):
        self.validation_rules = {
            # API configuration rules
            'api.port': self._validate_port,
            'api.workers': self._validate_positive_int,
            'api.max_request_size': self._validate_positive_int,
            'api.request_timeout': self._validate_positive_float,
            'api.max_concurrent_requests': self._validate_positive_int,
            'api.rate_limit_requests': self._validate_positive_int,
            'api.rate_limit_window': self._validate_positive_int,
            'api.max_websocket_connections': self._validate_positive_int,
            
            # Database configuration rules
            'database.port': self._validate_port,
            'database.min_connections': self._validate_positive_int,
            'database.max_connections': self._validate_positive_int,
            'database.connection_timeout': self._validate_positive_float,
            'database.redis_port': self._validate_port,
            'database.redis_db': self._validate_non_negative_int,
            'database.redis_ttl': self._validate_positive_int,
            
            # Video processing rules
            'video_processing.max_workers': self._validate_positive_int,
            'video_processing.batch_size': self._validate_positive_int,
            'video_processing.prefetch_frames': self._validate_non_negative_int,
            'video_processing.cache_size': self._validate_positive_int,
            'video_processing.max_retries': self._validate_non_negative_int,
            'video_processing.retry_delay': self._validate_positive_float,
            'video_processing.checkpoint_interval': self._validate_positive_int,
            'video_processing.max_memory_usage': self._validate_positive_float,
            'video_processing.min_wall_area': self._validate_positive_float,
            'video_processing.max_wall_distance': self._validate_positive_float,
            'video_processing.tv_width': self._validate_positive_float,
            'video_processing.tv_height': self._validate_positive_float,
            'video_processing.metrics_interval': self._validate_positive_float,
            'video_processing.health_check_interval': self._validate_positive_float,
            
            # Security configuration rules
            'security.jwt_expiration': self._validate_positive_int,
            'security.jwt_refresh_expiration': self._validate_positive_int,
            'security.api_key_length': self._validate_positive_int,
            'security.hsts_max_age': self._validate_positive_int,
            'security.min_password_length': self._validate_positive_int,
            'security.password_hash_rounds': self._validate_positive_int,
            'security.session_timeout': self._validate_positive_int,
            'security.max_sessions_per_user': self._validate_positive_int,
            
            # Cloud configuration rules
            'cloud.retention_days': self._validate_positive_int,
            'cloud.max_workers': self._validate_positive_int,
            
            # Monitoring configuration rules
            'monitoring.metrics_port': self._validate_port,
            'monitoring.health_check_interval': self._validate_positive_float,
            'monitoring.health_check_timeout': self._validate_positive_float,
            'monitoring.trace_sample_rate': self._validate_sample_rate,
            
            # Payment configuration rules
            'payment.price_per_minute': self._validate_positive_float,
            'payment.free_tier_minutes': self._validate_non_negative_int,
            'payment.premium_tier_multiplier': self._validate_positive_float,
            'payment.tax_rate': self._validate_tax_rate,
            
            # Application rules
            'max_video_size': self._validate_positive_int,
            'max_processing_time': self._validate_positive_int,
            'cleanup_interval': self._validate_positive_int,
        }
    
    def validate_config(self, config: AppConfig) -> ValidationResult:
        """Validate complete configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        try:
            # Basic structure validation
            self._validate_structure(config, result)
            
            # Field-specific validation
            self._validate_fields(config, result)
            
            # Cross-field validation
            self._validate_dependencies(config, result)
            
            # Environment-specific validation
            self._validate_environment_settings(config, result)
            
            # Security validation
            self._validate_security_settings(config, result)
            
            # Performance validation
            self._validate_performance_settings(config, result)
            
        except Exception as e:
            result.add_error(f"Validation failed with exception: {str(e)}")
        
        return result
    
    def _validate_structure(self, config: AppConfig, result: ValidationResult):
        """Validate configuration structure."""
        required_sections = [
            'video_processing', 'api', 'database', 'cloud', 
            'security', 'monitoring', 'payment'
        ]
        
        for section in required_sections:
            if not hasattr(config, section):
                result.add_error(f"Missing required configuration section: {section}")
    
    def _validate_fields(self, config: AppConfig, result: ValidationResult):
        """Validate individual configuration fields."""
        config_dict = self._config_to_dict(config)
        
        for field_path, validator in self.validation_rules.items():
            value = self._get_nested_value(config_dict, field_path)
            if value is not None:
                error = validator(value, field_path)
                if error:
                    result.add_error(error)
    
    def _validate_dependencies(self, config: AppConfig, result: ValidationResult):
        """Validate configuration dependencies."""
        
        # Database connection validation
        if config.database.max_connections <= config.database.min_connections:
            result.add_error("database.max_connections must be greater than min_connections")
        
        # API workers validation
        if config.api.workers > config.video_processing.max_workers * 2:
            result.add_warning("High API worker count may impact video processing performance")
        
        # Memory validation
        if config.video_processing.batch_size * 100 > config.video_processing.max_memory_usage:
            result.add_warning("Batch size may exceed memory limits")
        
        # GPU validation
        if config.video_processing.use_gpu and not config.video_processing.gpu_devices:
            result.add_error("GPU enabled but no GPU devices specified")
        
        # Rate limiting validation
        if config.api.rate_limit_enabled and config.api.rate_limit_storage == "redis":
            if not config.database.redis_host:
                result.add_error("Redis rate limiting enabled but no Redis host configured")
        
        # Cloud validation
        if config.cloud.enable_distributed and config.cloud.provider.value == "local":
            result.add_error("Distributed processing not supported with local cloud provider")
        
        # SSL validation
        if config.security.ssl_enabled:
            if not config.security.ssl_cert_path or not config.security.ssl_key_path:
                result.add_error("SSL enabled but certificate/key paths not specified")
        
        # OAuth validation
        if config.security.oauth_enabled:
            if not config.security.oauth_client_id or not config.security.oauth_client_secret:
                result.add_error("OAuth enabled but client credentials not specified")
    
    def _validate_environment_settings(self, config: AppConfig, result: ValidationResult):
        """Validate environment-specific settings."""
        
        if config.environment == Environment.PRODUCTION:
            # Production requirements
            if config.debug:
                result.add_error("Debug mode should be disabled in production")
            
            if not config.security.ssl_enabled:
                result.add_warning("SSL should be enabled in production")
            
            if config.monitoring.log_level.value == "DEBUG":
                result.add_warning("Debug logging not recommended in production")
            
            if config.api.reload:
                result.add_error("API reload should be disabled in production")
        
        elif config.environment == Environment.DEVELOPMENT:
            # Development warnings
            if config.security.ssl_enabled:
                result.add_warning("SSL typically not needed in development")
        
        elif config.environment == Environment.TESTING:
            # Testing requirements
            if not config.database.name.endswith("_test"):
                result.add_warning("Test database should have '_test' suffix")
    
    def _validate_security_settings(self, config: AppConfig, result: ValidationResult):
        """Validate security configuration."""
        
        # JWT secret validation
        if config.security.jwt_secret_key:
            if len(config.security.jwt_secret_key) < 32:
                result.add_error("JWT secret key should be at least 32 characters")
        
        # Password policy validation
        if config.security.min_password_length < 8:
            result.add_warning("Minimum password length should be at least 8 characters")
        
        # Session timeout validation
        if config.security.session_timeout > 86400:  # 24 hours
            result.add_warning("Session timeout longer than 24 hours may be a security risk")
        
        # API key validation
        if config.security.api_key_length < 16:
            result.add_warning("API key length should be at least 16 characters")
    
    def _validate_performance_settings(self, config: AppConfig, result: ValidationResult):
        """Validate performance-related settings."""
        
        # Memory usage validation
        total_memory_estimate = (
            config.video_processing.max_memory_usage +
            config.video_processing.cache_size * 10  # Rough estimate
        )
        
        if total_memory_estimate > 32768:  # 32GB
            result.add_warning("High memory usage configuration may cause system instability")
        
        # Worker validation
        if config.api.workers > 16:
            result.add_warning("High number of API workers may impact performance")
        
        if config.video_processing.max_workers > 32:
            result.add_warning("High number of processing workers may cause resource contention")
        
        # Batch size validation
        if config.video_processing.batch_size > 128:
            result.add_warning("Large batch size may cause memory issues")
        
        # Rate limiting validation
        if config.api.rate_limit_requests > 10000:
            result.add_warning("Very high rate limit may not provide protection")
    
    def _validate_port(self, value: int, field: str) -> Optional[str]:
        """Validate port number."""
        if not isinstance(value, int):
            return f"{field}: Port must be an integer"
        if not (1 <= value <= 65535):
            return f"{field}: Port must be between 1 and 65535"
        if value < 1024 and value != 80 and value != 443:
            return f"{field}: Port {value} may require root privileges"
        return None
    
    def _validate_positive_int(self, value: int, field: str) -> Optional[str]:
        """Validate positive integer."""
        if not isinstance(value, int):
            return f"{field}: Must be an integer"
        if value <= 0:
            return f"{field}: Must be positive"
        return None
    
    def _validate_non_negative_int(self, value: int, field: str) -> Optional[str]:
        """Validate non-negative integer."""
        if not isinstance(value, int):
            return f"{field}: Must be an integer"
        if value < 0:
            return f"{field}: Must be non-negative"
        return None
    
    def _validate_positive_float(self, value: float, field: str) -> Optional[str]:
        """Validate positive float."""
        if not isinstance(value, (int, float)):
            return f"{field}: Must be a number"
        if value <= 0:
            return f"{field}: Must be positive"
        return None
    
    def _validate_sample_rate(self, value: float, field: str) -> Optional[str]:
        """Validate sample rate (0.0 to 1.0)."""
        if not isinstance(value, (int, float)):
            return f"{field}: Must be a number"
        if not (0.0 <= value <= 1.0):
            return f"{field}: Must be between 0.0 and 1.0"
        return None
    
    def _validate_tax_rate(self, value: float, field: str) -> Optional[str]:
        """Validate tax rate."""
        if not isinstance(value, (int, float)):
            return f"{field}: Must be a number"
        if not (0.0 <= value <= 1.0):
            return f"{field}: Tax rate must be between 0.0 and 1.0"
        return None
    
    def _validate_email(self, value: str, field: str) -> Optional[str]:
        """Validate email format."""
        if not isinstance(value, str):
            return f"{field}: Must be a string"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            return f"{field}: Invalid email format"
        return None
    
    def _validate_url(self, value: str, field: str) -> Optional[str]:
        """Validate URL format."""
        if not isinstance(value, str):
            return f"{field}: Must be a string"
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, value):
            return f"{field}: Invalid URL format"
        return None
    
    def _validate_ip_address(self, value: str, field: str) -> Optional[str]:
        """Validate IP address."""
        if not isinstance(value, str):
            return f"{field}: Must be a string"
        
        try:
            ipaddress.ip_address(value)
            return None
        except ValueError:
            return f"{field}: Invalid IP address"
    
    def _validate_path(self, value: str, field: str) -> Optional[str]:
        """Validate file path."""
        if not isinstance(value, str):
            return f"{field}: Must be a string"
        
        try:
            path = Path(value)
            if not path.parent.exists():
                return f"{field}: Parent directory does not exist"
            return None
        except Exception:
            return f"{field}: Invalid path"
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert config object to nested dictionary."""
        def convert_value(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_value(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            else:
                return obj
        
        return convert_value(config)
    
    def _get_nested_value(self, config_dict: Dict[str, Any], path: str) -> Any:
        """Get nested value from config dictionary."""
        keys = path.split('.')
        value = config_dict
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def validate_runtime_update(self, updates: Dict[str, Any]) -> ValidationResult:
        """Validate runtime configuration updates."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if updates are allowed at runtime
        readonly_fields = {
            'environment', 'database.type', 'database.host', 'database.port',
            'database.name', 'security.ssl_cert_path', 'security.ssl_key_path'
        }
        
        for field_path in updates.keys():
            if field_path in readonly_fields:
                result.add_error(f"Field {field_path} cannot be updated at runtime")
            
            # Validate field value
            if field_path in self.validation_rules:
                value = updates[field_path]
                error = self.validation_rules[field_path](value, field_path)
                if error:
                    result.add_error(error)
        
        return result 