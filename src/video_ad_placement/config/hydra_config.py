"""
Hydra Configuration Classes

This module defines the hierarchical configuration structure using Hydra and Pydantic
for comprehensive validation and type checking.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, validator, Field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class QualityLevel(str, Enum):
    """Processing quality levels."""
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class ProcessingMode(str, Enum):
    """Processing modes."""
    BATCH = "batch"
    STREAMING = "streaming"
    REALTIME = "realtime"
    DISTRIBUTED = "distributed"


class CloudProvider(str, Enum):
    """Cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class DatabaseType(str, Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


@dataclass
class VideoProcessingConfig:
    """Video processing configuration."""
    
    # Quality and performance settings
    quality_level: QualityLevel = QualityLevel.HIGH
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    max_workers: int = 4
    use_gpu: bool = True
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    
    # Memory management
    max_memory_usage: float = 4096.0  # MB
    batch_size: int = 16
    prefetch_frames: int = 8
    cache_size: int = 500
    
    # Error handling and recovery
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_interval: int = 100
    auto_recovery: bool = True
    fallback_quality: QualityLevel = QualityLevel.MEDIUM
    
    # Processing parameters
    depth_estimation_model: str = "marigold"
    object_detection_model: str = "yolov9"
    min_wall_area: float = 1.5
    max_wall_distance: float = 8.0
    tv_width: float = 1.0
    tv_height: float = 0.6
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["mp4"])
    output_qualities: List[str] = field(default_factory=lambda: ["1080p"])
    intermediate_outputs: bool = False
    save_debug_data: bool = False
    
    # Monitoring
    enable_profiling: bool = True
    metrics_interval: float = 2.0
    health_check_interval: float = 5.0


@dataclass
class APIConfig:
    """API server configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    debug: bool = False
    
    # API settings
    title: str = "Video Ad Placement API"
    description: str = "Advanced video advertisement placement service"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # Request handling
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: float = 300.0  # 5 minutes
    max_concurrent_requests: int = 100
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    rate_limit_storage: str = "memory"  # "memory" or "redis"
    
    # WebSocket settings
    websocket_enabled: bool = True
    websocket_path: str = "/ws"
    max_websocket_connections: int = 1000


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    # Primary database
    type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    name: str = "video_ad_placement"
    username: str = "postgres"
    password: str = ""
    
    # Connection settings
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    pool_recycle: int = 3600
    
    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ttl: int = 3600
    
    # Database options
    echo_sql: bool = False
    auto_migrate: bool = True
    create_tables: bool = True


@dataclass
class CloudConfig:
    """Cloud services configuration."""
    
    # Provider settings
    provider: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"
    
    # Storage settings
    storage_bucket: str = "video-processing-bucket"
    storage_prefix: str = "processing/"
    enable_encryption: bool = True
    retention_days: int = 30
    
    # Credentials (will be overridden by secrets manager)
    access_key: str = ""
    secret_key: str = ""
    service_account_key: str = ""
    
    # CDN settings
    cdn_enabled: bool = False
    cdn_domain: str = ""
    
    # Distributed processing
    enable_distributed: bool = False
    max_workers: int = 10
    worker_instance_type: str = "m5.large"
    auto_scaling: bool = True
    
    # Monitoring
    enable_cloud_monitoring: bool = True
    metrics_namespace: str = "VideoProcessing"


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    
    # JWT settings
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    jwt_refresh_expiration: int = 86400  # 24 hours
    
    # API Keys
    api_key_header: str = "X-API-Key"
    api_key_length: int = 32
    
    # OAuth settings
    oauth_enabled: bool = False
    oauth_provider: str = "google"
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    oauth_redirect_uri: str = ""
    
    # HTTPS settings
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # Security headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000
    
    # Password settings
    min_password_length: int = 8
    password_hash_rounds: int = 12
    
    # Session settings
    session_timeout: int = 1800  # 30 minutes
    max_sessions_per_user: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: str = "video_ad_placement.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Prometheus settings
    prometheus_enabled: bool = False
    prometheus_gateway: str = ""
    prometheus_job: str = "video-ad-placement"
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    
    # Alerting
    alerting_enabled: bool = False
    alert_webhook: str = ""
    alert_channels: List[str] = field(default_factory=list)
    
    # Tracing
    tracing_enabled: bool = False
    jaeger_endpoint: str = ""
    trace_sample_rate: float = 0.1


@dataclass
class PaymentConfig:
    """Payment and billing configuration."""
    
    # Payment provider
    provider: str = "stripe"
    public_key: str = ""
    private_key: str = ""
    webhook_secret: str = ""
    
    # Pricing
    price_per_minute: float = 0.10
    free_tier_minutes: int = 60
    premium_tier_multiplier: float = 2.0
    
    # Billing
    billing_cycle: str = "monthly"
    currency: str = "USD"
    tax_rate: float = 0.0
    
    # Features
    enable_subscriptions: bool = True
    enable_usage_billing: bool = True
    enable_free_tier: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    
    # Application info
    name: str = "Video Ad Placement Service"
    version: str = "1.0.0"
    description: str = "AI-powered video advertisement placement service"
    
    # Configuration sections
    video_processing: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    payment: PaymentConfig = field(default_factory=PaymentConfig)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "websocket_support": True,
        "batch_processing": True,
        "real_time_processing": True,
        "cloud_storage": True,
        "payment_integration": False,
        "advanced_analytics": True
    })
    
    # Runtime settings
    max_video_size: int = 1024 * 1024 * 1024  # 1GB
    max_processing_time: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Adjust settings based on environment
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.api.reload = False
            self.security.ssl_enabled = True
            self.monitoring.log_level = LogLevel.INFO
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.api.reload = True
            self.monitoring.log_level = LogLevel.DEBUG
        elif self.environment == Environment.TESTING:
            self.testing = True
            self.database.name = f"{self.database.name}_test"
            self.monitoring.log_level = LogLevel.WARNING


# Hydra configuration registration
def register_configs():
    """Register configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # Main config
    cs.store(name="config", node=AppConfig)
    
    # Environment-specific configs
    cs.store(group="environment", name="development", node={
        "environment": "development",
        "debug": True,
        "api": {"reload": True, "workers": 1},
        "monitoring": {"log_level": "DEBUG"},
        "security": {"ssl_enabled": False}
    })
    
    cs.store(group="environment", name="staging", node={
        "environment": "staging",
        "debug": False,
        "api": {"reload": False, "workers": 2},
        "monitoring": {"log_level": "INFO"},
        "security": {"ssl_enabled": True}
    })
    
    cs.store(group="environment", name="production", node={
        "environment": "production",
        "debug": False,
        "api": {"reload": False, "workers": 4},
        "monitoring": {"log_level": "INFO"},
        "security": {"ssl_enabled": True},
        "cloud": {"enable_distributed": True}
    })
    
    # Quality presets
    cs.store(group="quality", name="fast", node={
        "video_processing": {
            "quality_level": "low",
            "processing_mode": "realtime",
            "batch_size": 8,
            "max_workers": 2
        }
    })
    
    cs.store(group="quality", name="balanced", node={
        "video_processing": {
            "quality_level": "medium",
            "processing_mode": "batch",
            "batch_size": 16,
            "max_workers": 4
        }
    })
    
    cs.store(group="quality", name="premium", node={
        "video_processing": {
            "quality_level": "ultra_high",
            "processing_mode": "batch",
            "batch_size": 32,
            "max_workers": 8
        }
    })


# Register configurations on import
register_configs()


class HydraConfigManager:
    """Manager for Hydra-based configuration."""
    
    def __init__(self, config_path: str = "conf", config_name: str = "config"):
        self.config_path = config_path
        self.config_name = config_name
        self._config: Optional[DictConfig] = None
    
    @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def load_config(self, cfg: DictConfig) -> AppConfig:
        """Load configuration using Hydra."""
        self._config = cfg
        
        # Convert to structured config
        app_config = OmegaConf.to_object(cfg)
        
        # Validate configuration
        if not isinstance(app_config, AppConfig):
            app_config = AppConfig(**app_config)
        
        return app_config
    
    def get_config(self) -> Optional[DictConfig]:
        """Get current configuration."""
        return self._config
    
    def override_config(self, overrides: Dict[str, Any]) -> DictConfig:
        """Override configuration values."""
        if self._config is None:
            raise ValueError("Configuration not loaded")
        
        config_copy = OmegaConf.copy(self._config)
        
        for key, value in overrides.items():
            OmegaConf.set(config_copy, key, value)
        
        return config_copy
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration structure."""
        try:
            OmegaConf.to_object(config)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False 