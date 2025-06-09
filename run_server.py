#!/usr/bin/env python3
"""
Development Server Runner

Simple script to run the Video Ad Placement API server for testing and development.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.video_ad_placement.config import ConfigManager, EnvironmentManager
from src.video_ad_placement.api.main import create_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    try:
        # Initialize configuration
        logger.info("Initializing configuration...")
        
        # Create a minimal configuration for testing
        from src.video_ad_placement.config.hydra_config import (
            AppConfig, Environment, VideoProcessingConfig, APIConfig, 
            DatabaseConfig, CloudConfig, SecurityConfig, MonitoringConfig, 
            PaymentConfig, QualityLevel, ProcessingMode, DatabaseType, 
            CloudProvider, LogLevel
        )
        
        # Create minimal config
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            name="Video Ad Placement Service",
            version="1.0.0",
            description="AI-powered video advertisement placement service",
            video_processing=VideoProcessingConfig(
                quality_level=QualityLevel.HIGH,
                processing_mode=ProcessingMode.BATCH,
                max_workers=2,
                use_gpu=False,
                max_memory_usage=2048.0,
                batch_size=8
            ),
            api=APIConfig(
                host="0.0.0.0",
                port=8000,
                workers=1,
                reload=True,
                debug=True,
                title="Video Ad Placement API",
                description="Advanced video advertisement placement service",
                version="1.0.0"
            ),
            database=DatabaseConfig(
                type=DatabaseType.SQLITE,
                name="video_ad_placement_dev"
            ),
            cloud=CloudConfig(
                provider=CloudProvider.LOCAL
            ),
            security=SecurityConfig(
                jwt_secret_key="dev_secret_key_change_in_production"
            ),
            monitoring=MonitoringConfig(
                log_level=LogLevel.DEBUG
            ),
            payment=PaymentConfig(),
            max_video_size=1073741824,  # 1GB
            max_processing_time=1800,   # 30 minutes
            cleanup_interval=300        # 5 minutes
        )
        
        logger.info(f"Configuration loaded for environment: {config.environment}")
        
        # Create FastAPI app
        logger.info("Creating FastAPI application...")
        app = create_app(config)
        
        # Run the server
        import uvicorn
        
        logger.info(f"Starting server on {config.api.host}:{config.api.port}")
        logger.info(f"API documentation available at: http://{config.api.host}:{config.api.port}/docs")
        
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            reload=False,  # Disable reload for now
            workers=1,  # Use 1 worker for development
            log_level="info" if config.debug else "warning"
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run the server
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 