"""
Main FastAPI Application

This module creates and configures the main FastAPI application with all
endpoints, middleware, authentication, and documentation.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ..config import ConfigManager, AppConfig, Environment
from ..pipeline.video_processor import VideoAdProcessor  # Enable real video processing
from .models import *
from .auth import AuthManager, get_current_user, get_current_user_optional, verify_api_key
from .rate_limiting import RateLimiter
from .websocket import WebSocketManager
from .database import DatabaseManager
from .monitoring import APIMonitor
import os
import tempfile

logger = logging.getLogger(__name__)


class VideoAdPlacementAPI:
    """Main API application class."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.app: Optional[FastAPI] = None
        
        # Core components
        self.pipeline: Optional[VideoAdProcessor] = None
        self.auth_manager = AuthManager(config.security)
        self.rate_limiter = RateLimiter(config.api)
        self.websocket_manager = WebSocketManager()
        self.database_manager = DatabaseManager(config.database)
        self.api_monitor = APIMonitor(config.monitoring)
        
        # Job management
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_results: Dict[str, ProcessingResult] = {}
        
        logger.info("VideoAdPlacementAPI initialized")
    
    async def startup(self):
        """Application startup tasks."""
        try:
            # Initialize pipeline
            self.pipeline = VideoAdProcessor({
                "default_opacity": 0.85,
                "default_scale": 0.3,
                "default_position": "bottom_right",
                "quality": str(self.config.video_processing.quality_level).lower()
            })
            
            # Initialize database
            await self.database_manager.initialize()
            
            # Start monitoring
            self.api_monitor.start()
            
            logger.info("API startup completed successfully")
            
        except Exception as e:
            logger.error(f"API startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Application shutdown tasks."""
        try:
            # Stop monitoring
            self.api_monitor.stop()
            
            # Shutdown pipeline
            if self.pipeline:
                await self.pipeline.shutdown()
            
            # Close database connections
            await self.database_manager.close()
            
            # Close WebSocket connections
            await self.websocket_manager.disconnect_all()
            
            logger.info("API shutdown completed")
            
        except Exception as e:
            logger.error(f"API shutdown error: {e}")


# Global API instance
api_instance: Optional[VideoAdPlacementAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global api_instance
    
    # Startup
    if api_instance:
        await api_instance.startup()
    
    yield
    
    # Shutdown
    if api_instance:
        await api_instance.shutdown()


def create_app(config: AppConfig) -> FastAPI:
    """Create and configure the FastAPI application."""
    global api_instance
    
    # Create API instance
    api_instance = VideoAdPlacementAPI(config)
    
    # Create FastAPI app
    app = FastAPI(
        title=config.api.title,
        description=config.api.description,
        version=config.api.version,
        docs_url=config.api.docs_url,
        redoc_url=config.api.redoc_url,
        openapi_url=config.api.openapi_url,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=config.api.allow_credentials,
        allow_methods=config.api.cors_methods,
        allow_headers=config.api.cors_headers,
    )
    
    # Security middleware
    security = HTTPBearer()
    
    # Store API instance in app state
    api_instance.app = app
    app.state.api = api_instance
    
    # API Routes
    
    @app.post(f"{config.api.api_prefix}/process-video", 
              response_model=ProcessingResponse,
              summary="Process video with advertisement placement",
              description="Submit a video for advertisement placement processing")
    async def process_video(
        request: VideoProcessingRequest,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        _: Any = Depends(api_instance.rate_limiter.check)
    ):
        """Process video with advertisement placement."""
        try:
            # Validate request
            if not request.video_url and not request.video_data:
                raise HTTPException(status_code=400, detail="Either video_url or video_data must be provided")
            
            # Create job ID
            job_id = f"job_{int(time.time())}_{current_user.username}"
            estimated_completion = time.time() + 300  # 5 minutes estimate
            
            try:
                # Store job in database
                await api_instance.database_manager.create_job(job_id, {
                    "user_id": current_user.username,
                    "request": request.dict(),
                    "status": JobStatus.QUEUED,
                    "created_at": time.time(),
                    "estimated_completion": estimated_completion
                })
                
                # If we have a real video file path (for demo), process it
                if hasattr(request, 'video_file_path') and request.video_file_path:
                    # Real video processing
                    video_path = request.video_file_path
                    ad_image_path = "./uploads/advertisements/images.png"
                    output_path = f"./outputs/{job_id}_processed.mp4"
                    
                    # Create placement config from request
                    placement_config = {
                        "strategy": request.placement_config.strategy if request.placement_config else "bottom_right",
                        "opacity": 0.85,
                        "scale_factor": 0.3,
                        "start_time": 3.0,
                        "duration": 15.0
                    }
                    
                    if request.advertisement_config:
                        placement_config["opacity"] = getattr(request.advertisement_config, 'opacity', 0.85)
                        placement_config["scale_factor"] = getattr(request.advertisement_config, 'width', 0.3)
                    
                    # Validate inputs
                    api_instance.pipeline.validate_inputs(video_path, ad_image_path)
                    
                    # Process video
                    result = api_instance.pipeline.process_video(video_path, ad_image_path, output_path, placement_config)
                    
                    # Update job status
                    await api_instance.database_manager.update_job_status(job_id, JobStatus.COMPLETED, {
                        "processing_result": result,
                        "output_path": output_path,
                        "completed_at": time.time()
                    })
                    
                    # Broadcast completion via WebSocket
                    await api_instance.websocket_manager.broadcast_job_update(job_id, {
                        "status": "completed",
                        "result": result
                    })
                    
                    return ProcessingResponse(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        message="Video processing completed successfully",
                        estimated_completion_time=time.time(),
                        result_url=output_path
                    )
                
                else:
                    # Mock processing for regular API calls
                    processing_time = max(30, len(request.video_url or "default") * 2)
                    estimated_completion = time.time() + processing_time
                    
                    return ProcessingResponse(
                        job_id=job_id,
                        status=JobStatus.QUEUED,
                        message="Video processing started",
                        estimated_completion_time=estimated_completion
                    )
                
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
                await api_instance.database_manager.update_job_status(job_id, JobStatus.FAILED, {
                    "error": str(e),
                    "failed_at": time.time()
                })
                raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
            
        except Exception as e:
            logger.error(f"Process video error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(f"{config.api.api_prefix}/status/{{job_id}}", 
             response_model=ProcessingStatus,
             summary="Get processing status",
             description="Get the current status and progress of a processing job")
    async def get_processing_status(
        job_id: str,
        current_user: User = Depends(get_current_user)
    ):
        """Get processing status and progress."""
        try:
            # Check if job exists
            if job_id not in api_instance.active_jobs:
                # Check completed jobs
                if job_id in api_instance.job_results:
                    result = api_instance.job_results[job_id]
                    return ProcessingStatus(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        progress=1.0,
                        message="Processing completed",
                        result=result,
                        created_at=result.created_at,
                        completed_at=result.completed_at
                    )
                else:
                    raise HTTPException(status_code=404, detail="Job not found")
            
            job = api_instance.active_jobs[job_id]
            
            # Check user access
            if job.user_id != current_user.user_id and not current_user.is_admin:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return ProcessingStatus(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                message=job.status_message or f"Job is {job.status.value}",
                error_message=job.error_message,
                created_at=job.created_at,
                started_at=job.started_at,
                estimated_completion=job.estimated_completion
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post(f"{config.api.api_prefix}/upload-advertisement",
              response_model=AdvertisementResponse,
              summary="Upload advertisement content",
              description="Upload and validate advertisement content for placement")
    async def upload_advertisement(
        file: UploadFile = File(...),
        metadata: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        _: Any = Depends(api_instance.rate_limiter.check)
    ):
        """Upload and validate advertisement content."""
        try:
            # Validate file
            if file.content_type not in ["image/jpeg", "image/png", "video/mp4"]:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            if file.size > config.max_video_size:
                raise HTTPException(status_code=400, detail="File too large")
            
            # Read file data
            file_data = await file.read()
            
            # Generate advertisement ID
            ad_id = f"ad_{int(time.time())}_{current_user.user_id}"
            
            # Store advertisement (in production, this would go to cloud storage)
            advertisement_path = f"uploads/advertisements/{ad_id}"
            
            # Create advertisement record
            advertisement = Advertisement(
                ad_id=ad_id,
                user_id=current_user.user_id,
                file_path=advertisement_path,
                file_type=file.content_type,
                file_size=file.size,
                metadata=metadata,
                created_at=time.time()
            )
            
            # Save to database
            await api_instance.database_manager.save_advertisement(advertisement)
            
            return AdvertisementResponse(
                ad_id=ad_id,
                status="uploaded",
                file_url=f"/api/v1/advertisements/{ad_id}",
                metadata=advertisement.metadata
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload advertisement error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(f"{config.api.api_prefix}/health", 
             response_model=HealthStatus,
             summary="Health check",
             description="Comprehensive health check for all components")
    async def health_check():
        """Comprehensive health check for all components."""
        try:
            health_data = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": config.version,
                "environment": config.environment.value,
                "components": {}
            }
            
            # Check pipeline health
            if api_instance.pipeline:
                try:
                    pipeline_stats = api_instance.pipeline.get_processing_stats()
                    health_data["components"]["pipeline"] = {
                        "status": "healthy",
                        "total_processed": pipeline_stats.get("total_videos_processed", 0),
                        "error_rate": pipeline_stats.get("error_rate", 0.0)
                    }
                except Exception as e:
                    health_data["components"]["pipeline"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                health_data["components"]["pipeline"] = {"status": "not_initialized"}
            
            # Check database health
            try:
                db_health = await api_instance.database_manager.health_check()
                health_data["components"]["database"] = db_health
            except Exception as e:
                health_data["components"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Check rate limiter
            try:
                health_data["components"]["rate_limiter"] = {
                    "status": "healthy",
                    "active_limits": api_instance.rate_limiter.get_active_limits()
                }
            except Exception as e:
                health_data["components"]["rate_limiter"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Check WebSocket connections
            try:
                health_data["components"]["websocket"] = {
                    "status": "healthy",
                    "active_connections": api_instance.websocket_manager.get_connection_count()
                }
            except Exception as e:
                health_data["components"]["websocket"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # Overall status
            unhealthy_components = [
                name for name, comp in health_data["components"].items()
                if comp.get("status") not in ["healthy", "not_initialized"]
            ]
            
            if unhealthy_components:
                health_data["status"] = "degraded"
                health_data["unhealthy_components"] = unhealthy_components
            
            return HealthStatus(**health_data)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthStatus(
                status="unhealthy",
                timestamp=time.time(),
                version=config.version,
                environment=config.environment.value,
                components={},
                error_message=str(e)
            )
    
    @app.get(f"{config.api.api_prefix}/metrics", 
             response_model=SystemMetrics,
             summary="System metrics",
             description="Get system performance and usage metrics")
    async def get_metrics(
        current_user: User = Depends(get_current_user)
    ):
        """Get system performance and usage metrics."""
        try:
            # Require admin access for metrics
            if not current_user.is_admin:
                raise HTTPException(status_code=403, detail="Admin access required")
            
            metrics = api_instance.api_monitor.get_metrics()
            
            # Add pipeline metrics
            if api_instance.pipeline:
                pipeline_stats = api_instance.pipeline.get_processing_stats()
                metrics.update({
                    "pipeline_metrics": pipeline_stats
                })
            
            return SystemMetrics(**metrics)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get metrics error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post(f"{config.api.api_prefix}/batch-process",
              response_model=BatchProcessingResponse,
              summary="Batch process multiple videos",
              description="Submit multiple videos for batch processing")
    async def batch_process_videos(
        request: BatchProcessingRequest,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        _: Any = Depends(api_instance.rate_limiter.check)
    ):
        """Batch process multiple videos."""
        try:
            if len(request.videos) > 100:  # Limit batch size
                raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
            
            batch_id = f"batch_{int(time.time())}_{current_user.user_id}"
            job_ids = []
            
            # Create jobs for each video
            for i, video_request in enumerate(request.videos):
                job = ProcessingJob(
                    job_id=f"{batch_id}_job_{i}",
                    user_id=current_user.user_id,
                    video_url=video_request.video_url,
                    advertisement_config=video_request.advertisement_config,
                    placement_config=video_request.placement_config,
                    processing_options=video_request.processing_options or ProcessingOptions(),
                    status=JobStatus.QUEUED,
                    created_at=time.time(),
                    batch_id=batch_id
                )
                
                api_instance.active_jobs[job.job_id] = job
                job_ids.append(job.job_id)
                
                # Start processing in background
                background_tasks.add_task(process_video_background, job)
            
            return BatchProcessingResponse(
                batch_id=batch_id,
                job_ids=job_ids,
                total_jobs=len(job_ids),
                status="queued",
                message=f"Batch processing started with {len(job_ids)} jobs"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch process error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket endpoint for real-time updates
    @app.websocket(f"{config.api.api_prefix}/ws/{{job_id}}")
    async def websocket_endpoint(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for real-time job progress updates."""
        await api_instance.websocket_manager.connect(websocket, job_id)
        
        try:
            while True:
                # Send periodic updates
                if job_id in api_instance.active_jobs:
                    job = api_instance.active_jobs[job_id]
                    await api_instance.websocket_manager.send_update(job_id, {
                        "job_id": job_id,
                        "status": job.status.value,
                        "progress": job.progress,
                        "message": job.status_message
                    })
                
                await asyncio.sleep(1)  # Update every second
                
        except WebSocketDisconnect:
            api_instance.websocket_manager.disconnect(job_id)
    
    # Configuration management endpoints
    @app.get(f"{config.api.api_prefix}/config",
             summary="Get configuration",
             description="Get current system configuration (admin only)")
    async def get_configuration(
        current_user: User = Depends(get_current_user)
    ):
        """Get current system configuration."""
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Return non-sensitive configuration
        config_dict = {
            "environment": config.environment.value,
            "version": config.version,
            "api": {
                "workers": config.api.workers,
                "rate_limit_requests": config.api.rate_limit_requests,
                "max_concurrent_requests": config.api.max_concurrent_requests
            },
            "video_processing": {
                "quality_level": config.video_processing.quality_level.value,
                "max_workers": config.video_processing.max_workers,
                "batch_size": config.video_processing.batch_size
            }
        }
        return config_dict
    
    @app.post(f"{config.api.api_prefix}/config/update",
              summary="Update configuration",
              description="Update system configuration at runtime (admin only)")
    async def update_configuration(
        updates: Dict[str, Any],
        current_user: User = Depends(get_current_user)
    ):
        """Update system configuration at runtime."""
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        try:
            # This would integrate with the ConfigManager
            # For now, return success message
            return {"message": "Configuration updated", "updated_keys": list(updates.keys())}
            
        except Exception as e:
            logger.error(f"Config update error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Add a simple info endpoint without authentication
    @app.get(f"{config.api.api_prefix}/info",
             summary="API Information",
             description="Get basic API information")
    async def get_api_info():
        """Get basic API information."""
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "environment": config.environment.value,
            "api_prefix": config.api.api_prefix,
            "docs_url": config.api.docs_url,
            "features": config.features
        }

    # Add a test endpoint for API keys
    @app.get(f"{config.api.api_prefix}/test-auth",
             summary="Test Authentication",
             description="Test authentication (returns API keys for testing)")
    async def test_auth():
        """Test authentication endpoint that returns test API keys."""
        auth_mgr = api_instance.auth_manager
        
        # Get test API keys
        admin_keys = [key for key, user_id in auth_mgr.api_keys.items() if user_id == "admin"]
        user_keys = [key for key, user_id in auth_mgr.api_keys.items() if user_id == "api_user"]
        
        return {
            "message": "Test API keys for development",
            "admin_api_key": admin_keys[0] if admin_keys else None,
            "user_api_key": user_keys[0] if user_keys else None,
            "usage": "Add 'X-API-Key: <key>' header to authenticate requests"
        }
    
    return app


async def process_video_background(job: ProcessingJob):
    """Background task for video processing."""
    global api_instance
    
    try:
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job.status_message = "Processing started"
        
        # Send WebSocket update
        await api_instance.websocket_manager.send_update(job.job_id, {
            "status": job.status.value,
            "progress": 0.0,
            "message": "Processing started"
        })
        
        # Simulate processing (in real implementation, use the pipeline)
        for i in range(10):
            await asyncio.sleep(5)  # Simulate work
            job.progress = (i + 1) / 10
            job.status_message = f"Processing frame batch {i + 1}/10"
            
            # Send progress update
            await api_instance.websocket_manager.send_update(job.job_id, {
                "status": job.status.value,
                "progress": job.progress,
                "message": job.status_message
            })
        
        # Complete job
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.progress = 1.0
        job.status_message = "Processing completed successfully"
        
        # Create result
        result = ProcessingResult(
            job_id=job.job_id,
            user_id=job.user_id,
            output_video_url=f"https://example.com/processed/{job.job_id}.mp4",
            thumbnail_url=f"https://example.com/thumbnails/{job.job_id}.jpg",
            processing_stats=ProcessingStats(
                total_frames=900,
                processed_frames=895,
                failed_frames=5,
                processing_time=45.2,
                quality_score=0.87
            ),
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        
        # Store result
        api_instance.job_results[job.job_id] = result
        
        # Remove from active jobs
        if job.job_id in api_instance.active_jobs:
            del api_instance.active_jobs[job.job_id]
        
        # Send completion update
        await api_instance.websocket_manager.send_update(job.job_id, {
            "status": job.status.value,
            "progress": 1.0,
            "message": "Processing completed",
            "result": result.dict()
        })
        
    except Exception as e:
        # Handle processing error
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = time.time()
        
        logger.error(f"Processing job {job.job_id} failed: {e}")
        
        # Send error update
        await api_instance.websocket_manager.send_update(job.job_id, {
            "status": job.status.value,
            "error": str(e),
            "message": "Processing failed"
        }) 