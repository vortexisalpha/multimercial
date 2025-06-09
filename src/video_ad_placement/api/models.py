"""
API Models

This module defines all Pydantic models used in the API for request/response validation.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import time


class JobStatus(str, Enum):
    """Processing job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Video processing quality levels."""
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class PlacementStrategy(str, Enum):
    """Advertisement placement strategies."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SMART_PLACEMENT = "smart_placement"
    WALL_ONLY = "wall_only"
    OBJECT_REPLACEMENT = "object_replacement"


# User Models
class User(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: str
    is_admin: bool = False
    is_active: bool = True
    created_at: float = Field(default_factory=time.time)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "username": "john_doe",
                "email": "john@example.com",
                "is_admin": False,
                "is_active": True
            }
        }


class UserCreate(BaseModel):
    """User creation request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


# Advertisement Models
class AdvertisementConfig(BaseModel):
    """Advertisement configuration."""
    ad_type: str = Field(default="image", description="Type of advertisement")
    ad_url: Optional[str] = Field(None, description="URL to advertisement content")
    ad_data: Optional[str] = Field(None, description="Base64 encoded advertisement data")
    width: float = Field(default=1.0, description="Advertisement width in meters")
    height: float = Field(default=0.6, description="Advertisement height in meters")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Advertisement opacity")
    duration: Optional[float] = Field(None, description="Duration for video ads")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ad_type": "image",
                "ad_url": "https://example.com/ad.jpg",
                "width": 1.0,
                "height": 0.6,
                "opacity": 0.8
            }
        }


class PlacementConfig(BaseModel):
    """Advertisement placement configuration."""
    strategy: PlacementStrategy = PlacementStrategy.AUTOMATIC
    min_wall_area: float = Field(default=1.5, description="Minimum wall area for placement")
    max_distance: float = Field(default=8.0, description="Maximum distance from camera")
    preferred_positions: List[str] = Field(default_factory=list, description="Preferred placement positions")
    avoid_objects: List[str] = Field(default_factory=list, description="Objects to avoid when placing ads")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Quality threshold for placement")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "automatic",
                "min_wall_area": 1.5,
                "max_distance": 8.0,
                "quality_threshold": 0.7
            }
        }


class ProcessingOptions(BaseModel):
    """Video processing options."""
    quality_level: QualityLevel = QualityLevel.HIGH
    use_gpu: bool = True
    max_processing_time: int = Field(default=3600, description="Maximum processing time in seconds")
    save_intermediate: bool = Field(default=False, description="Save intermediate processing results")
    output_format: str = Field(default="mp4", description="Output video format")
    output_quality: str = Field(default="1080p", description="Output video quality")
    
    class Config:
        json_schema_extra = {
            "example": {
                "quality_level": "high",
                "use_gpu": True,
                "max_processing_time": 1800,
                "output_format": "mp4",
                "output_quality": "1080p"
            }
        }


# Request Models
class VideoProcessingRequest(BaseModel):
    """Video processing request."""
    video_url: Optional[str] = Field(None, description="URL to video file")
    video_data: Optional[str] = Field(None, description="Base64 encoded video data")
    advertisement_config: AdvertisementConfig
    placement_config: Optional[PlacementConfig] = Field(default_factory=PlacementConfig)
    processing_options: Optional[ProcessingOptions] = Field(default_factory=ProcessingOptions)
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    @model_validator(mode='after')
    def validate_video_input(self):
        if not self.video_url and not self.video_data:
            raise ValueError('Either video_url or video_data must be provided')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "advertisement_config": {
                    "ad_type": "image",
                    "ad_url": "https://example.com/ad.jpg",
                    "width": 1.0,
                    "height": 0.6
                },
                "placement_config": {
                    "strategy": "automatic",
                    "quality_threshold": 0.7
                }
            }
        }


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""
    videos: List[VideoProcessingRequest] = Field(..., max_items=100)
    batch_options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=10, description="Processing priority")
    
    class Config:
        json_schema_extra = {
            "example": {
                "videos": [
                    {
                        "video_url": "https://example.com/video1.mp4",
                        "advertisement_config": {"ad_type": "image", "ad_url": "https://example.com/ad.jpg"}
                    }
                ],
                "priority": 5
            }
        }


# Response Models
class ProcessingResponse(BaseModel):
    """Processing response."""
    job_id: str
    status: JobStatus
    message: str
    estimated_completion_time: Optional[float] = None
    webhook_url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_1640995200_user123",
                "status": "queued",
                "message": "Video processing started",
                "estimated_completion_time": 1640995500.0
            }
        }


class ProcessingStats(BaseModel):
    """Processing statistics."""
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    processing_time: float = 0.0
    quality_score: float = 0.0
    placements_found: int = 0
    placements_used: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_frames": 900,
                "processed_frames": 895,
                "failed_frames": 5,
                "processing_time": 45.2,
                "quality_score": 0.87,
                "placements_found": 12,
                "placements_used": 8
            }
        }


class ProcessingResult(BaseModel):
    """Processing result."""
    job_id: str
    user_id: str
    output_video_url: str
    thumbnail_url: Optional[str] = None
    preview_url: Optional[str] = None
    processing_stats: ProcessingStats
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float
    completed_at: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_1640995200_user123",
                "user_id": "user123",
                "output_video_url": "https://example.com/processed/job_1640995200_user123.mp4",
                "thumbnail_url": "https://example.com/thumbnails/job_1640995200_user123.jpg",
                "processing_stats": {
                    "total_frames": 900,
                    "processed_frames": 895,
                    "processing_time": 45.2
                }
            }
        }


class ProcessingStatus(BaseModel):
    """Processing status."""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0, description="Processing progress (0.0 to 1.0)")
    message: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[ProcessingResult] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_completion: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_1640995200_user123",
                "status": "processing",
                "progress": 0.35,
                "message": "Processing frame batch 3/10",
                "created_at": 1640995200.0,
                "started_at": 1640995205.0
            }
        }


class ProcessingJob(BaseModel):
    """Internal processing job model."""
    job_id: str
    user_id: str
    video_url: Optional[str] = None
    video_data: Optional[str] = None
    advertisement_config: AdvertisementConfig
    placement_config: PlacementConfig
    processing_options: ProcessingOptions
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    status_message: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_completion: Optional[float] = None
    batch_id: Optional[str] = None
    callback_url: Optional[str] = None


class Advertisement(BaseModel):
    """Advertisement model."""
    ad_id: str
    user_id: str
    file_path: str
    file_type: str
    file_size: int
    metadata: Optional[str] = None
    created_at: float
    is_active: bool = True


class AdvertisementResponse(BaseModel):
    """Advertisement upload response."""
    ad_id: str
    status: str
    file_url: str
    metadata: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "ad_id": "ad_1640995200_user123",
                "status": "uploaded",
                "file_url": "/api/v1/advertisements/ad_1640995200_user123",
                "metadata": "Sample advertisement"
            }
        }


class BatchProcessingResponse(BaseModel):
    """Batch processing response."""
    batch_id: str
    job_ids: List[str]
    total_jobs: int
    status: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_1640995200_user123",
                "job_ids": ["batch_1640995200_user123_job_0", "batch_1640995200_user123_job_1"],
                "total_jobs": 2,
                "status": "queued",
                "message": "Batch processing started with 2 jobs"
            }
        }


# Health and Monitoring Models
class ComponentHealth(BaseModel):
    """Component health status."""
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    message: Optional[str] = None
    last_check: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system health")
    timestamp: float
    version: str
    environment: str
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
    unhealthy_components: Optional[List[str]] = None
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1640995200.0,
                "version": "1.0.0",
                "environment": "production",
                "components": {
                    "database": {"status": "healthy"},
                    "pipeline": {"status": "healthy"}
                }
            }
        }


class SystemMetrics(BaseModel):
    """System metrics."""
    timestamp: float = Field(default_factory=time.time)
    requests_per_minute: float = 0.0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    average_processing_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_usage: Optional[float] = None
    pipeline_metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1640995200.0,
                "requests_per_minute": 15.5,
                "active_jobs": 3,
                "completed_jobs": 147,
                "failed_jobs": 2,
                "average_processing_time": 42.3,
                "cpu_usage": 0.65,
                "memory_usage": 0.78
            }
        }


# Authentication Models
class Token(BaseModel):
    """Authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


# Error Models
class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid input parameters",
                "details": {"field": "video_url", "issue": "Invalid URL format"},
                "timestamp": 1640995200.0
            }
        } 