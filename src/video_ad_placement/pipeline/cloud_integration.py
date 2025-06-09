"""
Cloud Integration and Distributed Processing System

This module provides comprehensive cloud integration capabilities including cloud storage,
API gateway functionality, distributed processing, and scalable deployment management.
"""

import asyncio
import logging
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiohttp
import boto3
from botocore.exceptions import ClientError
import concurrent.futures
import weakref

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    MANUAL = "manual"
    CPU_BASED = "cpu_based"
    QUEUE_BASED = "queue_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class DeploymentStatus(Enum):
    """Deployment status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SCALING = "scaling"
    ERROR = "error"
    TERMINATING = "terminating"


@dataclass
class CloudConfig:
    """Cloud configuration settings."""
    provider: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"
    
    # Storage settings
    storage_bucket: str = "video-processing-bucket"
    storage_prefix: str = "processing/"
    enable_encryption: bool = True
    retention_days: int = 30
    
    # API Gateway settings
    api_endpoint: str = ""
    api_key: str = ""
    enable_authentication: bool = True
    rate_limit_requests_per_minute: int = 1000
    
    # Distributed processing
    enable_distributed: bool = False
    max_workers: int = 10
    worker_instance_type: str = "m5.large"
    auto_scaling: bool = True
    scaling_policy: ScalingPolicy = ScalingPolicy.QUEUE_BASED
    
    # Monitoring and logging
    enable_cloud_monitoring: bool = True
    log_level: str = "INFO"
    metrics_namespace: str = "VideoProcessing"
    
    # Security
    enable_vpc: bool = True
    security_groups: List[str] = field(default_factory=list)
    iam_role: str = ""


@dataclass
class ProcessingJob:
    """Distributed processing job."""
    job_id: str
    video_path: str
    output_path: str
    config: Dict[str, Any]
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerNode:
    """Distributed worker node."""
    worker_id: str
    instance_id: str
    instance_type: str
    status: str
    current_job: Optional[str] = None
    total_jobs_processed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0


class CloudStorageManager:
    """Manages cloud storage operations for video processing."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.client = None
        self.bucket_name = config.storage_bucket
        
        # Initialize cloud client based on provider
        self._initialize_client()
        
        # Upload/download tracking
        self.transfer_stats = {
            'uploads': 0,
            'downloads': 0,
            'bytes_uploaded': 0,
            'bytes_downloaded': 0,
            'errors': 0
        }
        
        logger.info(f"CloudStorageManager initialized for {config.provider.value}")
    
    def _initialize_client(self):
        """Initialize cloud storage client."""
        if self.config.provider == CloudProvider.AWS:
            self.client = boto3.client('s3', region_name=self.config.region)
        elif self.config.provider == CloudProvider.GCP:
            # Would initialize GCP client
            logger.warning("GCP storage not yet implemented")
        elif self.config.provider == CloudProvider.AZURE:
            # Would initialize Azure client
            logger.warning("Azure storage not yet implemented")
        elif self.config.provider == CloudProvider.LOCAL:
            # Local file system storage
            self.local_storage_path = Path("./cloud_storage")
            self.local_storage_path.mkdir(exist_ok=True)
    
    async def upload_file(self, local_path: str, remote_key: str, 
                         metadata: Dict[str, Any] = None) -> bool:
        """Upload file to cloud storage."""
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            full_key = f"{self.config.storage_prefix}{remote_key}"
            
            if self.config.provider == CloudProvider.AWS:
                # Upload to S3
                extra_args = {}
                if self.config.enable_encryption:
                    extra_args['ServerSideEncryption'] = 'AES256'
                if metadata:
                    extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
                
                self.client.upload_file(str(local_file), self.bucket_name, full_key, ExtraArgs=extra_args)
                
            elif self.config.provider == CloudProvider.LOCAL:
                # Copy to local storage
                target_path = self.local_storage_path / remote_key
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                import shutil
                shutil.copy2(local_file, target_path)
                
                # Store metadata separately
                if metadata:
                    metadata_path = target_path.with_suffix(target_path.suffix + '.meta')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
            
            # Update statistics
            file_size = local_file.stat().st_size
            self.transfer_stats['uploads'] += 1
            self.transfer_stats['bytes_uploaded'] += file_size
            
            logger.info(f"File uploaded: {local_path} -> {full_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_path}: {e}")
            self.transfer_stats['errors'] += 1
            return False
    
    async def download_file(self, remote_key: str, local_path: str) -> bool:
        """Download file from cloud storage."""
        try:
            full_key = f"{self.config.storage_prefix}{remote_key}"
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.provider == CloudProvider.AWS:
                # Download from S3
                self.client.download_file(self.bucket_name, full_key, str(local_file))
                
            elif self.config.provider == CloudProvider.LOCAL:
                # Copy from local storage
                source_path = self.local_storage_path / remote_key
                if not source_path.exists():
                    raise FileNotFoundError(f"Remote file not found: {remote_key}")
                
                import shutil
                shutil.copy2(source_path, local_file)
            
            # Update statistics
            file_size = local_file.stat().st_size
            self.transfer_stats['downloads'] += 1
            self.transfer_stats['bytes_downloaded'] += file_size
            
            logger.info(f"File downloaded: {full_key} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {remote_key}: {e}")
            self.transfer_stats['errors'] += 1
            return False
    
    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in cloud storage."""
        try:
            full_prefix = f"{self.config.storage_prefix}{prefix}"
            
            if self.config.provider == CloudProvider.AWS:
                # List S3 objects
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=full_prefix
                )
                
                files = []
                for obj in response.get('Contents', []):
                    files.append({
                        'key': obj['Key'].replace(self.config.storage_prefix, ''),
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"')
                    })
                
                return files
                
            elif self.config.provider == CloudProvider.LOCAL:
                # List local files
                files = []
                search_path = self.local_storage_path / prefix if prefix else self.local_storage_path
                
                if search_path.exists():
                    for file_path in search_path.rglob('*'):
                        if file_path.is_file() and not file_path.name.endswith('.meta'):
                            relative_path = file_path.relative_to(self.local_storage_path)
                            files.append({
                                'key': str(relative_path),
                                'size': file_path.stat().st_size,
                                'last_modified': time.ctime(file_path.stat().st_mtime),
                                'etag': hashlib.md5(file_path.read_bytes()).hexdigest()
                            })
                
                return files
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from cloud storage."""
        try:
            full_key = f"{self.config.storage_prefix}{remote_key}"
            
            if self.config.provider == CloudProvider.AWS:
                self.client.delete_object(Bucket=self.bucket_name, Key=full_key)
                
            elif self.config.provider == CloudProvider.LOCAL:
                file_path = self.local_storage_path / remote_key
                if file_path.exists():
                    file_path.unlink()
                
                # Delete metadata file if exists
                metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
                if metadata_path.exists():
                    metadata_path.unlink()
            
            logger.info(f"File deleted: {full_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {remote_key}: {e}")
            return False
    
    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get file transfer statistics."""
        return dict(self.transfer_stats)


class APIGateway:
    """API Gateway for handling video processing requests."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Request tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'rate_limit_hits': 0
        }
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_window = 60.0  # 1 minute
        
        logger.info("APIGateway initialized")
    
    async def initialize(self):
        """Initialize the API gateway."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the API gateway."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def submit_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """Submit processing job to API."""
        if not self.session:
            await self.initialize()
        
        # Check rate limiting
        if not self._check_rate_limit():
            self.request_stats['rate_limit_hits'] += 1
            raise Exception("Rate limit exceeded")
        
        start_time = time.time()
        
        try:
            job_data = {
                'job_id': job.job_id,
                'video_path': job.video_path,
                'output_path': job.output_path,
                'config': job.config,
                'priority': job.priority
            }
            
            headers = {}
            if self.config.api_key:
                headers['Authorization'] = f"Bearer {self.config.api_key}"
            
            async with self.session.post(
                f"{self.config.api_endpoint}/jobs",
                json=job_data,
                headers=headers
            ) as response:
                
                response_time = time.time() - start_time
                self._update_request_stats(True, response_time)
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Job submitted successfully: {job.job_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        
        except Exception as e:
            response_time = time.time() - start_time
            self._update_request_stats(False, response_time)
            logger.error(f"Failed to submit job {job.job_id}: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from API."""
        if not self.session:
            await self.initialize()
        
        if not self._check_rate_limit():
            self.request_stats['rate_limit_hits'] += 1
            raise Exception("Rate limit exceeded")
        
        start_time = time.time()
        
        try:
            headers = {}
            if self.config.api_key:
                headers['Authorization'] = f"Bearer {self.config.api_key}"
            
            async with self.session.get(
                f"{self.config.api_endpoint}/jobs/{job_id}",
                headers=headers
            ) as response:
                
                response_time = time.time() - start_time
                self._update_request_stats(True, response_time)
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        
        except Exception as e:
            response_time = time.time() - start_time
            self._update_request_stats(False, response_time)
            logger.error(f"Failed to get job status {job_id}: {e}")
            raise
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Clean old requests
        cutoff_time = current_time - self.rate_limit_window
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check limit
        if len(self.request_times) >= self.config.rate_limit_requests_per_minute:
            return False
        
        self.request_times.append(current_time)
        return True
    
    def _update_request_stats(self, success: bool, response_time: float):
        """Update request statistics."""
        self.request_stats['total_requests'] += 1
        
        if success:
            self.request_stats['successful_requests'] += 1
        else:
            self.request_stats['failed_requests'] += 1
        
        # Update average response time
        total_successful = self.request_stats['successful_requests']
        if total_successful > 0:
            current_avg = self.request_stats['avg_response_time']
            self.request_stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return dict(self.request_stats)


class DistributedProcessor:
    """Manages distributed video processing across multiple workers."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.workers: Dict[str, WorkerNode] = {}
        self.jobs: Dict[str, ProcessingJob] = {}
        
        # Processing state
        self.is_running = False
        self.coordinator_task: Optional[asyncio.Task] = None
        
        # Auto-scaling
        self.target_workers = config.max_workers
        self.scaling_cooldown = 300.0  # 5 minutes
        self.last_scaling_action = 0.0
        
        logger.info("DistributedProcessor initialized")
    
    async def start(self):
        """Start the distributed processor."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start coordinator task
        self.coordinator_task = asyncio.create_task(self._coordinator_loop())
        
        # Initialize workers if enabled
        if self.config.enable_distributed:
            await self._initialize_workers()
        
        logger.info("DistributedProcessor started")
    
    async def stop(self):
        """Stop the distributed processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel coordinator task
        if self.coordinator_task:
            self.coordinator_task.cancel()
            try:
                await self.coordinator_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown workers
        await self._shutdown_workers()
        
        logger.info("DistributedProcessor stopped")
    
    async def submit_job(self, job: ProcessingJob) -> str:
        """Submit job for distributed processing."""
        job.status = "queued"
        self.jobs[job.job_id] = job
        
        await self.job_queue.put(job)
        logger.info(f"Job queued: {job.job_id}")
        
        return job.job_id
    
    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job status."""
        return self.jobs.get(job_id)
    
    async def _coordinator_loop(self):
        """Main coordinator loop for job distribution."""
        while self.is_running:
            try:
                # Check for available workers and pending jobs
                available_workers = [
                    worker for worker in self.workers.values()
                    if worker.status == "idle" and worker.current_job is None
                ]
                
                if available_workers and not self.job_queue.empty():
                    # Assign job to worker
                    try:
                        job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                        worker = available_workers[0]  # Simple assignment
                        
                        await self._assign_job_to_worker(job, worker)
                        
                    except asyncio.TimeoutError:
                        pass
                
                # Auto-scaling check
                if self.config.auto_scaling:
                    await self._check_autoscaling()
                
                # Health check workers
                await self._health_check_workers()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Coordinator loop error: {e}")
    
    async def _assign_job_to_worker(self, job: ProcessingJob, worker: WorkerNode):
        """Assign job to a specific worker."""
        job.status = "running"
        job.started_at = time.time()
        job.worker_id = worker.worker_id
        
        worker.current_job = job.job_id
        worker.status = "busy"
        
        # In a real implementation, this would send the job to the worker
        # For now, simulate processing
        asyncio.create_task(self._simulate_job_processing(job, worker))
        
        logger.info(f"Job {job.job_id} assigned to worker {worker.worker_id}")
    
    async def _simulate_job_processing(self, job: ProcessingJob, worker: WorkerNode):
        """Simulate job processing (for demonstration)."""
        try:
            # Simulate processing time
            processing_time = 30.0  # 30 seconds
            await asyncio.sleep(processing_time)
            
            # Mark job as completed
            job.status = "completed"
            job.completed_at = time.time()
            job.progress = 1.0
            
            # Update worker
            worker.current_job = None
            worker.status = "idle"
            worker.total_jobs_processed += 1
            
            logger.info(f"Job {job.job_id} completed by worker {worker.worker_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            
            worker.current_job = None
            worker.status = "idle"
            
            logger.error(f"Job {job.job_id} failed on worker {worker.worker_id}: {e}")
    
    async def _initialize_workers(self):
        """Initialize worker nodes."""
        for i in range(self.target_workers):
            worker = WorkerNode(
                worker_id=f"worker_{i}",
                instance_id=f"instance_{i}",
                instance_type=self.config.worker_instance_type,
                status="idle"
            )
            self.workers[worker.worker_id] = worker
        
        logger.info(f"Initialized {len(self.workers)} workers")
    
    async def _shutdown_workers(self):
        """Shutdown all workers."""
        for worker in self.workers.values():
            worker.status = "terminating"
        
        self.workers.clear()
        logger.info("All workers shutdown")
    
    async def _check_autoscaling(self):
        """Check if autoscaling is needed."""
        current_time = time.time()
        
        if current_time - self.last_scaling_action < self.scaling_cooldown:
            return
        
        if self.config.scaling_policy == ScalingPolicy.QUEUE_BASED:
            queue_size = self.job_queue.qsize()
            busy_workers = len([w for w in self.workers.values() if w.status == "busy"])
            
            # Scale up if queue is backing up
            if queue_size > len(self.workers) and len(self.workers) < self.config.max_workers:
                await self._scale_up()
                self.last_scaling_action = current_time
            
            # Scale down if workers are idle
            elif busy_workers < len(self.workers) // 2 and len(self.workers) > 1:
                await self._scale_down()
                self.last_scaling_action = current_time
    
    async def _scale_up(self):
        """Add more workers."""
        new_worker_id = f"worker_{len(self.workers)}"
        worker = WorkerNode(
            worker_id=new_worker_id,
            instance_id=f"instance_{len(self.workers)}",
            instance_type=self.config.worker_instance_type,
            status="idle"
        )
        self.workers[new_worker_id] = worker
        logger.info(f"Scaled up: added worker {new_worker_id}")
    
    async def _scale_down(self):
        """Remove idle workers."""
        idle_workers = [w for w in self.workers.values() if w.status == "idle"]
        if idle_workers:
            worker_to_remove = idle_workers[0]
            del self.workers[worker_to_remove.worker_id]
            logger.info(f"Scaled down: removed worker {worker_to_remove.worker_id}")
    
    async def _health_check_workers(self):
        """Check health of all workers."""
        current_time = time.time()
        
        for worker in list(self.workers.values()):
            # Check if worker is responsive (simulated)
            if current_time - worker.last_heartbeat > 300.0:  # 5 minutes
                logger.warning(f"Worker {worker.worker_id} appears unresponsive")
                worker.status = "error"
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get distributed processing statistics."""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == "completed"])
        failed_jobs = len([j for j in self.jobs.values() if j.status == "failed"])
        
        return {
            'total_workers': len(self.workers),
            'busy_workers': len([w for w in self.workers.values() if w.status == "busy"]),
            'idle_workers': len([w for w in self.workers.values() if w.status == "idle"]),
            'queue_size': self.job_queue.qsize(),
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': completed_jobs / max(1, total_jobs)
        }


class DeploymentManager:
    """Manages scalable deployment of the video processing pipeline."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.status = DeploymentStatus.STOPPED
        
        # Components
        self.storage_manager = CloudStorageManager(config)
        self.api_gateway = APIGateway(config)
        self.distributed_processor = DistributedProcessor(config)
        
        # Deployment monitoring
        self.deployment_stats = {
            'start_time': None,
            'uptime_seconds': 0,
            'total_requests': 0,
            'successful_deployments': 0,
            'failed_deployments': 0
        }
        
        logger.info("DeploymentManager initialized")
    
    async def deploy(self) -> bool:
        """Deploy the entire system."""
        if self.status != DeploymentStatus.STOPPED:
            logger.warning("System is already deployed or deploying")
            return False
        
        self.status = DeploymentStatus.STARTING
        
        try:
            logger.info("Starting system deployment...")
            
            # Initialize API Gateway
            await self.api_gateway.initialize()
            
            # Start distributed processor if enabled
            if self.config.enable_distributed:
                await self.distributed_processor.start()
            
            # Verify cloud storage connectivity
            await self.storage_manager.list_files("")
            
            self.status = DeploymentStatus.RUNNING
            self.deployment_stats['start_time'] = time.time()
            self.deployment_stats['successful_deployments'] += 1
            
            logger.info("System deployment completed successfully")
            return True
            
        except Exception as e:
            self.status = DeploymentStatus.ERROR
            self.deployment_stats['failed_deployments'] += 1
            logger.error(f"Deployment failed: {e}")
            
            # Cleanup on failure
            await self._cleanup_deployment()
            return False
    
    async def undeploy(self) -> bool:
        """Undeploy the entire system."""
        if self.status == DeploymentStatus.STOPPED:
            return True
        
        self.status = DeploymentStatus.TERMINATING
        
        try:
            logger.info("Starting system undeployment...")
            
            # Stop distributed processor
            await self.distributed_processor.stop()
            
            # Close API Gateway
            await self.api_gateway.close()
            
            self.status = DeploymentStatus.STOPPED
            
            # Update uptime
            if self.deployment_stats['start_time']:
                self.deployment_stats['uptime_seconds'] += (
                    time.time() - self.deployment_stats['start_time']
                )
            
            logger.info("System undeployment completed")
            return True
            
        except Exception as e:
            logger.error(f"Undeployment error: {e}")
            return False
    
    async def _cleanup_deployment(self):
        """Cleanup resources after failed deployment."""
        try:
            await self.distributed_processor.stop()
            await self.api_gateway.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        # Check API Gateway
        try:
            api_stats = self.api_gateway.get_api_stats()
            health_status['components']['api_gateway'] = {
                'status': 'healthy',
                'stats': api_stats
            }
        except Exception as e:
            health_status['components']['api_gateway'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check Storage
        try:
            storage_stats = self.storage_manager.get_transfer_stats()
            health_status['components']['storage'] = {
                'status': 'healthy',
                'stats': storage_stats
            }
        except Exception as e:
            health_status['components']['storage'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check Distributed Processor
        try:
            processing_stats = self.distributed_processor.get_processing_stats()
            health_status['components']['distributed_processor'] = {
                'status': 'healthy',
                'stats': processing_stats
            }
        except Exception as e:
            health_status['components']['distributed_processor'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'unhealthy'
        
        return health_status
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        stats = dict(self.deployment_stats)
        
        if self.deployment_stats['start_time'] and self.status == DeploymentStatus.RUNNING:
            stats['current_uptime_seconds'] = time.time() - self.deployment_stats['start_time']
        
        stats['status'] = self.status.value
        
        return stats 