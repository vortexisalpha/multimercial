"""
Database Manager

This module provides database connectivity and data persistence
with health checks and connection management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .models import Advertisement, ProcessingJob, ProcessingResult, User
from ..config.hydra_config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for data persistence."""
    
    def __init__(self, database_config: DatabaseConfig):
        self.config = database_config
        self.connected = False
        
        # In-memory storage for testing (in production, use actual database)
        self.advertisements: Dict[str, Advertisement] = {}
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        self.processing_results: Dict[str, ProcessingResult] = {}
        self.users: Dict[str, User] = {}
        
        # Connection pool simulation
        self.connection_pool_size = database_config.max_connections
        self.active_connections = 0
        
        # Health check data
        self.last_health_check = 0.0
        self.health_status = "unknown"
        
        logger.info(f"DatabaseManager initialized with {database_config.type.value} backend")
    
    async def initialize(self):
        """Initialize database connection and tables."""
        try:
            # Simulate database connection
            await asyncio.sleep(0.1)  # Simulate connection time
            
            # Create tables (simulated)
            await self._create_tables()
            
            self.connected = True
            self.health_status = "healthy"
            self.last_health_check = time.time()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.connected = False
            self.health_status = "unhealthy"
            raise
    
    async def close(self):
        """Close database connections."""
        try:
            # Simulate closing connections
            await asyncio.sleep(0.1)
            
            self.connected = False
            self.active_connections = 0
            self.health_status = "disconnected"
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def _create_tables(self):
        """Create database tables."""
        # In a real implementation, this would create actual tables
        # For now, we'll just initialize our in-memory storage
        
        tables = [
            "users",
            "advertisements", 
            "processing_jobs",
            "processing_results",
            "system_config"
        ]
        
        logger.info(f"Created/verified {len(tables)} database tables")
    
    # User operations
    async def save_user(self, user: User) -> bool:
        """Save user to database."""
        try:
            self.users[user.user_id] = user
            logger.debug(f"Saved user: {user.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user {user.user_id}: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            logger.debug(f"Updated user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
    
    # Advertisement operations
    async def save_advertisement(self, advertisement: Advertisement) -> bool:
        """Save advertisement to database."""
        try:
            self.advertisements[advertisement.ad_id] = advertisement
            logger.debug(f"Saved advertisement: {advertisement.ad_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save advertisement {advertisement.ad_id}: {e}")
            return False
    
    async def get_advertisement(self, ad_id: str) -> Optional[Advertisement]:
        """Get advertisement by ID."""
        return self.advertisements.get(ad_id)
    
    async def get_user_advertisements(self, user_id: str) -> List[Advertisement]:
        """Get all advertisements for a user."""
        return [
            ad for ad in self.advertisements.values() 
            if ad.user_id == user_id
        ]
    
    async def delete_advertisement(self, ad_id: str) -> bool:
        """Delete advertisement."""
        try:
            if ad_id in self.advertisements:
                del self.advertisements[ad_id]
                logger.debug(f"Deleted advertisement: {ad_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete advertisement {ad_id}: {e}")
            return False
    
    # Processing job operations
    async def save_processing_job(self, job: ProcessingJob) -> bool:
        """Save processing job to database."""
        try:
            self.processing_jobs[job.job_id] = job
            logger.debug(f"Saved processing job: {job.job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processing job {job.job_id}: {e}")
            return False
    
    async def get_processing_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get processing job by ID."""
        return self.processing_jobs.get(job_id)
    
    async def get_user_jobs(self, user_id: str, limit: int = 100) -> List[ProcessingJob]:
        """Get processing jobs for a user."""
        user_jobs = [
            job for job in self.processing_jobs.values() 
            if job.user_id == user_id
        ]
        
        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_jobs[:limit]
    
    async def update_processing_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update processing job."""
        try:
            if job_id not in self.processing_jobs:
                return False
            
            job = self.processing_jobs[job_id]
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            logger.debug(f"Updated processing job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update processing job {job_id}: {e}")
            return False
    
    # Processing result operations
    async def save_processing_result(self, result: ProcessingResult) -> bool:
        """Save processing result to database."""
        try:
            self.processing_results[result.job_id] = result
            logger.debug(f"Saved processing result: {result.job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processing result {result.job_id}: {e}")
            return False
    
    async def get_processing_result(self, job_id: str) -> Optional[ProcessingResult]:
        """Get processing result by job ID."""
        return self.processing_results.get(job_id)
    
    async def get_user_results(self, user_id: str, limit: int = 100) -> List[ProcessingResult]:
        """Get processing results for a user."""
        user_results = [
            result for result in self.processing_results.values() 
            if result.user_id == user_id
        ]
        
        # Sort by completion time (newest first)
        user_results.sort(key=lambda x: x.completed_at, reverse=True)
        
        return user_results[:limit]
    
    # Analytics and statistics
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job processing statistics."""
        try:
            from .models import JobStatus
            
            total_jobs = len(self.processing_jobs)
            
            status_counts = {}
            for status in JobStatus:
                status_counts[status.value] = sum(
                    1 for job in self.processing_jobs.values() 
                    if job.status == status
                )
            
            # Calculate processing times
            completed_jobs = [
                job for job in self.processing_jobs.values() 
                if job.status == JobStatus.COMPLETED and job.completed_at and job.started_at
            ]
            
            avg_processing_time = 0.0
            if completed_jobs:
                total_time = sum(
                    job.completed_at - job.started_at 
                    for job in completed_jobs
                )
                avg_processing_time = total_time / len(completed_jobs)
            
            return {
                "total_jobs": total_jobs,
                "status_counts": status_counts,
                "completed_jobs": len(completed_jobs),
                "average_processing_time": avg_processing_time,
                "success_rate": len(completed_jobs) / total_jobs if total_jobs > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get job statistics: {e}")
            return {}
    
    async def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics."""
        try:
            total_users = len(self.users)
            active_users = sum(1 for user in self.users.values() if user.is_active)
            admin_users = sum(1 for user in self.users.values() if user.is_admin)
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "admin_users": admin_users,
                "inactive_users": total_users - active_users
            }
            
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            return {}
    
    async def cleanup_old_records(self, days: int = 30):
        """Clean up old records."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            # Clean up old jobs
            old_jobs = [
                job_id for job_id, job in self.processing_jobs.items()
                if job.created_at < cutoff_time
            ]
            
            for job_id in old_jobs:
                del self.processing_jobs[job_id]
                if job_id in self.processing_results:
                    del self.processing_results[job_id]
            
            # Clean up old advertisements
            old_ads = [
                ad_id for ad_id, ad in self.advertisements.items()
                if ad.created_at < cutoff_time
            ]
            
            for ad_id in old_ads:
                del self.advertisements[ad_id]
            
            logger.info(f"Cleaned up {len(old_jobs)} old jobs and {len(old_ads)} old advertisements")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    # Health and monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            start_time = time.time()
            
            # Simulate health check query
            await asyncio.sleep(0.01)
            
            query_time = time.time() - start_time
            self.last_health_check = time.time()
            
            # Check connection pool
            pool_utilization = self.active_connections / self.connection_pool_size
            
            if self.connected and query_time < 1.0 and pool_utilization < 0.8:
                self.health_status = "healthy"
                status = "healthy"
            elif self.connected:
                self.health_status = "degraded"
                status = "degraded" 
            else:
                self.health_status = "unhealthy"
                status = "unhealthy"
            
            return {
                "status": status,
                "connected": self.connected,
                "query_time": query_time,
                "pool_utilization": pool_utilization,
                "active_connections": self.active_connections,
                "max_connections": self.connection_pool_size,
                "last_check": self.last_health_check,
                "database_type": self.config.type.value,
                "records": {
                    "users": len(self.users),
                    "advertisements": len(self.advertisements),
                    "jobs": len(self.processing_jobs),
                    "results": len(self.processing_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_status = "unhealthy"
            
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "last_check": time.time()
            }
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        return {
            "database_type": self.config.type.value,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.name,
            "connected": self.connected,
            "pool_size": self.connection_pool_size,
            "active_connections": self.active_connections,
            "health_status": self.health_status
        }
    
    # Transaction simulation (for testing)
    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a series of operations as a transaction."""
        try:
            # In a real database, this would use actual transactions
            # For testing, we'll simulate success/failure
            
            for operation in operations:
                operation_type = operation.get("type")
                data = operation.get("data", {})
                
                if operation_type == "save_user":
                    await self.save_user(User(**data))
                elif operation_type == "save_job":
                    await self.save_processing_job(ProcessingJob(**data))
                elif operation_type == "save_result":
                    await self.save_processing_result(ProcessingResult(**data))
                # Add more operation types as needed
            
            logger.debug(f"Executed transaction with {len(operations)} operations")
            return True
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            # In a real database, this would rollback the transaction
            return False 