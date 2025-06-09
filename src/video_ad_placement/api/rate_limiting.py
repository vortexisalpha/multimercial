"""
Rate Limiting

This module provides rate limiting functionality for API endpoints
with support for different backends (memory, Redis) and configurable limits.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass
from abc import ABC, abstractmethod

from fastapi import Request, HTTPException, status
from ..config.hydra_config import APIConfig

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window: int  # seconds
    burst: Optional[int] = None  # burst allowance


@dataclass
class RateLimitState:
    """Rate limit state for a client."""
    requests: int
    window_start: float
    last_request: float
    burst_used: int = 0


class RateLimitBackend(ABC):
    """Abstract rate limit backend."""
    
    @abstractmethod
    async def check_rate_limit(self, key: str, limit: RateLimit) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit."""
        pass
    
    @abstractmethod
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        pass


class MemoryBackend(RateLimitBackend):
    """In-memory rate limiting backend."""
    
    def __init__(self):
        self.state: Dict[str, RateLimitState] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, key: str, limit: RateLimit) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit."""
        async with self._lock:
            current_time = time.time()
            
            # Get or create state
            if key not in self.state:
                self.state[key] = RateLimitState(
                    requests=0,
                    window_start=current_time,
                    last_request=current_time
                )
            
            state = self.state[key]
            history = self.request_history[key]
            
            # Clean old requests from history
            while history and history[0] <= current_time - limit.window:
                history.popleft()
            
            # Check if window has reset
            if current_time - state.window_start >= limit.window:
                state.requests = 0
                state.window_start = current_time
                state.burst_used = 0
            
            # Calculate current request count
            current_requests = len(history)
            
            # Check burst limit if configured
            burst_limit = limit.burst or limit.requests
            if current_requests >= burst_limit:
                return False, {
                    "allowed": False,
                    "limit": limit.requests,
                    "remaining": 0,
                    "reset_time": state.window_start + limit.window,
                    "retry_after": state.window_start + limit.window - current_time
                }
            
            # Check rate limit
            if current_requests >= limit.requests:
                return False, {
                    "allowed": False,
                    "limit": limit.requests,
                    "remaining": 0,
                    "reset_time": state.window_start + limit.window,
                    "retry_after": state.window_start + limit.window - current_time
                }
            
            # Allow request
            history.append(current_time)
            state.requests += 1
            state.last_request = current_time
            
            return True, {
                "allowed": True,
                "limit": limit.requests,
                "remaining": limit.requests - current_requests - 1,
                "reset_time": state.window_start + limit.window,
                "retry_after": 0
            }
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        async with self._lock:
            if key in self.state:
                del self.state[key]
            if key in self.request_history:
                self.request_history[key].clear()
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        async with self._lock:
            total_clients = len(self.state)
            total_requests = sum(len(history) for history in self.request_history.values())
            
            return {
                "backend": "memory",
                "total_clients": total_clients,
                "total_requests": total_requests,
                "active_limits": total_clients
            }


class RedisBackend(RateLimitBackend):
    """Redis-based rate limiting backend."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_db: int = 0, redis_password: Optional[str] = None):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.redis = None
        self._connected = False
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if not self._connected:
            try:
                import redis.asyncio as redis
                self.redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    decode_responses=True
                )
                await self.redis.ping()
                self._connected = True
                logger.info("Redis rate limiting backend connected")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def check_rate_limit(self, key: str, limit: RateLimit) -> tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limit using Redis."""
        await self._ensure_connected()
        
        current_time = time.time()
        window_start = current_time - (current_time % limit.window)
        redis_key = f"rate_limit:{key}:{int(window_start)}"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, limit.window + 1)  # Extra second for cleanup
            results = await pipe.execute()
            
            current_requests = results[0]
            
            if current_requests > limit.requests:
                return False, {
                    "allowed": False,
                    "limit": limit.requests,
                    "remaining": 0,
                    "reset_time": window_start + limit.window,
                    "retry_after": window_start + limit.window - current_time
                }
            
            return True, {
                "allowed": True,
                "limit": limit.requests,
                "remaining": limit.requests - current_requests,
                "reset_time": window_start + limit.window,
                "retry_after": 0
            }
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to allowing the request
            return True, {
                "allowed": True,
                "limit": limit.requests,
                "remaining": limit.requests,
                "reset_time": current_time + limit.window,
                "retry_after": 0,
                "error": "Backend unavailable"
            }
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        await self._ensure_connected()
        
        try:
            pattern = f"rate_limit:{key}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        await self._ensure_connected()
        
        try:
            info = await self.redis.info()
            pattern = "rate_limit:*"
            keys = await self.redis.keys(pattern)
            
            return {
                "backend": "redis",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "active_limits": len(keys),
                "memory_usage": info.get("used_memory_human")
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {
                "backend": "redis",
                "error": str(e)
            }


class RateLimiter:
    """Main rate limiter class."""
    
    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.enabled = api_config.rate_limit_enabled
        
        # Create backend
        if api_config.rate_limit_storage == "redis":
            self.backend = RedisBackend(
                redis_host=getattr(api_config, 'redis_host', 'localhost'),
                redis_port=getattr(api_config, 'redis_port', 6379),
                redis_db=getattr(api_config, 'redis_db', 0),
                redis_password=getattr(api_config, 'redis_password', None)
            )
        else:
            self.backend = MemoryBackend()
        
        # Default rate limits
        self.default_limit = RateLimit(
            requests=api_config.rate_limit_requests,
            window=api_config.rate_limit_window
        )
        
        # Per-endpoint limits
        self.endpoint_limits: Dict[str, RateLimit] = {
            "/api/v1/process-video": RateLimit(requests=10, window=60),  # 10 per minute
            "/api/v1/batch-process": RateLimit(requests=2, window=300),   # 2 per 5 minutes
            "/api/v1/upload-advertisement": RateLimit(requests=20, window=60),  # 20 per minute
        }
        
        # Per-user limits (user_id -> RateLimit)
        self.user_limits: Dict[str, RateLimit] = {}
        
        logger.info(f"RateLimiter initialized with {type(self.backend).__name__} backend")
    
    def get_client_key(self, request: Request, user_id: Optional[str] = None) -> str:
        """Generate client key for rate limiting."""
        if user_id:
            return f"user:{user_id}"
        
        # Use client IP as fallback
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def get_rate_limit(self, endpoint: str, user_id: Optional[str] = None) -> RateLimit:
        """Get rate limit for endpoint and user."""
        # Check user-specific limits first
        if user_id and user_id in self.user_limits:
            return self.user_limits[user_id]
        
        # Check endpoint-specific limits
        if endpoint in self.endpoint_limits:
            return self.endpoint_limits[endpoint]
        
        # Return default limit
        return self.default_limit
    
    async def check(self, request: Request, user_id: Optional[str] = None) -> bool:
        """Check rate limit for request."""
        if not self.enabled:
            return True
        
        try:
            client_key = self.get_client_key(request, user_id)
            endpoint = request.url.path
            rate_limit = self.get_rate_limit(endpoint, user_id)
            
            allowed, info = await self.backend.check_rate_limit(client_key, rate_limit)
            
            if not allowed:
                retry_after = int(info.get("retry_after", 60))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(info.get("limit", 0)),
                        "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                        "X-RateLimit-Reset": str(int(info.get("reset_time", 0)))
                    }
                )
            
            # Add rate limit headers to response (would need middleware for this)
            request.state.rate_limit_info = info
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request on error
            return True
    
    async def reset_user_limit(self, user_id: str) -> bool:
        """Reset rate limit for a specific user."""
        client_key = f"user:{user_id}"
        return await self.backend.reset_rate_limit(client_key)
    
    async def reset_ip_limit(self, ip_address: str) -> bool:
        """Reset rate limit for a specific IP."""
        client_key = f"ip:{ip_address}"
        return await self.backend.reset_rate_limit(client_key)
    
    def set_user_limit(self, user_id: str, requests: int, window: int):
        """Set custom rate limit for a user."""
        self.user_limits[user_id] = RateLimit(requests=requests, window=window)
        logger.info(f"Set custom rate limit for user {user_id}: {requests} requests per {window}s")
    
    def remove_user_limit(self, user_id: str):
        """Remove custom rate limit for a user."""
        if user_id in self.user_limits:
            del self.user_limits[user_id]
            logger.info(f"Removed custom rate limit for user {user_id}")
    
    def set_endpoint_limit(self, endpoint: str, requests: int, window: int):
        """Set custom rate limit for an endpoint."""
        self.endpoint_limits[endpoint] = RateLimit(requests=requests, window=window)
        logger.info(f"Set custom rate limit for endpoint {endpoint}: {requests} requests per {window}s")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        backend_stats = await self.backend.get_stats()
        
        return {
            "enabled": self.enabled,
            "default_limit": {
                "requests": self.default_limit.requests,
                "window": self.default_limit.window
            },
            "endpoint_limits": len(self.endpoint_limits),
            "user_limits": len(self.user_limits),
            "backend": backend_stats
        }
    
    def get_active_limits(self) -> Dict[str, Any]:
        """Get currently active rate limits."""
        return {
            "default": {
                "requests": self.default_limit.requests,
                "window": self.default_limit.window
            },
            "endpoints": {
                endpoint: {
                    "requests": limit.requests,
                    "window": limit.window
                }
                for endpoint, limit in self.endpoint_limits.items()
            },
            "users": {
                user_id: {
                    "requests": limit.requests,
                    "window": limit.window
                }
                for user_id, limit in self.user_limits.items()
            }
        }


# Global rate limiter instance
rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global rate_limiter
    if rate_limiter is None:
        # Create with default config for testing
        from ..config.hydra_config import APIConfig
        default_config = APIConfig()
        rate_limiter = RateLimiter(default_config)
    return rate_limiter


async def check_rate_limit(request: Request, user_id: Optional[str] = None) -> bool:
    """Dependency function to check rate limits."""
    limiter = get_rate_limiter()
    return await limiter.check(request, user_id) 