"""
API Monitoring

This module provides comprehensive monitoring and metrics collection
for the API with performance tracking and health monitoring.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..config.hydra_config import MonitoringConfig

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Request metrics data."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: float
    user_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System metrics data."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    load_average: List[float] = field(default_factory=list)


class APIMonitor:
    """API monitoring and metrics collection."""
    
    def __init__(self, monitoring_config: MonitoringConfig):
        self.config = monitoring_config
        self.enabled = monitoring_config.metrics_enabled
        
        # Request tracking
        self.request_history: deque = deque(maxlen=10000)  # Last 10k requests
        self.request_counts: Dict[str, int] = defaultdict(int)  # endpoint -> count
        self.error_counts: Dict[str, int] = defaultdict(int)  # endpoint -> error count
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System metrics
        self.system_metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Performance tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        
        # Monitoring tasks
        self._system_monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("APIMonitor initialized")
    
    def start(self):
        """Start monitoring tasks."""
        if not self.enabled or self._running:
            return
        
        self._running = True
        self._system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        
        logger.info("API monitoring started")
    
    def stop(self):
        """Stop monitoring tasks."""
        self._running = False
        
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
            self._system_monitor_task = None
        
        logger.info("API monitoring stopped")
    
    def log_request(self, endpoint: str, method: str = "GET", status_code: int = 200, 
                   response_time: float = 0.0, user_id: Optional[str] = None,
                   error_message: Optional[str] = None):
        """Log an API request."""
        if not self.enabled:
            return
        
        metrics = RequestMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            timestamp=time.time(),
            user_id=user_id,
            error_message=error_message
        )
        
        # Store metrics
        self.request_history.append(metrics)
        self.request_counts[endpoint] += 1
        self.total_requests += 1
        
        # Track errors
        if status_code >= 400:
            self.error_counts[endpoint] += 1
            self.total_errors += 1
        
        # Track response times
        self.response_times[endpoint].append(response_time)
        
        # Log slow requests
        if response_time > 5.0:  # 5 second threshold
            logger.warning(f"Slow request: {method} {endpoint} took {response_time:.2f}s")
    
    async def _system_monitor_loop(self):
        """System metrics monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Collect every minute
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            load_average = []
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage / 100.0,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        requests_per_minute = self._calculate_rate(self.request_history, 60)
        error_rate = self.total_errors / self.total_requests if self.total_requests > 0 else 0.0
        
        # Get recent system metrics
        latest_system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)
        
        # Get top endpoints by request count
        top_endpoints = sorted(
            self.request_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Get error statistics
        error_endpoints = [
            {"endpoint": endpoint, "errors": count, "rate": count / self.request_counts[endpoint]}
            for endpoint, count in self.error_counts.items()
            if count > 0
        ]
        
        metrics = {
            "timestamp": current_time,
            "uptime": uptime,
            "requests_per_minute": requests_per_minute,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "average_response_times": avg_response_times,
            "top_endpoints": top_endpoints,
            "error_endpoints": error_endpoints
        }
        
        # Add system metrics if available
        if latest_system_metrics:
            metrics.update({
                "cpu_usage": latest_system_metrics.cpu_usage,
                "memory_usage": latest_system_metrics.memory_usage,
                "disk_usage": latest_system_metrics.disk_usage,
                "process_count": latest_system_metrics.process_count,
                "load_average": latest_system_metrics.load_average,
                "network_io": latest_system_metrics.network_io
            })
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        metrics = self.get_metrics()
        current_time = time.time()
        
        # Health indicators
        health_issues = []
        status = "healthy"
        
        # Check error rate
        if metrics["error_rate"] > 0.1:  # 10% error rate
            health_issues.append("High error rate")
            status = "degraded"
        
        # Check CPU usage
        if metrics.get("cpu_usage", 0) > 0.8:  # 80% CPU
            health_issues.append("High CPU usage")
            status = "degraded"
        
        # Check memory usage
        if metrics.get("memory_usage", 0) > 0.9:  # 90% memory
            health_issues.append("High memory usage")
            status = "unhealthy"
        
        # Check disk usage
        if metrics.get("disk_usage", 0) > 0.9:  # 90% disk
            health_issues.append("High disk usage")
            status = "degraded"
        
        # Check if we're getting requests (if uptime > 5 minutes)
        if metrics["uptime"] > 300 and metrics["requests_per_minute"] == 0:
            health_issues.append("No recent requests")
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": current_time,
            "issues": health_issues,
            "uptime": metrics["uptime"],
            "total_requests": metrics["total_requests"],
            "error_rate": metrics["error_rate"]
        }
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint."""
        total_requests = self.request_counts.get(endpoint, 0)
        total_errors = self.error_counts.get(endpoint, 0)
        response_times = list(self.response_times.get(endpoint, []))
        
        stats = {
            "endpoint": endpoint,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0.0,
            "response_times": {
                "count": len(response_times),
                "average": sum(response_times) / len(response_times) if response_times else 0.0,
                "min": min(response_times) if response_times else 0.0,
                "max": max(response_times) if response_times else 0.0
            }
        }
        
        # Calculate percentiles
        if response_times:
            sorted_times = sorted(response_times)
            length = len(sorted_times)
            stats["response_times"].update({
                "p50": sorted_times[int(length * 0.5)],
                "p90": sorted_times[int(length * 0.9)],
                "p95": sorted_times[int(length * 0.95)],
                "p99": sorted_times[int(length * 0.99)] if length > 10 else sorted_times[-1]
            })
        
        return stats
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        user_requests = [
            req for req in self.request_history 
            if req.user_id == user_id
        ]
        
        if not user_requests:
            return {
                "user_id": user_id,
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "average_response_time": 0.0
            }
        
        total_requests = len(user_requests)
        total_errors = sum(1 for req in user_requests if req.status_code >= 400)
        avg_response_time = sum(req.response_time for req in user_requests) / total_requests
        
        # Recent activity (last hour)
        recent_time = time.time() - 3600
        recent_requests = [req for req in user_requests if req.timestamp > recent_time]
        
        return {
            "user_id": user_id,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests,
            "average_response_time": avg_response_time,
            "recent_requests": len(recent_requests),
            "first_request": min(req.timestamp for req in user_requests),
            "last_request": max(req.timestamp for req in user_requests)
        }
    
    def _calculate_rate(self, history: deque, window_seconds: int) -> float:
        """Calculate request rate over a time window."""
        if not history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_requests = sum(
            1 for item in history 
            if item.timestamp > cutoff_time
        )
        
        return recent_requests / (window_seconds / 60)  # requests per minute
    
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        metrics = self.get_metrics()
        
        # High error rate alert
        if metrics["error_rate"] > 0.05:  # 5% error rate
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning" if metrics["error_rate"] < 0.1 else "critical",
                "message": f"Error rate is {metrics['error_rate']:.2%}",
                "value": metrics["error_rate"],
                "threshold": 0.05
            })
        
        # High CPU alert
        if metrics.get("cpu_usage", 0) > 0.7:  # 70% CPU
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning" if metrics["cpu_usage"] < 0.9 else "critical",
                "message": f"CPU usage is {metrics['cpu_usage']:.1%}",
                "value": metrics["cpu_usage"],
                "threshold": 0.7
            })
        
        # High memory alert
        if metrics.get("memory_usage", 0) > 0.8:  # 80% memory
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning" if metrics["memory_usage"] < 0.9 else "critical",
                "message": f"Memory usage is {metrics['memory_usage']:.1%}",
                "value": metrics["memory_usage"],
                "threshold": 0.8
            })
        
        # Slow response time alert
        for endpoint, avg_time in metrics["average_response_times"].items():
            if avg_time > 2.0:  # 2 second threshold
                alerts.append({
                    "type": "slow_response",
                    "severity": "warning",
                    "message": f"Slow response time for {endpoint}: {avg_time:.2f}s",
                    "endpoint": endpoint,
                    "value": avg_time,
                    "threshold": 2.0
                })
        
        return alerts
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        metrics = self.get_metrics()
        
        if format == "json":
            import json
            return json.dumps(metrics, indent=2)
        
        elif format == "prometheus":
            # Export in Prometheus format
            lines = []
            
            # Basic metrics
            lines.append(f"# HELP api_requests_total Total number of API requests")
            lines.append(f"# TYPE api_requests_total counter")
            lines.append(f"api_requests_total {metrics['total_requests']}")
            
            lines.append(f"# HELP api_errors_total Total number of API errors")
            lines.append(f"# TYPE api_errors_total counter")
            lines.append(f"api_errors_total {metrics['total_errors']}")
            
            lines.append(f"# HELP api_requests_per_minute Current requests per minute")
            lines.append(f"# TYPE api_requests_per_minute gauge")
            lines.append(f"api_requests_per_minute {metrics['requests_per_minute']}")
            
            # System metrics
            if "cpu_usage" in metrics:
                lines.append(f"# HELP system_cpu_usage Current CPU usage")
                lines.append(f"# TYPE system_cpu_usage gauge")
                lines.append(f"system_cpu_usage {metrics['cpu_usage']}")
            
            if "memory_usage" in metrics:
                lines.append(f"# HELP system_memory_usage Current memory usage")
                lines.append(f"# TYPE system_memory_usage gauge")
                lines.append(f"system_memory_usage {metrics['memory_usage']}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.request_history.clear()
        self.request_counts.clear()
        self.error_counts.clear()
        self.response_times.clear()
        self.system_metrics_history.clear()
        
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        
        logger.info("Metrics reset") 