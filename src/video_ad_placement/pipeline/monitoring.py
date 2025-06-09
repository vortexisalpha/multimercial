"""
Pipeline Monitoring and Metrics System

This module provides comprehensive monitoring capabilities for the video processing pipeline,
including metrics collection, health checking, performance profiling, and alerting.
"""

import time
import threading
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import asyncio
import psutil
import torch

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    message: str
    timestamp: float
    component: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: float
    component: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status of a pipeline component."""
    component_name: str
    status: HealthStatus
    last_update: float
    error_count: int = 0
    warning_count: int = 0
    uptime_seconds: float = 0.0
    performance_score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history = deque(maxlen=max_history_size)
        self.metrics_by_component = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_metrics = defaultdict(list)
        self.collection_lock = threading.RLock()
        
        # Real-time metrics
        self.current_metrics = {}
        self.metric_callbacks = defaultdict(list)
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self.collection_lock:
            self.metrics_history.append(metric)
            self.metrics_by_component[metric.component].append(metric)
            self.current_metrics[f"{metric.component}.{metric.name}"] = metric
            
            # Trigger callbacks
            for callback in self.metric_callbacks[metric.name]:
                try:
                    callback(metric)
                except Exception as e:
                    logger.error(f"Metric callback failed: {e}")
    
    def record_simple_metric(self, name: str, value: float, component: str, tags: Dict[str, str] = None):
        """Record a simple metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            component=component,
            tags=tags or {}
        )
        self.record_metric(metric)
    
    def get_metric_history(self, component: str, metric_name: str, 
                          time_window_seconds: Optional[float] = None) -> List[PerformanceMetric]:
        """Get metric history for a component."""
        with self.collection_lock:
            component_metrics = list(self.metrics_by_component[component])
        
        # Filter by metric name
        filtered_metrics = [m for m in component_metrics if m.name == metric_name]
        
        # Apply time window if specified
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]
        
        return sorted(filtered_metrics, key=lambda m: m.timestamp)
    
    def get_metric_stats(self, component: str, metric_name: str, 
                        time_window_seconds: float = 300.0) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(component, metric_name, time_window_seconds)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def register_metric_callback(self, metric_name: str, callback: Callable[[PerformanceMetric], None]):
        """Register callback for metric updates."""
        self.metric_callbacks[metric_name].append(callback)
    
    def get_current_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get current metric values."""
        with self.collection_lock:
            return dict(self.current_metrics)
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        with self.collection_lock:
            metrics_data = [
                {
                    'name': m.name,
                    'value': m.value,
                    'timestamp': m.timestamp,
                    'component': m.component,
                    'tags': m.tags
                }
                for m in self.metrics_history
            ]
        
        if format_type == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


class HealthChecker:
    """Monitors component health and system status."""
    
    def __init__(self, check_interval_seconds: float = 10.0):
        self.check_interval = check_interval_seconds
        self.component_health = {}
        self.health_lock = threading.RLock()
        self.health_callbacks = []
        
        # Health checking
        self.health_checks = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Thresholds
        self.error_threshold = 5  # errors per minute
        self.warning_threshold = 3  # warnings per minute
        self.response_time_threshold = 5.0  # seconds
        
    def register_component(self, component_name: str, 
                          health_check_func: Optional[Callable[[], bool]] = None):
        """Register a component for health monitoring."""
        with self.health_lock:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.HEALTHY,
                last_update=time.time()
            )
            
            if health_check_func:
                self.health_checks[component_name] = health_check_func
    
    def update_component_health(self, component_name: str, status: HealthStatus, 
                               details: Dict[str, Any] = None):
        """Update component health status."""
        with self.health_lock:
            if component_name not in self.component_health:
                self.register_component(component_name)
            
            health = self.component_health[component_name]
            old_status = health.status
            
            health.status = status
            health.last_update = time.time()
            if details:
                health.details.update(details)
            
            # Trigger callbacks if status changed
            if old_status != status:
                for callback in self.health_callbacks:
                    try:
                        callback(component_name, old_status, status)
                    except Exception as e:
                        logger.error(f"Health callback failed: {e}")
    
    def record_component_error(self, component_name: str, error_message: str):
        """Record an error for a component."""
        with self.health_lock:
            if component_name not in self.component_health:
                self.register_component(component_name)
            
            health = self.component_health[component_name]
            health.error_count += 1
            health.details['last_error'] = error_message
            health.details['last_error_time'] = time.time()
            
            # Update status based on error rate
            if health.error_count > self.error_threshold:
                self.update_component_health(component_name, HealthStatus.UNHEALTHY)
            elif health.error_count > self.warning_threshold:
                self.update_component_health(component_name, HealthStatus.WARNING)
    
    def record_component_warning(self, component_name: str, warning_message: str):
        """Record a warning for a component."""
        with self.health_lock:
            if component_name not in self.component_health:
                self.register_component(component_name)
            
            health = self.component_health[component_name]
            health.warning_count += 1
            health.details['last_warning'] = warning_message
            health.details['last_warning_time'] = time.time()
            
            # Update status if too many warnings
            if health.warning_count > self.warning_threshold * 2:
                self.update_component_health(component_name, HealthStatus.WARNING)
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self.health_lock:
            if not self.component_health:
                return HealthStatus.HEALTHY
            
            statuses = [h.status for h in self.component_health.values()]
            
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            elif HealthStatus.WARNING in statuses:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        with self.health_lock:
            return self.component_health.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        with self.health_lock:
            return dict(self.component_health)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Health monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                with self.health_lock:
                    for component_name, health_check in self.health_checks.items():
                        try:
                            is_healthy = health_check()
                            if is_healthy:
                                self.update_component_health(component_name, HealthStatus.HEALTHY)
                            else:
                                self.update_component_health(component_name, HealthStatus.UNHEALTHY)
                        except Exception as e:
                            logger.error(f"Health check failed for {component_name}: {e}")
                            self.record_component_error(component_name, str(e))
                
                # Reset error/warning counters periodically
                self._reset_counters()
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def _reset_counters(self):
        """Reset error and warning counters periodically."""
        current_time = time.time()
        
        with self.health_lock:
            for health in self.component_health.values():
                # Reset counters if enough time has passed
                if current_time - health.last_update > 60.0:  # 1 minute
                    health.error_count = max(0, health.error_count - 1)
                    health.warning_count = max(0, health.warning_count - 1)
    
    def register_health_callback(self, callback: Callable[[str, HealthStatus, HealthStatus], None]):
        """Register callback for health status changes."""
        self.health_callbacks.append(callback)


class PerformanceProfiler:
    """Profiles performance of pipeline components."""
    
    def __init__(self):
        self.profiles = defaultdict(lambda: defaultdict(list))
        self.profiling_lock = threading.RLock()
        self.active_profiles = {}
        
    def start_profile(self, component: str, operation: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{component}_{operation}_{time.time()}"
        
        with self.profiling_lock:
            self.active_profiles[profile_id] = {
                'component': component,
                'operation': operation,
                'start_time': time.time(),
                'start_memory': self._get_memory_usage(),
                'start_gpu_memory': self._get_gpu_memory_usage()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End profiling and record results."""
        with self.profiling_lock:
            if profile_id not in self.active_profiles:
                return {}
            
            profile_data = self.active_profiles.pop(profile_id)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            result = {
                'component': profile_data['component'],
                'operation': profile_data['operation'],
                'duration_seconds': end_time - profile_data['start_time'],
                'memory_delta_mb': end_memory - profile_data['start_memory'],
                'gpu_memory_delta_mb': end_gpu_memory - profile_data['start_gpu_memory'],
                'timestamp': end_time
            }
            
            # Store result
            component = profile_data['component']
            operation = profile_data['operation']
            self.profiles[component][operation].append(result)
            
            # Keep only recent profiles
            if len(self.profiles[component][operation]) > 1000:
                self.profiles[component][operation] = self.profiles[component][operation][-500:]
            
            return result
    
    def get_profile_stats(self, component: str, operation: str) -> Dict[str, Any]:
        """Get profiling statistics for a component operation."""
        with self.profiling_lock:
            profiles = self.profiles[component][operation]
        
        if not profiles:
            return {}
        
        durations = [p['duration_seconds'] for p in profiles]
        memory_deltas = [p['memory_delta_mb'] for p in profiles]
        gpu_memory_deltas = [p['gpu_memory_delta_mb'] for p in profiles]
        
        return {
            'count': len(profiles),
            'duration_stats': {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'p95': statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
            },
            'memory_stats': {
                'mean_delta_mb': statistics.mean(memory_deltas),
                'max_delta_mb': max(memory_deltas),
                'min_delta_mb': min(memory_deltas)
            },
            'gpu_memory_stats': {
                'mean_delta_mb': statistics.mean(gpu_memory_deltas),
                'max_delta_mb': max(gpu_memory_deltas),
                'min_delta_mb': min(gpu_memory_deltas)
            }
        }
    
    def get_all_profile_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get profiling statistics for all components."""
        stats = {}
        
        with self.profiling_lock:
            for component, operations in self.profiles.items():
                stats[component] = {}
                for operation in operations:
                    stats[component][operation] = self.get_profile_stats(component, operation)
        
        return stats
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.alert_callbacks = defaultdict(list)
        self.alert_lock = threading.RLock()
        
        # Alert suppression
        self.suppression_rules = {}
        self.last_alert_times = defaultdict(float)
        self.suppression_window = 60.0  # seconds
    
    def create_alert(self, level: AlertLevel, message: str, component: str,
                    metric_name: Optional[str] = None, metric_value: Optional[float] = None,
                    threshold: Optional[float] = None):
        """Create and process an alert."""
        alert = Alert(
            level=level,
            message=message,
            timestamp=time.time(),
            component=component,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        # Check suppression
        if self._should_suppress_alert(alert):
            return
        
        with self.alert_lock:
            self.alerts.append(alert)
            self.last_alert_times[f"{component}_{message}"] = alert.timestamp
        
        # Trigger callbacks
        for callback in self.alert_callbacks[level]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        logger.log(log_level, f"ALERT [{level.value.upper()}] {component}: {message}")
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        alert_key = f"{alert.component}_{alert.message}"
        last_time = self.last_alert_times.get(alert_key, 0)
        
        return (alert.timestamp - last_time) < self.suppression_window
    
    def get_recent_alerts(self, time_window_seconds: float = 3600.0,
                         level_filter: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts within time window."""
        cutoff_time = time.time() - time_window_seconds
        
        with self.alert_lock:
            filtered_alerts = [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
        
        if level_filter:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.level == level_filter
            ]
        
        return sorted(filtered_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self, time_window_seconds: float = 3600.0) -> Dict[str, int]:
        """Get summary of alerts by level."""
        recent_alerts = self.get_recent_alerts(time_window_seconds)
        
        summary = {level.value: 0 for level in AlertLevel}
        
        for alert in recent_alerts:
            summary[alert.level.value] += 1
        
        return summary
    
    def register_alert_callback(self, level: AlertLevel, callback: Callable[[Alert], None]):
        """Register callback for specific alert level."""
        self.alert_callbacks[level].append(callback)
    
    def add_suppression_rule(self, component: str, message_pattern: str, duration_seconds: float):
        """Add alert suppression rule."""
        self.suppression_rules[f"{component}_{message_pattern}"] = duration_seconds


class PipelineMonitor:
    """Main pipeline monitoring coordinator."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize subsystems
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(config.health_check_interval)
        self.profiler = PerformanceProfiler()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = None
        self.processing_start_time = None
        
        # Progress tracking
        self.current_progress = 0.0
        self.estimated_completion_time = None
        
        logger.info("PipelineMonitor initialized")
    
    def start_monitoring(self, video_info, processing_estimate):
        """Start comprehensive monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.processing_start_time = time.time()
        
        # Start subsystems
        self.health_checker.start_monitoring()
        
        # Register components
        self._register_pipeline_components()
        
        # Set up metric thresholds
        self._setup_metric_thresholds()
        
        # Set up alerts
        self._setup_alert_callbacks()
        
        logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        self.health_checker.stop_monitoring()
        logger.info("Pipeline monitoring stopped")
    
    def update_progress(self, progress: float):
        """Update processing progress."""
        self.current_progress = progress
        
        # Estimate completion time
        if progress > 0 and self.processing_start_time:
            elapsed_time = time.time() - self.processing_start_time
            estimated_total_time = elapsed_time / progress
            self.estimated_completion_time = self.processing_start_time + estimated_total_time
        
        # Record progress metric
        self.metrics_collector.record_simple_metric(
            "progress", progress, "pipeline"
        )
    
    def update_frame_metrics(self, frame_result):
        """Update metrics from frame processing result."""
        component = "frame_processor"
        
        # Record processing time
        self.metrics_collector.record_simple_metric(
            "processing_time_seconds", frame_result.processing_time, component
        )
        
        # Record quality score
        self.metrics_collector.record_simple_metric(
            "quality_score", frame_result.quality_score, component
        )
        
        # Record success/failure
        self.metrics_collector.record_simple_metric(
            "success_rate", 1.0 if frame_result.success else 0.0, component
        )
        
        # Update health based on success
        if frame_result.success:
            self.health_checker.update_component_health(component, HealthStatus.HEALTHY)
        else:
            self.health_checker.record_component_error(component, frame_result.error_message or "Unknown error")
    
    def get_performance_profile(self) -> Dict[str, Any]:
        """Get comprehensive performance profile."""
        current_time = time.time()
        
        return {
            'monitoring_duration_seconds': current_time - self.start_time if self.start_time else 0,
            'processing_duration_seconds': current_time - self.processing_start_time if self.processing_start_time else 0,
            'current_progress': self.current_progress,
            'estimated_completion_time': self.estimated_completion_time,
            'health_status': self.health_checker.get_overall_health().value,
            'component_health': {
                name: health.status.value 
                for name, health in self.health_checker.get_all_component_health().items()
            },
            'metric_stats': self._get_key_metric_stats(),
            'profiling_stats': self.profiler.get_all_profile_stats(),
            'recent_alerts': len(self.alert_manager.get_recent_alerts(300.0))  # 5 minutes
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'overall_status': self.health_checker.get_overall_health().value,
            'components': {
                name: {
                    'status': health.status.value,
                    'error_count': health.error_count,
                    'warning_count': health.warning_count,
                    'last_update': health.last_update
                }
                for name, health in self.health_checker.get_all_component_health().items()
            },
            'alert_summary': self.alert_manager.get_alert_summary(3600.0)  # 1 hour
        }
    
    def _register_pipeline_components(self):
        """Register all pipeline components for monitoring."""
        components = [
            "depth_estimator",
            "object_detector", 
            "plane_detector",
            "temporal_tracker",
            "consistency_manager",
            "video_renderer",
            "frame_processor",
            "pipeline"
        ]
        
        for component in components:
            self.health_checker.register_component(component)
    
    def _setup_metric_thresholds(self):
        """Set up metric monitoring thresholds."""
        
        # Processing time thresholds
        def check_processing_time(metric):
            if metric.value > 10.0:  # 10 seconds per frame
                self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    f"High processing time: {metric.value:.2f}s",
                    metric.component,
                    metric.name,
                    metric.value,
                    10.0
                )
        
        self.metrics_collector.register_metric_callback("processing_time_seconds", check_processing_time)
        
        # Quality score thresholds
        def check_quality_score(metric):
            if metric.value < 0.5:  # Low quality threshold
                self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    f"Low quality score: {metric.value:.2f}",
                    metric.component,
                    metric.name,
                    metric.value,
                    0.5
                )
        
        self.metrics_collector.register_metric_callback("quality_score", check_quality_score)
    
    def _setup_alert_callbacks(self):
        """Set up alert handling callbacks."""
        
        def handle_critical_alert(alert):
            logger.critical(f"CRITICAL ALERT: {alert.component} - {alert.message}")
            # In production, would send notifications, page on-call, etc.
        
        def handle_error_alert(alert):
            logger.error(f"ERROR ALERT: {alert.component} - {alert.message}")
        
        self.alert_manager.register_alert_callback(AlertLevel.CRITICAL, handle_critical_alert)
        self.alert_manager.register_alert_callback(AlertLevel.ERROR, handle_error_alert)
    
    def _get_key_metric_stats(self) -> Dict[str, Any]:
        """Get statistics for key metrics."""
        stats = {}
        
        # Processing time stats
        processing_stats = self.metrics_collector.get_metric_stats(
            "frame_processor", "processing_time_seconds", 300.0
        )
        if processing_stats:
            stats['processing_time'] = processing_stats
        
        # Quality score stats
        quality_stats = self.metrics_collector.get_metric_stats(
            "frame_processor", "quality_score", 300.0
        )
        if quality_stats:
            stats['quality_score'] = quality_stats
        
        # Success rate stats
        success_stats = self.metrics_collector.get_metric_stats(
            "frame_processor", "success_rate", 300.0
        )
        if success_stats:
            stats['success_rate'] = success_stats
        
        return stats 