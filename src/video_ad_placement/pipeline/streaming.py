"""
Streaming Video Processing System

This module provides comprehensive streaming capabilities for real-time video processing,
including video streams, buffering, and real-time processing with temporal consistency.
"""

import asyncio
import cv2
import torch
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import queue
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of video streams."""
    FILE = "file"              # File-based video
    CAMERA = "camera"          # Camera input
    NETWORK = "network"        # Network stream (RTMP, WebRTC, etc.)
    MEMORY = "memory"          # In-memory stream
    PIPE = "pipe"              # Named pipe


class StreamState(Enum):
    """Video stream states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EOF = "eof"


class BufferStrategy(Enum):
    """Buffer management strategies."""
    FIFO = "fifo"              # First in, first out
    DROP_OLD = "drop_old"      # Drop oldest frames when full
    DROP_NEW = "drop_new"      # Drop newest frames when full
    ADAPTIVE = "adaptive"      # Adaptive based on processing speed


@dataclass
class StreamInfo:
    """Video stream information."""
    stream_type: StreamType
    width: int
    height: int
    fps: float
    total_frames: Optional[int] = None
    duration: Optional[float] = None
    codec: Optional[str] = None
    pixel_format: str = "bgr24"
    has_audio: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Configuration for video streams."""
    buffer_size: int = 30
    buffer_strategy: BufferStrategy = BufferStrategy.DROP_OLD
    max_buffer_memory_mb: float = 512.0
    prefetch_frames: int = 5
    enable_threading: bool = True
    target_fps: Optional[float] = None
    quality_priority: bool = False  # True for quality, False for speed
    
    # Real-time processing
    max_latency_ms: float = 100.0
    enable_frame_skipping: bool = True
    adaptive_quality: bool = True
    
    # Network streaming
    connection_timeout: float = 10.0
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame_id: int
    timestamp: float
    frame: np.ndarray
    pts: Optional[float] = None  # Presentation timestamp
    dts: Optional[float] = None  # Decode timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_size(self) -> int:
        """Calculate memory size of frame data."""
        return self.frame.nbytes if self.frame is not None else 0


class VideoStream:
    """Async video stream with buffering and real-time capabilities."""
    
    def __init__(self, source: Union[str, int], mode: str = 'read', 
                 config: Optional[StreamConfig] = None):
        """
        Initialize video stream.
        
        Args:
            source: Video source (file path, camera index, or URL)
            mode: Stream mode ('read' or 'write')
            config: Stream configuration
        """
        self.source = source
        self.mode = mode
        self.config = config or StreamConfig()
        
        # Stream state
        self.state = StreamState.STOPPED
        self.info: Optional[StreamInfo] = None
        self.error_message: Optional[str] = None
        
        # OpenCV objects
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        
        # Buffering
        self.frame_buffer = deque(maxlen=self.config.buffer_size)
        self.buffer_lock = threading.RLock()
        self.current_memory_usage = 0.0
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self.write_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Frame tracking
        self.frame_counter = 0
        self.dropped_frames = 0
        self.last_frame_time = 0.0
        
        # Callbacks
        self.frame_callbacks: List[Callable[[FrameData], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        logger.info(f"VideoStream initialized: {source} ({mode})")
    
    async def start(self):
        """Start the video stream."""
        if self.state != StreamState.STOPPED:
            return
        
        self.state = StreamState.STARTING
        
        try:
            if self.mode == 'read':
                await self._start_reading()
            elif self.mode == 'write':
                await self._start_writing()
            else:
                raise ValueError(f"Invalid stream mode: {self.mode}")
            
            self.state = StreamState.RUNNING
            logger.info(f"VideoStream started: {self.source}")
            
        except Exception as e:
            self.state = StreamState.ERROR
            self.error_message = str(e)
            logger.error(f"Failed to start VideoStream: {e}")
            raise
    
    async def stop(self):
        """Stop the video stream."""
        if self.state == StreamState.STOPPED:
            return
        
        logger.info(f"Stopping VideoStream: {self.source}")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=5.0)
        
        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=5.0)
        
        # Close OpenCV objects
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        # Clear buffer
        with self.buffer_lock:
            self.frame_buffer.clear()
            self.current_memory_usage = 0.0
        
        self.state = StreamState.STOPPED
        logger.info(f"VideoStream stopped: {self.source}")
    
    async def pause(self):
        """Pause the video stream."""
        if self.state == StreamState.RUNNING:
            self.state = StreamState.PAUSED
    
    async def resume(self):
        """Resume the video stream."""
        if self.state == StreamState.PAUSED:
            self.state = StreamState.RUNNING
    
    async def _start_reading(self):
        """Start reading from video source."""
        # Initialize video capture
        if isinstance(self.source, str):
            self.cap = cv2.VideoCapture(self.source)
            stream_type = StreamType.FILE if Path(self.source).exists() else StreamType.NETWORK
        else:
            self.cap = cv2.VideoCapture(self.source)
            stream_type = StreamType.CAMERA
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        # Extract stream info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.info = StreamInfo(
            stream_type=stream_type,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames if total_frames > 0 else None,
            duration=total_frames / fps if fps > 0 and total_frames > 0 else None
        )
        
        # Start reading thread if enabled
        if self.config.enable_threading:
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
    
    async def _start_writing(self):
        """Start writing to video output."""
        # Note: Writing would be initialized when first frame is written
        # to get proper dimensions and format
        pass
    
    def _read_loop(self):
        """Background thread for reading frames."""
        frame_interval = 1.0 / self.config.target_fps if self.config.target_fps else 0
        
        while not self.stop_event.is_set() and self.state in [StreamState.RUNNING, StreamState.PAUSED]:
            
            if self.state == StreamState.PAUSED:
                time.sleep(0.1)
                continue
            
            try:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    # End of stream
                    self.state = StreamState.EOF
                    break
                
                # Create frame data
                frame_data = FrameData(
                    frame_id=self.frame_counter,
                    timestamp=time.time(),
                    frame=frame.copy(),
                    pts=self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )
                
                # Add to buffer
                self._add_to_buffer(frame_data)
                
                # Update counters
                self.frame_counter += 1
                self.last_frame_time = time.time()
                
                # Trigger callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame_data)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                # Frame rate limiting
                if frame_interval > 0:
                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                self.state = StreamState.ERROR
                self.error_message = str(e)
                
                # Trigger error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(e)
                    except Exception:
                        pass
                
                break
    
    def _add_to_buffer(self, frame_data: FrameData):
        """Add frame to buffer with strategy handling."""
        with self.buffer_lock:
            # Check memory limit
            frame_memory = frame_data.memory_size / (1024 * 1024)  # MB
            
            if self.current_memory_usage + frame_memory > self.config.max_buffer_memory_mb:
                if self.config.buffer_strategy == BufferStrategy.DROP_NEW:
                    self.dropped_frames += 1
                    return
                elif self.config.buffer_strategy == BufferStrategy.DROP_OLD:
                    while (self.frame_buffer and 
                           self.current_memory_usage + frame_memory > self.config.max_buffer_memory_mb):
                        old_frame = self.frame_buffer.popleft()
                        self.current_memory_usage -= old_frame.memory_size / (1024 * 1024)
                        self.dropped_frames += 1
            
            # Add frame to buffer
            if len(self.frame_buffer) >= self.config.buffer_size:
                if self.config.buffer_strategy == BufferStrategy.FIFO:
                    old_frame = self.frame_buffer.popleft()
                    self.current_memory_usage -= old_frame.memory_size / (1024 * 1024)
                elif self.config.buffer_strategy == BufferStrategy.DROP_NEW:
                    self.dropped_frames += 1
                    return
            
            self.frame_buffer.append(frame_data)
            self.current_memory_usage += frame_memory
    
    async def read_frame(self) -> Optional[FrameData]:
        """Read next frame from stream."""
        if self.mode != 'read':
            raise RuntimeError("Stream not in read mode")
        
        if not self.config.enable_threading:
            # Synchronous reading
            if not self.cap or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            if not ret:
                self.state = StreamState.EOF
                return None
            
            frame_data = FrameData(
                frame_id=self.frame_counter,
                timestamp=time.time(),
                frame=frame,
                pts=self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )
            
            self.frame_counter += 1
            return frame_data
        else:
            # Asynchronous reading from buffer
            while self.state in [StreamState.RUNNING, StreamState.PAUSED]:
                with self.buffer_lock:
                    if self.frame_buffer:
                        frame_data = self.frame_buffer.popleft()
                        self.current_memory_usage -= frame_data.memory_size / (1024 * 1024)
                        return frame_data
                
                if self.state == StreamState.EOF:
                    break
                
                # Wait for frames
                await asyncio.sleep(0.01)
            
            return None
    
    async def write_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Write frame to output stream."""
        if self.mode != 'write':
            raise RuntimeError("Stream not in write mode")
        
        # Initialize writer if needed
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.config.target_fps or 30.0
            
            self.writer = cv2.VideoWriter(
                str(self.source), fourcc, fps, (width, height)
            )
            
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.source}")
            
            self.info = StreamInfo(
                stream_type=StreamType.FILE,
                width=width,
                height=height,
                fps=fps
            )
        
        # Write frame
        self.writer.write(frame)
        self.frame_counter += 1
    
    async def __aiter__(self) -> AsyncIterator[FrameData]:
        """Async iterator for reading frames."""
        if self.mode != 'read':
            raise RuntimeError("Stream not in read mode")
        
        while self.state != StreamState.STOPPED:
            frame_data = await self.read_frame()
            if frame_data is None:
                break
            yield frame_data
    
    def register_frame_callback(self, callback: Callable[[FrameData], None]):
        """Register callback for new frames."""
        self.frame_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception], None]):
        """Register callback for errors."""
        self.error_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        with self.buffer_lock:
            buffer_frames = len(self.frame_buffer)
            buffer_memory = self.current_memory_usage
        
        return {
            'state': self.state.value,
            'frame_counter': self.frame_counter,
            'dropped_frames': self.dropped_frames,
            'buffer_frames': buffer_frames,
            'buffer_memory_mb': buffer_memory,
            'last_frame_time': self.last_frame_time,
            'drop_rate': self.dropped_frames / max(1, self.frame_counter + self.dropped_frames)
        }
    
    async def close(self):
        """Close the video stream."""
        await self.stop()


class BufferManager:
    """Manages multiple video buffers with memory optimization."""
    
    def __init__(self, config):
        self.config = config
        self.buffers: Dict[str, deque] = {}
        self.buffer_configs: Dict[str, Any] = {}
        self.memory_usage = 0.0
        self.max_memory_mb = getattr(config, 'max_buffer_memory_mb', 1024.0)
        
        # Buffer monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.buffer_lock = threading.RLock()
        
        logger.info("BufferManager initialized")
    
    def initialize_buffers(self, stream_info: StreamInfo):
        """Initialize buffers based on stream information."""
        # Calculate optimal buffer sizes based on stream characteristics
        frame_size_mb = (stream_info.width * stream_info.height * 3) / (1024 * 1024)
        max_frames_per_buffer = int(self.max_memory_mb // frame_size_mb // 4)  # Conservative estimate
        
        buffer_config = {
            'max_size': max(10, min(100, max_frames_per_buffer)),
            'frame_size_mb': frame_size_mb,
            'fps': stream_info.fps
        }
        
        with self.buffer_lock:
            for buffer_name in ['input', 'processing', 'output']:
                self.buffers[buffer_name] = deque(maxlen=buffer_config['max_size'])
                self.buffer_configs[buffer_name] = buffer_config.copy()
        
        # Start monitoring
        if not self.monitoring_active:
            self.start_monitoring()
        
        logger.info(f"Buffers initialized for {stream_info.width}x{stream_info.height} stream")
    
    def add_frame(self, buffer_name: str, frame_data: FrameData) -> bool:
        """Add frame to specified buffer."""
        with self.buffer_lock:
            if buffer_name not in self.buffers:
                return False
            
            buffer = self.buffers[buffer_name]
            config = self.buffer_configs[buffer_name]
            
            # Check memory limits
            frame_memory = frame_data.memory_size / (1024 * 1024)
            
            if self.memory_usage + frame_memory > self.max_memory_mb:
                # Try to free space by removing old frames
                self._cleanup_old_frames()
                
                if self.memory_usage + frame_memory > self.max_memory_mb:
                    return False  # Cannot add frame
            
            # Add frame
            if len(buffer) >= buffer.maxlen:
                # Remove oldest frame
                old_frame = buffer.popleft()
                self.memory_usage -= old_frame.memory_size / (1024 * 1024)
            
            buffer.append(frame_data)
            self.memory_usage += frame_memory
            
            return True
    
    def get_frame(self, buffer_name: str) -> Optional[FrameData]:
        """Get frame from specified buffer."""
        with self.buffer_lock:
            if buffer_name not in self.buffers:
                return None
            
            buffer = self.buffers[buffer_name]
            
            if buffer:
                frame_data = buffer.popleft()
                self.memory_usage -= frame_data.memory_size / (1024 * 1024)
                return frame_data
            
            return None
    
    def get_buffer_size(self, buffer_name: str) -> int:
        """Get current size of buffer."""
        with self.buffer_lock:
            return len(self.buffers.get(buffer_name, []))
    
    def clear_buffer(self, buffer_name: str):
        """Clear specified buffer."""
        with self.buffer_lock:
            if buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                
                # Update memory usage
                for frame_data in buffer:
                    self.memory_usage -= frame_data.memory_size / (1024 * 1024)
                
                buffer.clear()
    
    def _cleanup_old_frames(self):
        """Remove old frames to free memory."""
        # Simple cleanup: remove frames from each buffer in round-robin fashion
        for buffer_name in self.buffers:
            buffer = self.buffers[buffer_name]
            if buffer:
                old_frame = buffer.popleft()
                self.memory_usage -= old_frame.memory_size / (1024 * 1024)
                break
    
    def start_monitoring(self):
        """Start buffer monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Buffer monitoring started")
    
    def stop_monitoring(self):
        """Stop buffer monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Buffer monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                with self.buffer_lock:
                    # Check memory usage
                    memory_usage_percent = (self.memory_usage / self.max_memory_mb) * 100
                    
                    if memory_usage_percent > 90:
                        logger.warning(f"High buffer memory usage: {memory_usage_percent:.1f}%")
                        self._cleanup_old_frames()
                    
                    # Log buffer statistics
                    buffer_stats = {
                        name: len(buffer) for name, buffer in self.buffers.items()
                    }
                    
                    logger.debug(f"Buffer stats: {buffer_stats}, Memory: {self.memory_usage:.1f}MB")
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Buffer monitoring error: {e}")
    
    def cleanup_buffers(self):
        """Clean up all buffers."""
        with self.buffer_lock:
            for buffer_name in list(self.buffers.keys()):
                self.clear_buffer(buffer_name)
            
            self.buffers.clear()
            self.buffer_configs.clear()
            self.memory_usage = 0.0
        
        self.stop_monitoring()
        logger.info("Buffers cleaned up")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self.buffer_lock:
            buffer_breakdown = {}
            for name, buffer in self.buffers.items():
                buffer_memory = sum(frame.memory_size for frame in buffer) / (1024 * 1024)
                buffer_breakdown[name] = {
                    'frames': len(buffer),
                    'memory_mb': buffer_memory
                }
            
            return {
                'total_memory_mb': self.memory_usage,
                'max_memory_mb': self.max_memory_mb,
                'usage_percent': (self.memory_usage / self.max_memory_mb) * 100,
                'buffer_breakdown': buffer_breakdown
            }


class StreamProcessor:
    """Base class for stream processors."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_processing = False
        self.stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'processing_time_total': 0.0,
            'errors': 0
        }
    
    async def process_frame(self, frame_data: FrameData) -> Optional[FrameData]:
        """Process a single frame. Override in subclasses."""
        raise NotImplementedError
    
    async def start_processing(self):
        """Start the processor."""
        self.is_processing = True
    
    async def stop_processing(self):
        """Stop the processor."""
        self.is_processing = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_frames = self.stats['frames_processed'] + self.stats['frames_dropped']
        avg_processing_time = (self.stats['processing_time_total'] / 
                             max(1, self.stats['frames_processed']))
        
        return {
            **self.stats,
            'drop_rate': self.stats['frames_dropped'] / max(1, total_frames),
            'avg_processing_time': avg_processing_time
        }


class RealtimeProcessor(StreamProcessor):
    """Real-time stream processor with strict timing constraints."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.max_latency = config.max_latency_ms / 1000.0
        self.enable_frame_skipping = config.enable_frame_skipping
        self.last_processed_time = 0.0
    
    async def process_frame(self, frame_data: FrameData) -> Optional[FrameData]:
        """Process frame with real-time constraints."""
        start_time = time.time()
        
        # Check if we should skip this frame for timing
        if self.enable_frame_skipping:
            time_since_last = start_time - self.last_processed_time
            if time_since_last < (1.0 / 60.0):  # Skip if processing faster than 60 FPS
                self.stats['frames_dropped'] += 1
                return None
        
        try:
            # Actual processing would happen here
            # For now, just add processing delay simulation
            processing_delay = 0.01  # 10ms processing time
            await asyncio.sleep(processing_delay)
            
            # Check if we exceeded latency budget
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_latency:
                logger.warning(f"Frame processing exceeded latency budget: {elapsed_time:.3f}s")
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['processing_time_total'] += elapsed_time
            self.last_processed_time = time.time()
            
            # Return processed frame (same frame for now)
            return frame_data
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.stats['errors'] += 1
            return None 