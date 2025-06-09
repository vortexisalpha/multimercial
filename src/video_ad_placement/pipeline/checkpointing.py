"""
Checkpointing and Recovery System

This module provides comprehensive checkpointing and recovery capabilities for the video
processing pipeline, enabling resume from failures and long video processing.
"""

import time
import pickle
import json
import asyncio
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import gzip
import shutil

logger = logging.getLogger(__name__)


class CheckpointStrategy(Enum):
    """Checkpointing strategies."""
    TIME_BASED = "time_based"        # Checkpoint every N seconds
    FRAME_BASED = "frame_based"      # Checkpoint every N frames
    QUALITY_BASED = "quality_based"  # Checkpoint when quality drops
    ADAPTIVE = "adaptive"            # Dynamic checkpointing
    MANUAL = "manual"                # Manual checkpoints only


class CompressionType(Enum):
    """Checkpoint compression options."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    checkpoint_id: str
    timestamp: float
    frame_id: int
    total_frames: int
    progress: float
    pipeline_config: Dict[str, Any]
    video_info: Dict[str, Any]
    checksum: str
    file_size: int
    compression: CompressionType
    version: str = "1.0"


@dataclass
class ProcessingCheckpoint:
    """Complete checkpoint of processing state."""
    checkpoint_id: str
    timestamp: float
    frame_id: int
    pipeline_state: Dict[str, Any]
    frame_results: List[Any]  # Recent frame results
    resource_allocations: List[str]  # Resource allocation IDs
    component_states: Dict[str, Any] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(**data)


class CheckpointStorage:
    """Manages checkpoint storage and retrieval."""
    
    def __init__(self, storage_path: str, max_checkpoints: int = 10):
        self.storage_path = Path(storage_path)
        self.max_checkpoints = max_checkpoints
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage settings
        self.compression = CompressionType.GZIP
        self.verify_checksums = True
        
        logger.info(f"Checkpoint storage initialized at {self.storage_path}")
    
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint, 
                       metadata: CheckpointMetadata) -> str:
        """Save checkpoint to storage."""
        checkpoint_file = self.storage_path / f"checkpoint_{checkpoint.checkpoint_id}.pkl"
        metadata_file = self.storage_path / f"metadata_{checkpoint.checkpoint_id}.json"
        
        try:
            # Serialize checkpoint data
            checkpoint_data = checkpoint.to_dict()
            
            # Apply compression if enabled
            if self.compression == CompressionType.GZIP:
                with gzip.open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            else:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            # Calculate checksum
            checksum = self._calculate_checksum(checkpoint_file)
            metadata.checksum = checksum
            metadata.file_size = checkpoint_file.stat().st_size
            metadata.compression = self.compression
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint.checkpoint_id}")
            return str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            # Cleanup partial files
            for file_path in [checkpoint_file, metadata_file]:
                if file_path.exists():
                    file_path.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ProcessingCheckpoint]:
        """Load checkpoint from storage."""
        checkpoint_file = self.storage_path / f"checkpoint_{checkpoint_id}.pkl"
        metadata_file = self.storage_path / f"metadata_{checkpoint_id}.json"
        
        if not checkpoint_file.exists() or not metadata_file.exists():
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return None
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
                metadata = CheckpointMetadata(**metadata_data)
            
            # Verify checksum if enabled
            if self.verify_checksums:
                current_checksum = self._calculate_checksum(checkpoint_file)
                if current_checksum != metadata.checksum:
                    raise ValueError(f"Checksum mismatch for checkpoint {checkpoint_id}")
            
            # Load checkpoint data
            if metadata.compression == CompressionType.GZIP:
                with gzip.open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            else:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            checkpoint = ProcessingCheckpoint.from_dict(checkpoint_data)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []
        
        for metadata_file in self.storage_path.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                    metadata = CheckpointMetadata(**metadata_data)
                    checkpoints.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_file = self.storage_path / f"checkpoint_{checkpoint_id}.pkl"
        metadata_file = self.storage_path / f"metadata_{checkpoint_id}.json"
        
        success = True
        for file_path in [checkpoint_file, metadata_file]:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
                    success = False
        
        if success:
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
        
        return success
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            checkpoints_to_delete = checkpoints[self.max_checkpoints:]
            
            for metadata in checkpoints_to_delete:
                self.delete_checkpoint(metadata.checkpoint_id)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        checkpoints = self.list_checkpoints()
        
        total_size = sum(cp.file_size for cp in checkpoints)
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_checkpoint': min(cp.timestamp for cp in checkpoints) if checkpoints else None,
            'newest_checkpoint': max(cp.timestamp for cp in checkpoints) if checkpoints else None,
            'storage_path': str(self.storage_path)
        }


class CheckpointManager:
    """Manages checkpointing strategy and execution."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize storage
        storage_path = getattr(config, 'checkpoint_storage_path', './checkpoints')
        max_checkpoints = getattr(config, 'max_checkpoints', 10)
        self.storage = CheckpointStorage(storage_path, max_checkpoints)
        
        # Checkpointing strategy
        self.strategy = getattr(config, 'checkpoint_strategy', CheckpointStrategy.FRAME_BASED)
        self.checkpoint_interval = getattr(config, 'checkpoint_interval', 100)
        self.auto_cleanup = getattr(config, 'auto_cleanup_checkpoints', True)
        
        # State tracking
        self.last_checkpoint_time = 0.0
        self.last_checkpoint_frame = 0
        self.checkpoint_counter = 0
        self.pending_checkpoints = {}
        
        # Threading
        self.checkpoint_lock = threading.RLock()
        self.background_save_executor = None
        
        logger.info(f"CheckpointManager initialized with strategy: {self.strategy.value}")
    
    async def save_checkpoint(self, checkpoint: ProcessingCheckpoint) -> str:
        """Save a checkpoint asynchronously."""
        
        with self.checkpoint_lock:
            self.checkpoint_counter += 1
            checkpoint.checkpoint_id = f"cp_{int(time.time())}_{self.checkpoint_counter}"
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint.checkpoint_id,
            timestamp=checkpoint.timestamp,
            frame_id=checkpoint.frame_id,
            total_frames=0,  # Would be filled from video info
            progress=0.0,    # Would be calculated
            pipeline_config={},  # Would be filled from config
            video_info={},   # Would be filled from video info
            checksum="",     # Will be calculated during save
            file_size=0,     # Will be calculated during save
            compression=self.storage.compression
        )
        
        try:
            # Save checkpoint in background if possible
            if self.background_save_executor:
                loop = asyncio.get_event_loop()
                checkpoint_file = await loop.run_in_executor(
                    self.background_save_executor,
                    self.storage.save_checkpoint,
                    checkpoint,
                    metadata
                )
            else:
                checkpoint_file = self.storage.save_checkpoint(checkpoint, metadata)
            
            # Update tracking
            self.last_checkpoint_time = checkpoint.timestamp
            self.last_checkpoint_frame = checkpoint.frame_id
            
            return checkpoint_file
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[ProcessingCheckpoint]:
        """Load a checkpoint asynchronously."""
        try:
            if self.background_save_executor:
                loop = asyncio.get_event_loop()
                checkpoint = await loop.run_in_executor(
                    self.background_save_executor,
                    self.storage.load_checkpoint,
                    checkpoint_id
                )
            else:
                checkpoint = self.storage.load_checkpoint(checkpoint_id)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def should_checkpoint(self, frame_id: int, timestamp: float, 
                         quality_score: float = 1.0) -> bool:
        """Determine if a checkpoint should be created."""
        
        if self.strategy == CheckpointStrategy.MANUAL:
            return False
        
        elif self.strategy == CheckpointStrategy.FRAME_BASED:
            return (frame_id - self.last_checkpoint_frame) >= self.checkpoint_interval
        
        elif self.strategy == CheckpointStrategy.TIME_BASED:
            return (timestamp - self.last_checkpoint_time) >= self.checkpoint_interval
        
        elif self.strategy == CheckpointStrategy.QUALITY_BASED:
            # Checkpoint when quality drops significantly
            return quality_score < 0.5
        
        elif self.strategy == CheckpointStrategy.ADAPTIVE:
            # Dynamic strategy based on processing characteristics
            frame_interval = frame_id - self.last_checkpoint_frame
            time_interval = timestamp - self.last_checkpoint_time
            
            # Checkpoint more frequently if processing is unstable
            base_interval = self.checkpoint_interval
            if quality_score < 0.7:
                base_interval = max(10, base_interval // 2)
            
            return frame_interval >= base_interval or time_interval >= 300.0  # 5 minutes max
        
        return False
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get metadata for the most recent checkpoint."""
        checkpoints = self.storage.list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def get_checkpoint_for_frame(self, target_frame: int) -> Optional[CheckpointMetadata]:
        """Find the best checkpoint for resuming from a specific frame."""
        checkpoints = self.storage.list_checkpoints()
        
        # Find checkpoint with frame_id <= target_frame, closest to target
        valid_checkpoints = [cp for cp in checkpoints if cp.frame_id <= target_frame]
        
        if not valid_checkpoints:
            return None
        
        # Return checkpoint closest to target frame
        return max(valid_checkpoints, key=lambda cp: cp.frame_id)
    
    def cleanup_checkpoints(self, keep_count: int = 3):
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self.storage.list_checkpoints()
        
        if len(checkpoints) > keep_count:
            checkpoints_to_delete = checkpoints[keep_count:]
            
            for metadata in checkpoints_to_delete:
                self.storage.delete_checkpoint(metadata.checkpoint_id)
            
            logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            **storage_stats,
            'strategy': self.strategy.value,
            'checkpoint_interval': self.checkpoint_interval,
            'last_checkpoint_time': self.last_checkpoint_time,
            'last_checkpoint_frame': self.last_checkpoint_frame,
            'checkpoint_counter': self.checkpoint_counter
        }


class RecoveryManager:
    """Manages recovery from checkpoints and error states."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        
        logger.info("RecoveryManager initialized")
    
    async def recover_from_failure(self, error: Exception, 
                                  current_frame: int = None) -> Optional[ProcessingCheckpoint]:
        """Attempt to recover from a processing failure."""
        
        # Find appropriate checkpoint
        if current_frame is not None:
            checkpoint_metadata = self.checkpoint_manager.get_checkpoint_for_frame(current_frame)
        else:
            checkpoint_metadata = self.checkpoint_manager.get_latest_checkpoint()
        
        if not checkpoint_metadata:
            logger.error("No suitable checkpoint found for recovery")
            return None
        
        # Check recovery attempt limits
        checkpoint_id = checkpoint_metadata.checkpoint_id
        attempts = self.recovery_attempts.get(checkpoint_id, 0)
        
        if attempts >= self.max_recovery_attempts:
            logger.error(f"Maximum recovery attempts exceeded for checkpoint {checkpoint_id}")
            return None
        
        # Attempt recovery
        try:
            logger.info(f"Attempting recovery from checkpoint {checkpoint_id} (attempt {attempts + 1})")
            
            checkpoint = await self.checkpoint_manager.load_checkpoint(checkpoint_id)
            
            if checkpoint:
                self.recovery_attempts[checkpoint_id] = attempts + 1
                logger.info(f"Successfully recovered from checkpoint {checkpoint_id}")
                return checkpoint
            else:
                logger.error(f"Failed to load checkpoint {checkpoint_id}")
                return None
                
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            self.recovery_attempts[checkpoint_id] = attempts + 1
            return None
    
    def validate_checkpoint(self, checkpoint: ProcessingCheckpoint) -> bool:
        """Validate that a checkpoint is suitable for recovery."""
        
        # Basic validation
        if not checkpoint.checkpoint_id or not checkpoint.pipeline_state:
            return False
        
        # Check timestamp is not too old (e.g., more than 24 hours)
        max_age = 24 * 3600  # 24 hours
        if time.time() - checkpoint.timestamp > max_age:
            logger.warning(f"Checkpoint {checkpoint.checkpoint_id} is too old for recovery")
            return False
        
        # Validate required fields
        required_fields = ['frame_id', 'pipeline_state']
        for field in required_fields:
            if not hasattr(checkpoint, field) or getattr(checkpoint, field) is None:
                logger.error(f"Checkpoint {checkpoint.checkpoint_id} missing required field: {field}")
                return False
        
        return True
    
    def create_recovery_plan(self, target_frame: int, 
                           available_checkpoints: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Create a recovery plan for reaching a target frame."""
        
        # Find best checkpoint
        suitable_checkpoints = [cp for cp in available_checkpoints if cp.frame_id <= target_frame]
        
        if not suitable_checkpoints:
            return {
                'can_recover': False,
                'reason': 'No suitable checkpoints available'
            }
        
        best_checkpoint = max(suitable_checkpoints, key=lambda cp: cp.frame_id)
        frames_to_reprocess = target_frame - best_checkpoint.frame_id
        
        return {
            'can_recover': True,
            'checkpoint_id': best_checkpoint.checkpoint_id,
            'checkpoint_frame': best_checkpoint.frame_id,
            'target_frame': target_frame,
            'frames_to_reprocess': frames_to_reprocess,
            'estimated_recovery_time': frames_to_reprocess * 0.1,  # Rough estimate
            'checkpoint_age_seconds': time.time() - best_checkpoint.timestamp
        }
    
    def reset_recovery_attempts(self, checkpoint_id: str = None):
        """Reset recovery attempt counters."""
        if checkpoint_id:
            self.recovery_attempts.pop(checkpoint_id, None)
            logger.info(f"Reset recovery attempts for checkpoint {checkpoint_id}")
        else:
            self.recovery_attempts.clear()
            logger.info("Reset all recovery attempts")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total_attempts = sum(self.recovery_attempts.values())
        failed_checkpoints = len([cp for cp, attempts in self.recovery_attempts.items() 
                                if attempts >= self.max_recovery_attempts])
        
        return {
            'total_recovery_attempts': total_attempts,
            'failed_checkpoints': failed_checkpoints,
            'active_recovery_attempts': len(self.recovery_attempts),
            'max_attempts_per_checkpoint': self.max_recovery_attempts
        } 