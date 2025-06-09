"""
Configuration Manager

This module provides comprehensive configuration management with versioning,
validation, secrets integration, and runtime updates.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import yaml
from omegaconf import DictConfig, OmegaConf
import asyncio
import threading

from .hydra_config import AppConfig, Environment
from .validators import ConfigValidator
from .secrets_manager import SecretsManager

logger = logging.getLogger(__name__)


class ConfigChangeType(Enum):
    """Types of configuration changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ROLLBACK = "rollback"


@dataclass
class ConfigVersion:
    """Configuration version information."""
    version_id: str
    timestamp: float
    change_type: ConfigChangeType
    changed_keys: List[str]
    user: str
    description: str
    config_hash: str
    backup_path: Optional[str] = None
    
    @property
    def version_date(self) -> str:
        """Get human-readable version date."""
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))


@dataclass
class ConfigChange:
    """Individual configuration change."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    user: str
    reason: str


class ConfigManager:
    """Comprehensive configuration manager with versioning and validation."""
    
    def __init__(self, config_dir: str = "conf", backup_dir: str = "conf/backups"):
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Current configuration
        self._config: Optional[AppConfig] = None
        self._config_dict: Optional[DictConfig] = None
        
        # Configuration management
        self.versions: List[ConfigVersion] = []
        self.change_history: List[ConfigChange] = []
        self.version_file = self.config_dir / "versions.json"
        self.config_file = self.config_dir / "config.yaml"
        
        # Validators and secrets
        self.validator = ConfigValidator()
        self.secrets_manager = SecretsManager()
        
        # Runtime update support
        self.update_callbacks: List[Callable[[AppConfig, AppConfig], None]] = []
        self.watch_enabled = False
        self._watch_thread: Optional[threading.Thread] = None
        self._last_modified_time = 0.0
        
        # Load existing versions
        self._load_version_history()
        
        logger.info(f"ConfigManager initialized with config dir: {self.config_dir}")
    
    async def load_config(self, config_path: Optional[str] = None, 
                         environment: Optional[Environment] = None) -> AppConfig:
        """Load configuration from file with environment overrides."""
        
        config_path = config_path or self.config_file
        
        try:
            # Load base configuration
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                config_dict = OmegaConf.create(config_data)
            else:
                # Create default configuration
                config_dict = OmegaConf.structured(AppConfig())
                await self.save_config(config_dict, "Initial configuration")
            
            # Apply environment-specific overrides
            if environment:
                env_overrides = self._get_environment_overrides(environment)
                config_dict = OmegaConf.merge(config_dict, env_overrides)
            
            # Apply secrets
            config_dict = await self._apply_secrets(config_dict)
            
            # Convert to structured config
            self._config_dict = config_dict
            self._config = OmegaConf.to_object(config_dict)
            
            # Validate configuration
            validation_result = self.validator.validate_config(self._config)
            if not validation_result.is_valid:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")
            
            logger.info(f"Configuration loaded successfully for environment: {environment}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def save_config(self, config: Union[AppConfig, DictConfig], 
                         description: str = "", user: str = "system") -> str:
        """Save configuration with versioning."""
        
        try:
            # Convert to dict config if needed
            if isinstance(config, AppConfig):
                config_dict = OmegaConf.structured(config)
            else:
                config_dict = config
            
            # Validate before saving
            config_obj = OmegaConf.to_object(config_dict)
            validation_result = self.validator.validate_config(config_obj)
            if not validation_result.is_valid:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")
            
            # Create backup of current config
            backup_path = None
            if self.config_file.exists():
                backup_path = await self._create_backup()
            
            # Calculate config hash
            config_str = OmegaConf.to_yaml(config_dict)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Determine changed keys
            changed_keys = []
            if self._config_dict:
                changed_keys = self._get_changed_keys(self._config_dict, config_dict)
            
            # Create version record
            version = ConfigVersion(
                version_id=f"v{int(time.time())}_{config_hash[:8]}",
                timestamp=time.time(),
                change_type=ConfigChangeType.UPDATE if self._config else ConfigChangeType.CREATE,
                changed_keys=changed_keys,
                user=user,
                description=description or f"Configuration {'update' if self._config else 'creation'}",
                config_hash=config_hash,
                backup_path=str(backup_path) if backup_path else None
            )
            
            # Save configuration
            with open(self.config_file, 'w') as f:
                yaml.dump(OmegaConf.to_container(config_dict), f, default_flow_style=False)
            
            # Update internal state
            self._config_dict = config_dict
            self._config = config_obj
            self.versions.append(version)
            self._save_version_history()
            
            # Trigger update callbacks
            if changed_keys:
                await self._trigger_update_callbacks()
            
            logger.info(f"Configuration saved with version: {version.version_id}")
            return version.version_id
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    async def update_config(self, updates: Dict[str, Any], 
                           description: str = "", user: str = "system") -> str:
        """Update specific configuration values."""
        
        if not self._config_dict:
            raise ValueError("No configuration loaded")
        
        # Create copy of current config
        updated_config = OmegaConf.copy(self._config_dict)
        
        # Record changes
        changes = []
        for key, new_value in updates.items():
            old_value = OmegaConf.select(updated_config, key)
            if old_value != new_value:
                changes.append(ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=time.time(),
                    user=user,
                    reason=description
                ))
                OmegaConf.set(updated_config, key, new_value)
        
        if not changes:
            logger.info("No configuration changes detected")
            return self.versions[-1].version_id if self.versions else ""
        
        # Save updated configuration
        self.change_history.extend(changes)
        version_id = await self.save_config(
            updated_config, 
            description or f"Updated {len(changes)} configuration values",
            user
        )
        
        return version_id
    
    async def rollback_config(self, version_id: str, user: str = "system") -> bool:
        """Rollback to a previous configuration version."""
        
        try:
            # Find target version
            target_version = None
            for version in self.versions:
                if version.version_id == version_id:
                    target_version = version
                    break
            
            if not target_version:
                raise ValueError(f"Version {version_id} not found")
            
            # Load backup configuration
            if not target_version.backup_path or not Path(target_version.backup_path).exists():
                raise ValueError(f"Backup file not found for version {version_id}")
            
            with open(target_version.backup_path, 'r') as f:
                rollback_config = yaml.safe_load(f)
            
            config_dict = OmegaConf.create(rollback_config)
            
            # Create rollback version
            await self.save_config(
                config_dict,
                f"Rollback to version {version_id}",
                user
            )
            
            # Update version type
            self.versions[-1].change_type = ConfigChangeType.ROLLBACK
            self._save_version_history()
            
            logger.info(f"Successfully rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_config(self) -> Optional[AppConfig]:
        """Get current configuration."""
        return self._config
    
    def get_config_dict(self) -> Optional[DictConfig]:
        """Get current configuration as DictConfig."""
        return self._config_dict
    
    def get_versions(self) -> List[ConfigVersion]:
        """Get all configuration versions."""
        return self.versions.copy()
    
    def get_change_history(self) -> List[ConfigChange]:
        """Get configuration change history."""
        return self.change_history.copy()
    
    def register_update_callback(self, callback: Callable[[AppConfig, AppConfig], None]):
        """Register callback for configuration updates."""
        self.update_callbacks.append(callback)
    
    async def start_config_watch(self):
        """Start watching configuration file for changes."""
        if self.watch_enabled:
            return
        
        self.watch_enabled = True
        self._watch_thread = threading.Thread(target=self._watch_config_file, daemon=True)
        self._watch_thread.start()
        logger.info("Configuration file watching started")
    
    def stop_config_watch(self):
        """Stop watching configuration file."""
        self.watch_enabled = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5.0)
        logger.info("Configuration file watching stopped")
    
    def _watch_config_file(self):
        """Watch configuration file for changes."""
        while self.watch_enabled:
            try:
                if self.config_file.exists():
                    current_modified = self.config_file.stat().st_mtime
                    
                    if current_modified > self._last_modified_time:
                        self._last_modified_time = current_modified
                        
                        # Reload configuration
                        asyncio.create_task(self._reload_config_from_file())
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Config file watch error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    async def _reload_config_from_file(self):
        """Reload configuration from file."""
        try:
            old_config = self._config
            await self.load_config()
            
            if old_config and self._config:
                await self._trigger_update_callbacks(old_config, self._config)
            
            logger.info("Configuration reloaded from file")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    async def _trigger_update_callbacks(self, old_config: Optional[AppConfig] = None, 
                                       new_config: Optional[AppConfig] = None):
        """Trigger configuration update callbacks."""
        old_config = old_config or AppConfig()  # Default to empty config
        new_config = new_config or self._config
        
        if not new_config:
            return
        
        for callback in self.update_callbacks:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, callback, old_config, new_config
                )
            except Exception as e:
                logger.error(f"Configuration update callback failed: {e}")
    
    async def _create_backup(self) -> Path:
        """Create backup of current configuration."""
        timestamp = int(time.time())
        backup_file = self.backup_dir / f"config_backup_{timestamp}.yaml"
        
        import shutil
        shutil.copy2(self.config_file, backup_file)
        
        # Clean old backups (keep last 10)
        backups = sorted(self.backup_dir.glob("config_backup_*.yaml"))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()
        
        return backup_file
    
    def _get_environment_overrides(self, environment: Environment) -> DictConfig:
        """Get environment-specific configuration overrides."""
        env_file = self.config_dir / f"{environment.value}.yaml"
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_data = yaml.safe_load(f)
            return OmegaConf.create(env_data)
        
        # Default environment overrides
        env_overrides = {}
        
        if environment == Environment.PRODUCTION:
            env_overrides = {
                "debug": False,
                "api": {"reload": False, "workers": 4},
                "security": {"ssl_enabled": True},
                "monitoring": {"log_level": "INFO"}
            }
        elif environment == Environment.DEVELOPMENT:
            env_overrides = {
                "debug": True,
                "api": {"reload": True, "workers": 1},
                "monitoring": {"log_level": "DEBUG"}
            }
        elif environment == Environment.STAGING:
            env_overrides = {
                "debug": False,
                "api": {"reload": False, "workers": 2},
                "monitoring": {"log_level": "INFO"}
            }
        
        return OmegaConf.create(env_overrides)
    
    async def _apply_secrets(self, config: DictConfig) -> DictConfig:
        """Apply secrets to configuration."""
        try:
            # Load secrets for sensitive configuration values
            secrets = await self.secrets_manager.get_secrets([
                "jwt_secret_key",
                "database_password",
                "redis_password",
                "cloud_access_key",
                "cloud_secret_key",
                "api_key"
            ])
            
            # Apply secrets to configuration
            if "jwt_secret_key" in secrets:
                OmegaConf.set(config, "security.jwt_secret_key", secrets["jwt_secret_key"])
            
            if "database_password" in secrets:
                OmegaConf.set(config, "database.password", secrets["database_password"])
            
            if "redis_password" in secrets:
                OmegaConf.set(config, "database.redis_password", secrets["redis_password"])
            
            if "cloud_access_key" in secrets:
                OmegaConf.set(config, "cloud.access_key", secrets["cloud_access_key"])
            
            if "cloud_secret_key" in secrets:
                OmegaConf.set(config, "cloud.secret_key", secrets["cloud_secret_key"])
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to apply secrets: {e}")
            return config
    
    def _get_changed_keys(self, old_config: DictConfig, new_config: DictConfig) -> List[str]:
        """Get list of changed configuration keys."""
        changed_keys = []
        
        def compare_nested(old_dict, new_dict, prefix=""):
            for key in set(list(old_dict.keys()) + list(new_dict.keys())):
                full_key = f"{prefix}.{key}" if prefix else key
                
                if key not in old_dict:
                    changed_keys.append(full_key)
                elif key not in new_dict:
                    changed_keys.append(full_key)
                elif OmegaConf.is_dict(old_dict[key]) and OmegaConf.is_dict(new_dict[key]):
                    compare_nested(old_dict[key], new_dict[key], full_key)
                elif old_dict[key] != new_dict[key]:
                    changed_keys.append(full_key)
        
        compare_nested(old_config, new_config)
        return changed_keys
    
    def _load_version_history(self):
        """Load version history from file."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    version_data = json.load(f)
                
                self.versions = [
                    ConfigVersion(**version) for version in version_data.get("versions", [])
                ]
                self.change_history = [
                    ConfigChange(**change) for change in version_data.get("changes", [])
                ]
                
            except Exception as e:
                logger.error(f"Failed to load version history: {e}")
                self.versions = []
                self.change_history = []
    
    def _save_version_history(self):
        """Save version history to file."""
        try:
            version_data = {
                "versions": [asdict(version) for version in self.versions],
                "changes": [asdict(change) for change in self.change_history]
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(version_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Get configuration management statistics."""
        return {
            "total_versions": len(self.versions),
            "total_changes": len(self.change_history),
            "current_version": self.versions[-1].version_id if self.versions else None,
            "last_updated": self.versions[-1].timestamp if self.versions else None,
            "watch_enabled": self.watch_enabled,
            "backup_count": len(list(self.backup_dir.glob("config_backup_*.yaml"))),
            "config_file_size": self.config_file.stat().st_size if self.config_file.exists() else 0
        } 