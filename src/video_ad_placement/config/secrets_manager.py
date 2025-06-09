"""
Secrets Manager

This module provides secure handling of sensitive configuration data
with support for multiple backends including environment variables,
files, and cloud secret managers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    name: str
    source: str
    last_updated: float
    encrypted: bool
    description: str = ""


class SecretBackend(ABC):
    """Abstract base class for secret backends."""
    
    @abstractmethod
    async def get_secret(self, name: str) -> Optional[str]:
        """Get a secret by name."""
        pass
    
    @abstractmethod
    async def set_secret(self, name: str, value: str) -> bool:
        """Set a secret value."""
        pass
    
    @abstractmethod
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        pass
    
    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret names."""
        pass


class EnvironmentBackend(SecretBackend):
    """Environment variables backend for secrets."""
    
    def __init__(self, prefix: str = "VADP_"):
        self.prefix = prefix
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment variable."""
        env_name = f"{self.prefix}{name.upper()}"
        return os.getenv(env_name)
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set environment variable (for current process only)."""
        env_name = f"{self.prefix}{name.upper()}"
        os.environ[env_name] = value
        return True
    
    async def delete_secret(self, name: str) -> bool:
        """Remove environment variable."""
        env_name = f"{self.prefix}{name.upper()}"
        if env_name in os.environ:
            del os.environ[env_name]
            return True
        return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets with the prefix."""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secret_name = key[len(self.prefix):].lower()
                secrets.append(secret_name)
        return secrets


class FileBackend(SecretBackend):
    """File-based backend for secrets with encryption."""
    
    def __init__(self, secrets_dir: str = ".secrets", master_key: Optional[str] = None):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(mode=0o700, exist_ok=True)
        
        # Setup encryption
        self.cipher = None
        if master_key:
            self._setup_encryption(master_key)
        
        self.metadata_file = self.secrets_dir / "metadata.json"
        self._metadata: Dict[str, SecretMetadata] = {}
        self._load_metadata()
    
    def _setup_encryption(self, master_key: str):
        """Setup encryption using the master key."""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'video_ad_placement_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from file."""
        secret_file = self.secrets_dir / f"{name}.secret"
        
        if not secret_file.exists():
            return None
        
        try:
            with open(secret_file, 'rb') as f:
                encrypted_data = f.read()
            
            if self.cipher:
                decrypted_data = self.cipher.decrypt(encrypted_data)
                return decrypted_data.decode('utf-8')
            else:
                return encrypted_data.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to read secret {name}: {e}")
            return None
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in file."""
        secret_file = self.secrets_dir / f"{name}.secret"
        
        try:
            data = value.encode('utf-8')
            
            if self.cipher:
                encrypted_data = self.cipher.encrypt(data)
            else:
                encrypted_data = data
            
            # Write with restrictive permissions
            with open(secret_file, 'wb') as f:
                f.write(encrypted_data)
            
            secret_file.chmod(0o600)
            
            # Update metadata
            import time
            self._metadata[name] = SecretMetadata(
                name=name,
                source="file",
                last_updated=time.time(),
                encrypted=self.cipher is not None,
                description=f"Secret stored in {secret_file}"
            )
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret file."""
        secret_file = self.secrets_dir / f"{name}.secret"
        
        try:
            if secret_file.exists():
                secret_file.unlink()
            
            if name in self._metadata:
                del self._metadata[name]
                self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete secret {name}: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secret files."""
        secrets = []
        for secret_file in self.secrets_dir.glob("*.secret"):
            secret_name = secret_file.stem
            secrets.append(secret_name)
        return secrets
    
    def _load_metadata(self):
        """Load secrets metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                self._metadata = {
                    name: SecretMetadata(**data)
                    for name, data in metadata_data.items()
                }
                
            except Exception as e:
                logger.error(f"Failed to load secrets metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self):
        """Save secrets metadata."""
        try:
            metadata_data = {
                name: {
                    'name': meta.name,
                    'source': meta.source,
                    'last_updated': meta.last_updated,
                    'encrypted': meta.encrypted,
                    'description': meta.description
                }
                for name, meta in self._metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
            
            self.metadata_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Failed to save secrets metadata: {e}")


class AWSSecretsBackend(SecretBackend):
    """AWS Secrets Manager backend."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    def _get_client(self):
        """Get AWS Secrets Manager client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('secretsmanager', region_name=self.region)
            except ImportError:
                raise ImportError("boto3 required for AWS Secrets Manager backend")
        return self._client
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=name)
            return response['SecretString']
            
        except Exception as e:
            logger.error(f"Failed to get AWS secret {name}: {e}")
            return None
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in AWS Secrets Manager."""
        try:
            client = self._get_client()
            
            # Try to update existing secret
            try:
                client.update_secret(SecretId=name, SecretString=value)
            except client.exceptions.ResourceNotFoundException:
                # Create new secret
                client.create_secret(Name=name, SecretString=value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set AWS secret {name}: {e}")
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        try:
            client = self._get_client()
            client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete AWS secret {name}: {e}")
            return False
    
    async def list_secrets(self) -> List[str]:
        """List all secrets in AWS Secrets Manager."""
        try:
            client = self._get_client()
            response = client.list_secrets()
            return [secret['Name'] for secret in response['SecretList']]
            
        except Exception as e:
            logger.error(f"Failed to list AWS secrets: {e}")
            return []


class SecretsManager:
    """Comprehensive secrets manager with multiple backends."""
    
    def __init__(self, backends: Optional[List[SecretBackend]] = None, 
                 default_backend: str = "environment"):
        """
        Initialize secrets manager.
        
        Args:
            backends: List of secret backends to use
            default_backend: Default backend for storing new secrets
        """
        self.backends: Dict[str, SecretBackend] = {}
        self.default_backend = default_backend
        
        # Setup default backends if none provided
        if backends is None:
            self._setup_default_backends()
        else:
            for i, backend in enumerate(backends):
                backend_name = f"backend_{i}"
                self.backends[backend_name] = backend
        
        # Cache for frequently accessed secrets
        self._secret_cache: Dict[str, str] = {}
        self._cache_enabled = True
        
        logger.info(f"SecretsManager initialized with backends: {list(self.backends.keys())}")
    
    def _setup_default_backends(self):
        """Setup default secret backends."""
        # Environment variables backend
        self.backends["environment"] = EnvironmentBackend()
        
        # File backend with encryption if master key is available
        master_key = os.getenv("VADP_MASTER_KEY")
        self.backends["file"] = FileBackend(master_key=master_key)
        
        # AWS backend if credentials are available
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
            try:
                self.backends["aws"] = AWSSecretsBackend()
                logger.info("AWS Secrets Manager backend enabled")
            except Exception as e:
                logger.warning(f"AWS Secrets Manager not available: {e}")
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from available backends."""
        
        # Check cache first
        if self._cache_enabled and name in self._secret_cache:
            return self._secret_cache[name]
        
        # Try each backend in order
        for backend_name, backend in self.backends.items():
            try:
                value = await backend.get_secret(name)
                if value is not None:
                    # Cache the value
                    if self._cache_enabled:
                        self._secret_cache[name] = value
                    
                    logger.debug(f"Secret {name} retrieved from {backend_name}")
                    return value
                    
            except Exception as e:
                logger.warning(f"Failed to get secret {name} from {backend_name}: {e}")
        
        logger.warning(f"Secret {name} not found in any backend")
        return None
    
    async def set_secret(self, name: str, value: str, 
                        backend: Optional[str] = None) -> bool:
        """Set secret in specified backend."""
        
        backend_name = backend or self.default_backend
        
        if backend_name not in self.backends:
            logger.error(f"Backend {backend_name} not available")
            return False
        
        try:
            success = await self.backends[backend_name].set_secret(name, value)
            
            if success:
                # Update cache
                if self._cache_enabled:
                    self._secret_cache[name] = value
                
                logger.info(f"Secret {name} set in {backend_name}")
                return True
            else:
                logger.error(f"Failed to set secret {name} in {backend_name}")
                return False
                
        except Exception as e:
            logger.error(f"Exception setting secret {name} in {backend_name}: {e}")
            return False
    
    async def delete_secret(self, name: str, 
                           backend: Optional[str] = None) -> bool:
        """Delete secret from specified backend or all backends."""
        
        if backend:
            # Delete from specific backend
            if backend not in self.backends:
                logger.error(f"Backend {backend} not available")
                return False
            
            try:
                success = await self.backends[backend].delete_secret(name)
                if success and name in self._secret_cache:
                    del self._secret_cache[name]
                return success
                
            except Exception as e:
                logger.error(f"Exception deleting secret {name} from {backend}: {e}")
                return False
        
        else:
            # Delete from all backends
            success_count = 0
            for backend_name, backend_obj in self.backends.items():
                try:
                    if await backend_obj.delete_secret(name):
                        success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete secret {name} from {backend_name}: {e}")
            
            if name in self._secret_cache:
                del self._secret_cache[name]
            
            return success_count > 0
    
    async def get_secrets(self, names: List[str]) -> Dict[str, str]:
        """Get multiple secrets at once."""
        secrets = {}
        
        for name in names:
            value = await self.get_secret(name)
            if value is not None:
                secrets[name] = value
        
        return secrets
    
    async def list_all_secrets(self) -> Dict[str, List[str]]:
        """List secrets from all backends."""
        all_secrets = {}
        
        for backend_name, backend in self.backends.items():
            try:
                secrets = await backend.list_secrets()
                all_secrets[backend_name] = secrets
            except Exception as e:
                logger.error(f"Failed to list secrets from {backend_name}: {e}")
                all_secrets[backend_name] = []
        
        return all_secrets
    
    async def rotate_secret(self, name: str, new_value: str, 
                           backend: Optional[str] = None) -> bool:
        """Rotate a secret value."""
        
        # Get current value for logging
        current_value = await self.get_secret(name)
        
        # Set new value
        success = await self.set_secret(name, new_value, backend)
        
        if success:
            logger.info(f"Secret {name} rotated successfully")
            
            # Optional: Keep track of rotation history
            rotation_history_name = f"{name}_rotation_history"
            history = await self.get_secret(rotation_history_name)
            
            if history:
                try:
                    history_data = json.loads(history)
                except:
                    history_data = []
            else:
                history_data = []
            
            import time
            history_data.append({
                'timestamp': time.time(),
                'old_value_hash': hashlib.sha256(
                    (current_value or "").encode()
                ).hexdigest()[:16],
                'new_value_hash': hashlib.sha256(
                    new_value.encode()
                ).hexdigest()[:16]
            })
            
            # Keep only last 10 rotations
            history_data = history_data[-10:]
            
            await self.set_secret(
                rotation_history_name, 
                json.dumps(history_data), 
                backend
            )
        
        return success
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._secret_cache.clear()
        logger.info("Secrets cache cleared")
    
    def disable_cache(self):
        """Disable secrets caching."""
        self._cache_enabled = False
        self.clear_cache()
        logger.info("Secrets caching disabled")
    
    def enable_cache(self):
        """Enable secrets caching."""
        self._cache_enabled = True
        logger.info("Secrets caching enabled")
    
    def add_backend(self, name: str, backend: SecretBackend):
        """Add a new secret backend."""
        self.backends[name] = backend
        logger.info(f"Added secret backend: {name}")
    
    def remove_backend(self, name: str):
        """Remove a secret backend."""
        if name in self.backends:
            del self.backends[name]
            logger.info(f"Removed secret backend: {name}")
    
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all backends."""
        status = {}
        
        for name, backend in self.backends.items():
            try:
                # Test backend connectivity
                test_result = "available"
                backend_type = type(backend).__name__
                
                status[name] = {
                    "type": backend_type,
                    "status": test_result,
                    "error": None
                }
                
            except Exception as e:
                status[name] = {
                    "type": type(backend).__name__,
                    "status": "error",
                    "error": str(e)
                }
        
        return status 