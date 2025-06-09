"""
Authentication and Authorization

This module provides JWT-based authentication, API key validation,
and user management for the FastAPI application.
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from passlib.context import CryptContext

from .models import User, Token, TokenData
from ..config.hydra_config import SecurityConfig

logger = logging.getLogger(__name__)


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Security schemes
        self.bearer_scheme = HTTPBearer()
        self.api_key_header = APIKeyHeader(name=self.config.api_key_header, auto_error=False)
        
        # In-memory user store (in production, use a database)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Create default admin user
        self._create_default_users()
        
        logger.info("AuthManager initialized")
    
    def _create_default_users(self):
        """Create default users for testing."""
        # Default admin user
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            is_admin=True,
            is_active=True
        )
        self.users["admin"] = admin_user
        
        # Default API user
        api_user = User(
            user_id="api_user",
            username="api_user",
            email="api@example.com",
            is_admin=False,
            is_active=True
        )
        self.users["api_user"] = api_user
        
        # Generate default API keys
        admin_api_key = "admin_" + hashlib.sha256("admin_secret".encode()).hexdigest()[:24]
        user_api_key = "user_" + hashlib.sha256("user_secret".encode()).hexdigest()[:24]
        
        self.api_keys[admin_api_key] = "admin"
        self.api_keys[user_api_key] = "api_user"
        
        logger.info(f"Created default users with API keys: admin={admin_api_key[:8]}..., user={user_api_key[:8]}...")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        # For testing, accept any password (in production, verify against stored hash)
        if user.is_active:
            return user
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(seconds=self.config.jwt_expiration)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.config.jwt_secret_key or "default_secret", 
            algorithm=self.config.jwt_algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(seconds=self.config.jwt_refresh_expiration)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.jwt_secret_key or "default_secret",
            algorithm=self.config.jwt_algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key or "default_secret",
                algorithms=[self.config.jwt_algorithm]
            )
            
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            
            if username is None and user_id is None:
                return None
            
            token_data = TokenData(username=username, user_id=user_id)
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT error: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify an API key and return the associated user."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            user = self.get_user_by_id(user_id)
            if user and user.is_active:
                return user
        return None
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user."""
        timestamp = str(int(time.time()))
        key_data = f"{user_id}_{timestamp}_{self.config.api_key_length}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()[:self.config.api_key_length]
        
        self.api_keys[api_key] = user_id
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False
    
    def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> User:
        """Create a new user."""
        user_id = f"user_{int(time.time())}_{username}"
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            is_admin=is_admin,
            is_active=True
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {username} (ID: {user_id})")
        
        return user
    
    def update_user(self, user_id: str, **updates) -> Optional[User]:
        """Update user information."""
        user = self.users.get(user_id)
        if not user:
            return None
        
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return user
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.is_active)
        admin_users = sum(1 for user in self.users.values() if user.is_admin)
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users,
            "api_keys_issued": len(self.api_keys)
        }


# Global auth manager instance
auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global auth_manager
    if auth_manager is None:
        # Create with default config for testing
        from ..config.hydra_config import SecurityConfig
        default_config = SecurityConfig()
        auth_manager = AuthManager(default_config)
    return auth_manager


async def get_current_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Optional[User]:
    """Get current user from JWT token."""
    if not credentials:
        return None
        
    auth_mgr = get_auth_manager()
    
    try:
        token_data = auth_mgr.verify_token(credentials.credentials)
        if token_data is None:
            return None
        
        # Get user by username or user_id
        user = None
        if token_data.username:
            user = auth_mgr.get_user_by_username(token_data.username)
        elif token_data.user_id:
            user = auth_mgr.get_user_by_id(token_data.user_id)
        
        if user is None:
            return None
        
        if not user.is_active:
            return None
        
        return user
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None


async def get_current_user_from_api_key(api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))) -> Optional[User]:
    """Get current user from API key."""
    if not api_key:
        return None
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.verify_api_key(api_key)
    
    if user and not user.is_active:
        return None
    
    return user


async def get_current_user(
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
    token_user: Optional[User] = Depends(get_current_user_from_token)
) -> User:
    """Get current user from either JWT token or API key."""
    
    # Try API key authentication first (for easier testing)
    if api_key_user:
        return api_key_user
    
    # Fall back to token authentication
    if token_user:
        return token_user
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key),
    token_user: Optional[User] = Depends(get_current_user_from_token)
) -> Optional[User]:
    """Get current user optionally (doesn't raise exception if not authenticated)."""
    
    # Try API key authentication first
    if api_key_user:
        return api_key_user
    
    # Fall back to token authentication
    if token_user:
        return token_user
    
    # Return None if no authentication
    return None


async def verify_api_key(api_key: str = Depends(APIKeyHeader(name="X-API-Key"))) -> User:
    """Verify API key and return user (for endpoints that require API key specifically)."""
    auth_mgr = get_auth_manager()
    user = auth_mgr.verify_api_key(api_key)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin privileges."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


def create_login_response(user: User) -> Token:
    """Create a login response with tokens."""
    auth_mgr = get_auth_manager()
    
    access_token_expires = timedelta(seconds=auth_mgr.config.jwt_expiration)
    access_token = auth_mgr.create_access_token(
        data={"sub": user.username, "user_id": user.user_id}, 
        expires_delta=access_token_expires
    )
    
    refresh_token = auth_mgr.create_refresh_token(
        data={"sub": user.username, "user_id": user.user_id}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_mgr.config.jwt_expiration,
        refresh_token=refresh_token
    ) 