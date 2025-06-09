"""
Video Advertisement Placement API

This module provides a comprehensive RESTful API for the video advertisement
placement service using FastAPI with authentication, rate limiting, and
comprehensive documentation.
"""

from .main import create_app
from .models import *
from .auth import get_current_user, verify_api_key
from .rate_limiting import rate_limiter
from .websocket import websocket_manager

__all__ = [
    'create_app',
    'get_current_user',
    'verify_api_key',
    'rate_limiter',
    'websocket_manager'
] 