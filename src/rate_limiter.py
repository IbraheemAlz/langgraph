"""
Simplified rate limiting utility for API calls.
Now integrated with the API Key Manager for automatic key rotation.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps
import threading
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple thread-safe rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int = 5):
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()
        
        # Note: Rate limiting is now handled by the API Key Manager
        # This class is kept for backwards compatibility
        logger.info("⚠️  Rate limiting is now handled by API Key Manager")
        
    def _wait_if_needed(self) -> None:
        """Wait if we've exceeded the rate limit."""
        # API Key Manager handles rate limiting automatically
        # This method is kept for backwards compatibility but does nothing
        pass
    
    def call_with_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function with rate limiting."""
        # Rate limiting is now handled automatically by the API Key Manager
        # through dynamic key rotation, so we just call the function directly
        return func(*args, **kwargs)

# Global rate limiter instance (kept for backwards compatibility)
_global_rate_limiter = RateLimiter()

def rate_limited(func: Callable) -> Callable:
    """
    Decorator for rate-limited function calls.
    
    Note: Rate limiting is now handled by the API Key Manager.
    This decorator is kept for backwards compatibility.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # API Key Manager handles rate limiting automatically
        return func(*args, **kwargs)
    return wrapper

def set_rate_limit(max_requests_per_minute: int) -> None:
    """Update the global rate limit settings."""
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(max_requests_per_minute)
    logger.info(f"⚠️  Rate limit setting ignored - using API Key Manager instead")
