"""
Simple rate limiting utility for API calls.
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
        
    def _wait_if_needed(self) -> None:
        """Wait if we've exceeded the rate limit."""
        with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            while self.request_times and now - self.request_times[0] > timedelta(minutes=1):
                self.request_times.popleft()
            
            # If we've made too many requests, wait
            if len(self.request_times) >= self.max_requests_per_minute:
                wait_time = 60.1  # Wait just over a minute
                logger.info(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                # Clear old requests after waiting
                self.request_times.clear()
            
            # Add current request
            self.request_times.append(now)
    
    def call_with_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function with rate limiting."""
        self._wait_if_needed()
        return func(*args, **kwargs)

# Global rate limiter instance (kept for backwards compatibility)
_global_rate_limiter = RateLimiter()

def rate_limited(func: Callable) -> Callable:
    """Decorator for rate-limited function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return _global_rate_limiter.call_with_rate_limit(func, *args, **kwargs)
    return wrapper

def set_rate_limit(max_requests_per_minute: int) -> None:
    """Set the global rate limit."""
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(max_requests_per_minute)
    logger.info(f"⚡ Rate limiter set to {max_requests_per_minute} requests per minute")
