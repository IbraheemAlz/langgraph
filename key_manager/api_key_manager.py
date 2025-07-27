"""
API Key Manager for handling multiple Google Gemini API keys with automatic rotation.

This system manages multiple API keys to maximize request throughput by rotating
between keys when rate limits are reached.
"""

import os
import time
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

# Load environment variables from .env file in parent directory
try:
    from dotenv import load_dotenv
    # Look for .env file in parent directory (project root)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

logger = logging.getLogger(__name__)

@dataclass
class KeyUsage:
    """Track usage statistics for an API key."""
    key_id: str
    requests_made: int = 0
    last_used: Optional[datetime] = None
    rate_limit_reset: Optional[datetime] = None
    is_blocked: bool = False
    total_requests: int = 0

class APIKeyManager:
    """
    Manages multiple Google Gemini API keys with automatic rotation and rate limiting.
    
    Features:
    - Automatic key rotation when rate limits are hit
    - Per-key usage tracking
    - Thread-safe operations
    - Configurable rate limits
    - Automatic recovery from rate-limited keys
    """
    
    def __init__(self, 
                 rate_limit_per_minute: int = 5,
                 rate_limit_window_seconds: int = 60):
        """
        Initialize the API Key Manager.
        
        Args:
            rate_limit_per_minute: Maximum requests per minute per key
            rate_limit_window_seconds: Time window for rate limiting
        """
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self._lock = threading.Lock()
        
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
        self.key_usage: Dict[str, KeyUsage] = {}
        self.current_key_index = 0
        
        # Initialize usage tracking for all keys
        for i, key in enumerate(self.api_keys):
            key_id = f"key_{i+1}"
            self.key_usage[key_id] = KeyUsage(key_id=key_id)
        
        logger.info(f"🔑 API Key Manager initialized with {len(self.api_keys)} keys")
        logger.info(f"📊 Rate limit: {rate_limit_per_minute} requests per {rate_limit_window_seconds} seconds per key")
    
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables."""
        keys = []
        
        # Try to load keys from GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
        for i in range(1, 10):  # Support up to 9 keys
            key_name = f"GOOGLE_API_KEY_{i}"
            key_value = os.getenv(key_name)
            if key_value:
                keys.append(key_value)
                logger.info(f"✅ Loaded {key_name}")
            else:
                # If no numbered key found, try the main one for backwards compatibility
                if i == 1:
                    main_key = os.getenv("GOOGLE_API_KEY")
                    if main_key:
                        keys.append(main_key)
                        logger.info("✅ Loaded GOOGLE_API_KEY (main)")
                break
        
        if not keys:
            raise ValueError("No API keys found. Please set GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.")
        
        return keys
    
    def get_current_key(self) -> str:
        """
        Get the current active API key, rotating if necessary.
        
        Returns:
            str: The current API key to use
        """
        with self._lock:
            # Check if current key is available
            if self._is_key_available(self.current_key_index):
                key = self.api_keys[self.current_key_index]
                self._record_key_usage(self.current_key_index)
                return key
            
            # Find next available key
            original_index = self.current_key_index
            for _ in range(len(self.api_keys)):
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                if self._is_key_available(self.current_key_index):
                    key = self.api_keys[self.current_key_index]
                    key_id = f"key_{self.current_key_index + 1}"
                    logger.info(f"🔄 Switched to {key_id} (index {self.current_key_index})")
                    self._record_key_usage(self.current_key_index)
                    return key
            
            # If no keys are available, wait for the next available key
            self.current_key_index = original_index
            wait_time = self._get_wait_time_for_key(self.current_key_index)
            logger.warning(f"⏳ All keys rate-limited. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
            # Try again after waiting
            return self.get_current_key()
    
    def _is_key_available(self, key_index: int) -> bool:
        """Check if a key is available for use (not rate-limited)."""
        key_id = f"key_{key_index + 1}"
        usage = self.key_usage.get(key_id)
        
        if not usage:
            return True
        
        now = datetime.now()
        
        # If key was blocked, check if enough time has passed
        if usage.is_blocked and usage.rate_limit_reset:
            if now >= usage.rate_limit_reset:
                usage.is_blocked = False
                usage.requests_made = 0
                logger.info(f"🔓 {key_id} unblocked and available")
                return True
            return False
        
        # Check if we're within rate limit
        if usage.requests_made >= self.rate_limit_per_minute:
            # Calculate when this key will be available again
            if usage.last_used:
                reset_time = usage.last_used + timedelta(seconds=self.rate_limit_window_seconds)
                if now >= reset_time:
                    # Reset the counter
                    usage.requests_made = 0
                    usage.is_blocked = False
                    return True
                else:
                    # Mark as blocked
                    usage.is_blocked = True
                    usage.rate_limit_reset = reset_time
                    return False
        
        return True
    
    def _record_key_usage(self, key_index: int):
        """Record usage of a key."""
        key_id = f"key_{key_index + 1}"
        usage = self.key_usage[key_id]
        
        now = datetime.now()
        usage.requests_made += 1
        usage.total_requests += 1
        usage.last_used = now
        
        logger.debug(f"📊 {key_id}: {usage.requests_made}/{self.rate_limit_per_minute} requests used")
        
        # Check if we've hit the rate limit
        if usage.requests_made >= self.rate_limit_per_minute:
            usage.is_blocked = True
            usage.rate_limit_reset = now + timedelta(seconds=self.rate_limit_window_seconds)
            logger.info(f"🚫 {key_id} rate limit reached. Next available at {usage.rate_limit_reset.strftime('%H:%M:%S')}")
    
    def _get_wait_time_for_key(self, key_index: int) -> float:
        """Get the wait time until a key becomes available again."""
        key_id = f"key_{key_index + 1}"
        usage = self.key_usage.get(key_id)
        
        if not usage or not usage.rate_limit_reset:
            return 0.0
        
        now = datetime.now()
        if usage.rate_limit_reset > now:
            return (usage.rate_limit_reset - now).total_seconds()
        
        return 0.0
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all keys."""
        stats = {}
        
        with self._lock:
            for key_id, usage in self.key_usage.items():
                stats[key_id] = {
                    "requests_made_current_window": usage.requests_made,
                    "total_requests": usage.total_requests,
                    "is_blocked": usage.is_blocked,
                    "last_used": usage.last_used.isoformat() if usage.last_used else None,
                    "rate_limit_reset": usage.rate_limit_reset.isoformat() if usage.rate_limit_reset else None,
                }
        
        return stats
    
# Global instance
_api_key_manager: Optional[APIKeyManager] = None

def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager

def initialize_api_key_manager(rate_limit_per_minute: int = 5, 
                              rate_limit_window_seconds: int = 60) -> APIKeyManager:
    """Initialize the global API key manager with custom rate limits."""
    global _api_key_manager
    _api_key_manager = APIKeyManager(
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_window_seconds=rate_limit_window_seconds
    )
    return _api_key_manager

def get_current_api_key() -> str:
    """Get the current API key to use."""
    return get_api_key_manager().get_current_key()

def get_key_usage_stats() -> Dict[str, Dict[str, Any]]:
    """Get usage statistics for all API keys."""
    return get_api_key_manager().get_usage_stats()
