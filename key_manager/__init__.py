"""
API Key Manager Package

This package provides comprehensive API key management for handling multiple
Google Gemini API keys with automatic rotation and rate limiting.

Components:
- api_key_manager.py: Core API key management functionality
- setup_api_keys.py: Interactive setup script for configuring API keys
- monitor_api_keys.py: Real-time monitoring tool for API key usage
- batch_processor.py: Batch processing with API key statistics
"""

from .api_key_manager import (
    get_api_key_manager,
    get_current_api_key,
    get_key_usage_stats,
    initialize_api_key_manager,
    APIKeyManager
)

__all__ = [
    'get_api_key_manager',
    'get_current_api_key', 
    'get_key_usage_stats',
    'initialize_api_key_manager',
    'APIKeyManager'
]
