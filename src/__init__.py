"""
Multi-Agent AI Hiring System

A LangGraph-based system that implements a two-agent architecture for fair hiring decisions.
Includes bias detection and self-auditing capabilities.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Main workflow function
from .main import create_hiring_workflow

# Configuration
from .config import Config

# Agents
from .agents.job_matching_agent import JobMatchingAgent
from .agents.bias_classification_agent import BiasClassificationAgent

__all__ = [
    "create_hiring_workflow",
    "Config",
    "JobMatchingAgent", 
    "BiasClassificationAgent"
]
