"""
Backends package for tool-call dispatcher.
Provides model backends for Needle and Qwen3.
"""

from .needle_backend import NeedleBackend
from .qwen3_backend import Qwen3Backend

__all__ = ["NeedleBackend", "Qwen3Backend"]
