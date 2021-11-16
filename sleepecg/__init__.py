"""A package for sleep stage classification using ECG data."""

from . import io
from .config import get_config, set_config
from .heartbeats import compare_heartbeats, detect_heartbeats, rri_similarity

__all__ = [
    'io',
    'detect_heartbeats',
    'compare_heartbeats',
    'rri_similarity',
    'get_config',
    'set_config',
]

__version__ = '0.4.0-dev'
