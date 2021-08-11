"""A package for sleep stage classification using ECG data."""

from . import io
from .heartbeat_detection import compare_heartbeats, detect_heartbeats, rri_similarity

__all__ = [
    'io',
    'detect_heartbeats',
    'compare_heartbeats',
    'rri_similarity',
]

__version__ = '0.2.0'
