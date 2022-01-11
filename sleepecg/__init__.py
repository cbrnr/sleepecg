"""A package for sleep stage classification using ECG data."""

from .config import get_config, set_config
from .feature_extraction import extract_features, preprocess_rri
from .heartbeats import compare_heartbeats, detect_heartbeats, rri_similarity
from .io import *  # noqa: F403

__version__ = '0.4.0'
