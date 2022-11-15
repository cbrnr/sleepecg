"""A package for sleep stage classification using ECG data."""

from .classification import (
    SleepClassifier,
    evaluate,
    list_classifiers,
    load_classifier,
    prepare_data_keras,
    print_class_balance,
    save_classifier,
    stage,
)
from .config import get_config, set_config
from .feature_extraction import extract_features, preprocess_rri
from .heartbeats import compare_heartbeats, detect_heartbeats, rri_similarity
from .io import *  # noqa: F403
from .plot import plot_ecg, plot_hypnogram

__version__ = "0.6.0-dev"
