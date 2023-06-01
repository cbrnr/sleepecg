"""A package for sleep stage classification using ECG data."""

from sleepecg.classification import (
    SleepClassifier,
    evaluate,
    list_classifiers,
    load_classifier,
    prepare_data_keras,
    print_class_balance,
    save_classifier,
    stage,
)
from sleepecg.config import get_config, set_config
from sleepecg.feature_extraction import extract_features, preprocess_rri
from sleepecg.heartbeats import compare_heartbeats, detect_heartbeats, rri_similarity
from sleepecg.io import *  # noqa: F403
from sleepecg.plot import plot_ecg, plot_hypnogram

__version__ = "0.5.5"
