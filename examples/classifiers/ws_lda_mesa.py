import warnings


import numpy as np
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

from tqdm import tqdm

from sleepecg import (
    evaluate,
    extract_features,
    load_classifier,
    prepare_data_sklearn,
    print_class_balance,
    read_mesa,
    read_shhs,
    save_classifier,
    set_nsrr_token,
)

set_nsrr_token("YOUR TOKEN HERE")

TRAIN = True  # set to False to skip training and load classifier from disk

# silence warnings (which might pop up during feature extraction)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="HR analysis window too short"
)

if TRAIN:
    print("‣  Starting training...")
    print("‣‣ Extracting features...")
    records = list(read_mesa(offline=True, data_dir="D:\SleepData", activity_source="actigraphy"))

    feature_extraction_params = {
        "lookback": 240,
        "lookforward": 270,
        "feature_selection": [
            "hrv-time",
            "hrv-frequency",
            "recording_start_time",
            "age",
            "gender",
            "activity_counts"
        ],
        "min_rri": 0.3,
        "max_rri": 2,
        "max_nans": 0.5,
    }

    features, stages, feature_ids = extract_features(
        tqdm(records),
        **feature_extraction_params,
        n_jobs=-1,
    )

    features_train, features_test, stages_train, stages_test = train_test_split(features,
                                                                                stages,
                                                                                test_size=0.2)

    print("‣‣ Preparing data for Sklearn...")
    stages_mode = "wake-sleep"

    features_train_pad, stages_train_pad, sample_weight = prepare_data_sklearn(
        features_train,
        stages_train,
        stages_mode,
    )
    print_class_balance(stages_train_pad, stages_mode)

    print("‣‣ Defining model...")
    pipe = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        sklearn.svm.SVC(),
        verbose=False,
    )

    print("‣‣ Training model...")

    pipe.fit(
        X=features_train_pad,
        y=stages_train_pad,
    )

print("‣  Starting testing...")

features_test_pad, stages_test_pad, sample_weight = prepare_data_sklearn(
        features_test,
        stages_test,
        stages_mode,
    )
y_pred = pipe.predict(features_test_pad)
evaluate(stages_test_pad, y_pred, stages_mode)