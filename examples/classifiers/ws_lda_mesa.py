import warnings


import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

set_nsrr_token("25042-5JxoCwc8KQ3uV3ubyK-D")

TRAIN = True  # set to False to skip training and load classifier from disk

# silence warnings (which might pop up during feature extraction)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="HR analysis window too short"
)

if TRAIN:
    print("‣  Starting training...")
    print("‣‣ Extracting features...")
    records = list(read_mesa(offline=False, activity_source="actigraphy", records_pattern="000*"))

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

    features_train, stages_train, feature_ids = extract_features(
        tqdm(records),
        **feature_extraction_params,
        n_jobs=-1,
    )

    print("‣‣ Preparing data for Sklearn...")
    stages_mode = "wake-sleep"

    features_train_pad, stages_train_pad, sample_weight = prepare_data_sklearn(
        features_train,
        stages_train,
        feature_ids,
        stages_mode,
    )
    print_class_balance(stages_train_pad, stages_mode)

    print("‣‣ Defining model...")