import warnings

import sklearn
from sklearn.impute import SimpleImputer
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
    save_classifier,
    set_nsrr_token,
)

set_nsrr_token("your-token-here")

TRAIN = False  # set to False to skip training and load classifier from disk

# silence warnings (which might pop up during feature extraction)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="HR analysis window too short"
)

if TRAIN:
    print("‣  Starting training...")
    print("‣‣ Extracting features...")

    feature_extraction_params = {
        "lookback": 240,
        "lookforward": 270,
        "feature_selection": [
            "hrv-time",
            "hrv-frequency",
            "recording_start_time",
            "age",
            "gender",
            "activity_counts",
        ],
        "min_rri": 0.3,
        "max_rri": 2,
        "max_nans": 0.5,
    }
    records_train = (
        list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="0*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="1*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="2*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="3*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="4*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="50*",
            )
        )
        + list(
            read_mesa(
                offline=False,
               activity_source="actigraphy",
                records_pattern="51*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="52*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="53*",
            )
        )
        + list(
            read_mesa(
                offline=False,
                activity_source="actigraphy",
                records_pattern="54*",
            )
        )
    )

    features_train, stages_train, feature_ids_train = extract_features(
        tqdm(records_train),
        **feature_extraction_params,
        n_jobs=-1,
    )

    print("‣‣ Preparing data for Sklearn...")
    stages_mode = "wake-sleep"

    features_train_pad, stages_train_pad, record_ids = prepare_data_sklearn(
        features_train,
        stages_train,
        stages_mode,
    )
    print_class_balance(stages_train_pad, stages_mode)

    print("‣‣ Defining model...")
    pipe = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        sklearn.ensemble.RandomForestClassifier(),
        verbose=False,
    )

    print("‣‣ Training model...")

    pipe.fit(
        X=features_train_pad,
        y=stages_train_pad,
    )

    print("‣‣ Saving model...")
    save_classifier(
        name="ws-sklearn-mesa-actigraphy",
        model=pipe,
        stages_mode=stages_mode,
        feature_extraction_params=feature_extraction_params,
        mask_value=-1,
        classifiers_dir="./classifiers",
    )

print("‣  Starting testing...")
print("‣‣ Loading classifier...")
clf = load_classifier("ws-sklearn-mesa-actigraphy", "./classifiers")
stages_mode = clf.stages_mode

print("‣‣ Extracting features...")
records_test = (
    list(
        read_mesa(
            offline=False,
            activity_source="actigraphy",
            records_pattern="55*",
        )
    )
    + list(
        read_mesa(
            offline=False,
           activity_source="actigraphy",
            records_pattern="56*",
        )
    )
    + list(
        read_mesa(
            offline=False,
            activity_source="actigraphy",
            records_pattern="57*",
        )
    )
    + list(
        read_mesa(
            offline=False,
            activity_source="actigraphy",
            records_pattern="58*",
        )
    )
    + list(
        read_mesa(
            offline=False,
            activity_source="actigraphy",
            records_pattern="59*",
        )
    )
    + list(
        read_mesa(
            offline=False,
            activity_source="actigraphy",
            records_pattern="6*",
        )
    )
)

features_test, stages_test, feature_ids_test = extract_features(
    tqdm(records_test),
    **clf.feature_extraction_params,
    n_jobs=-1,
)

features_test_pad, stages_test_pad, record_ids_test = prepare_data_sklearn(
    features_test,
    stages_test,
    stages_mode,
)

y_pred = clf.model.predict(features_test_pad)
evaluate(stages_test_pad, y_pred, stages_mode)
