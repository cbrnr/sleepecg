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
    records = list(read_mesa(offline=False))

    feature_extraction_params = {
        "lookback": 120,
        "lookforward": 150,
        "feature_selection": [
            "hrv-time",
            "hrv-frequency",
            "recording_start_time",
            "age",
            "gender",
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
        name="ws-sklearn-mesa",
        model=pipe,
        stages_mode=stages_mode,
        feature_extraction_params=feature_extraction_params,
        mask_value=-1,
        classifiers_dir="./classifiers",
    )

print("‣  Starting testing...")
print("‣‣ Loading classifier...")
clf = load_classifier("ws-sklearn-mesa", "./classifiers")

print("‣‣ Extracting features...")
shhs = list(read_shhs(offline=False))

features_test, stages_test, feature_ids = extract_features(
    tqdm(shhs),
    **clf.feature_extraction_params,
    n_jobs=-2,
)

print("‣‣ Evaluating classifier...")
features_test_pad, stages_test_pad, record_ids_test = prepare_data_sklearn(
    features_test,
    stages_test,
    clf.stages_mode,
)

y_pred = clf.model.predict(features_test_pad)
evaluate(stages_test_pad, y_pred, clf.stages_mode)
