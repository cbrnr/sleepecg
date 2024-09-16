import warnings

from tensorflow.keras import layers, models
from tqdm import tqdm

from sleepecg import (
    evaluate,
    extract_features,
    load_classifier,
    prepare_data_keras,
    print_class_balance,
    read_mesa,
    read_shhs,
    save_classifier,
    set_nsrr_token,
)

set_nsrr_token("your-token-here")

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

    print("‣‣ Preparing data for Keras...")
    stages_mode = "wake-rem-nrem"

    features_train_pad, stages_train_pad, sample_weight = prepare_data_keras(
        features_train,
        stages_train,
        stages_mode,
    )
    print_class_balance(stages_train_pad, stages_mode)

    print("‣‣ Defining model...")
    model = models.Sequential(
        [
            layers.Input((None, features_train_pad.shape[2])),
            layers.Masking(-1),
            layers.BatchNormalization(),
            layers.Dense(64),
            layers.ReLU(),
            layers.Bidirectional(layers.GRU(8, return_sequences=True)),
            layers.Bidirectional(layers.GRU(8, return_sequences=True)),
            layers.Dense(stages_train_pad.shape[-1], activation="softmax"),
        ]
    )

    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.build()
    model.summary()

    print("‣‣ Training model...")
    model.fit(
        features_train_pad,
        stages_train_pad,
        epochs=25,
        sample_weight=sample_weight,
    )

    print("‣‣ Saving classifier...")
    save_classifier(
        name="wrn-gru-mesa-weighted",
        model=model,
        stages_mode=stages_mode,
        feature_extraction_params=feature_extraction_params,
        mask_value=-1,
        classifiers_dir="./classifiers",
    )

print("‣  Starting testing...")
print("‣‣ Loading classifier...")
clf = load_classifier("wrn-gru-mesa-weighted", "./classifiers")

print("‣‣ Extracting features...")
shhs = list(read_shhs(offline=False))

features_test, stages_test, feature_ids = extract_features(
    tqdm(shhs),
    **clf.feature_extraction_params,
    n_jobs=-1,
)

print("‣‣ Evaluating classifier...")
features_test_pad, stages_test_pad, _ = prepare_data_keras(
    features_test,
    stages_test,
    clf.stages_mode,
)
y_pred = clf.model.predict(features_test_pad)
evaluate(stages_test_pad, y_pred, clf.stages_mode)
