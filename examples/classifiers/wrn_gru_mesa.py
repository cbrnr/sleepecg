# %%
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

# %% Read data and extract features
set_nsrr_token("your-token-here")
records = list(read_mesa())

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
    n_jobs=-2,
)

# %% Merge sleep stages, pad and mask data as preparation for keras NN
stages_mode = "wake-rem-nrem"

features_train_pad, stages_train_pad, _ = prepare_data_keras(
    features_train,
    stages_train,
    stages_mode,
)
print_class_balance(stages_train_pad, stages_mode)

# %% Define and train model
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

# %% Train model
model.fit(
    features_train_pad,
    stages_train_pad,
    epochs=25,
)

# %% Store classifier
save_classifier(
    name="wrn-gru-mesa",
    model=model,
    stages_mode=stages_mode,
    feature_extraction_params=feature_extraction_params,
    mask_value=-1,
    classifiers_dir="./classifiers",
)

# %% Load classifier from disk for validation
clf = load_classifier("wrn-gru-mesa", "./classifiers")

# %% Read data and extract features
shhs = list(read_shhs())

features_test, stages_test, feature_ids = extract_features(
    tqdm(shhs),
    **clf.feature_extraction_params,
    n_jobs=-2,
)

# %% Predict & evaluate
features_test_pad, stages_test_pad, _ = prepare_data_keras(
    features_test,
    stages_test,
    clf.stages_mode,
)
y_pred = clf.model.predict(features_test_pad)
evaluate(stages_test_pad, y_pred, clf.stages_mode)
