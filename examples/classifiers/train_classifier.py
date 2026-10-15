# © SleepECG developers
#
# License: BSD (3-clause)

"""Train and evaluate the built-in classifiers.

Each classifier is trained on MESA and evaluated on SHHS. Both datasets are downloaded
from NSRR, which requires a download token (https://sleepdata.org/token) in the
environment variable `NSRR_TOKEN`.

Usage examples (run from the repository root):

    export NSRR_TOKEN=...
    uv run python examples/classifiers/train_classifier.py all
    uv run python examples/classifiers/train_classifier.py ws-gru-mesa
    uv run python examples/classifiers/train_classifier.py all --test-only

By default, classifiers are written to (and read from) `src/sleepecg/classifiers`, so
retraining overwrites the bundled classifiers.

Training runs on the CPU, which is faster than a GPU for these small models. The
recurrent layers in Keras only use a fused GPU kernel on CUDA, and with 8 units per GRU
there is little to parallelize. The script therefore sets the environment variable
`KERAS_TORCH_DEVICE=cpu` unless it is already set. To set the device explicitly, export
the variable yourself:

    export KERAS_TORCH_DEVICE=cpu

Features are cached in memory, so classifiers with identical feature extraction
parameters (currently `wrn-gru-mesa` and `wrn-gru-mesa-weighted`) share a single
extraction run per dataset when they are trained in the same invocation (`all`).
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")  # faster than GPU, see docstring

import keras
import numpy as np
from keras import layers, models
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

DEFAULT_CLASSIFIERS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "sleepecg" / "classifiers"
)
DEFAULT_SEED = 42
MASK_VALUE = -1

# in-memory cache for extract_features(), keyed by dataset and parameters
_feature_cache: dict[tuple[str, str], tuple[list, list]] = {}


@dataclass(frozen=True)
class ClassifierConfig:
    """Settings that differ between the built-in classifiers."""

    stages_mode: str
    lookback: int
    lookforward: int
    epochs: int
    use_sample_weight: bool


CLASSIFIERS = {
    "ws-gru-mesa": ClassifierConfig(
        stages_mode="wake-sleep",
        lookback=240,
        lookforward=270,
        epochs=5,
        use_sample_weight=True,
    ),
    "wrn-gru-mesa": ClassifierConfig(
        stages_mode="wake-rem-nrem",
        lookback=120,
        lookforward=150,
        epochs=25,
        use_sample_weight=False,
    ),
    "wrn-gru-mesa-weighted": ClassifierConfig(
        stages_mode="wake-rem-nrem",
        lookback=120,
        lookforward=150,
        epochs=25,
        use_sample_weight=True,
    ),
}


def _extract(
    dataset: str, params: dict, n_jobs: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract features for MESA or SHHS, reusing results for identical parameters."""
    key = (dataset, json.dumps(params, sort_keys=True))
    if key not in _feature_cache:
        reader = {"mesa": read_mesa, "shhs": read_shhs}[dataset]
        records = list(reader(offline=False))
        features, stages, _ = extract_features(tqdm(records), **params, n_jobs=n_jobs)
        _feature_cache[key] = (features, stages)
    else:
        print(f"‣‣ Reusing cached {dataset.upper()} features")
    return _feature_cache[key]


def train(
    name: str,
    config: ClassifierConfig,
    classifiers_dir: Path,
    seed: int,
    n_jobs: int,
) -> None:
    """Train a classifier on MESA and save it to `classifiers_dir`."""
    print(f"‣  Starting training of {name}...")
    keras.utils.set_random_seed(seed)  # seeds Python, NumPy, and PyTorch

    feature_extraction_params = {
        "lookback": config.lookback,
        "lookforward": config.lookforward,
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

    print("‣‣ Extracting features...")
    features_train, stages_train = _extract("mesa", feature_extraction_params, n_jobs)

    print("‣‣ Preparing data for Keras...")
    features_train_pad, stages_train_pad, sample_weight = prepare_data_keras(
        features_train,
        stages_train,
        config.stages_mode,
        mask_value=MASK_VALUE,
    )
    print_class_balance(stages_train_pad, config.stages_mode)

    print("‣‣ Defining model...")
    model = models.Sequential(
        [
            layers.Input((None, features_train_pad.shape[2])),
            layers.Masking(MASK_VALUE),
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
        epochs=config.epochs,
        sample_weight=sample_weight if config.use_sample_weight else None,
    )

    print("‣‣ Saving classifier...")
    classifiers_dir.mkdir(parents=True, exist_ok=True)
    save_classifier(
        name=name,
        model=model,
        stages_mode=config.stages_mode,
        feature_extraction_params=feature_extraction_params,
        mask_value=MASK_VALUE,
        classifiers_dir=classifiers_dir,
    )


def test(name: str, classifiers_dir: Path, n_jobs: int) -> None:
    """Evaluate a saved classifier on SHHS."""
    print(f"‣  Starting testing of {name}...")
    print("‣‣ Loading classifier...")
    clf = load_classifier(name, classifiers_dir)

    print("‣‣ Extracting features...")
    features_test, stages_test = _extract("shhs", clf.feature_extraction_params, n_jobs)

    print("‣‣ Evaluating classifier...")
    features_test_pad, stages_test_pad, _ = prepare_data_keras(
        features_test,
        stages_test,
        clf.stages_mode,
        mask_value=clf.mask_value,
    )
    y_pred = clf.model.predict(features_test_pad)
    evaluate(stages_test_pad, y_pred, clf.stages_mode)


def main(argv: list[str] | None = None) -> None:
    """Parse command line arguments and train and/or test classifiers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "name",
        choices=[*CLASSIFIERS, "all"],
        help="classifier to train and test, or 'all'",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="skip training and evaluate the classifier already saved on disk",
    )
    parser.add_argument(
        "--classifiers-dir",
        type=Path,
        default=DEFAULT_CLASSIFIERS_DIR,
        help="directory to save classifiers to and load them from "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="random seed used for training (default: %(default)s)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="number of parallel jobs for feature extraction (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("NSRR_TOKEN")
    if token is None:
        parser.error("the environment variable NSRR_TOKEN must contain an NSRR token")
    set_nsrr_token(token)

    # silence warnings (which might pop up during feature extraction)
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="HR analysis window too short"
    )

    names = list(CLASSIFIERS) if args.name == "all" else [args.name]
    for name in names:
        if not args.test_only:
            train(name, CLASSIFIERS[name], args.classifiers_dir, args.seed, args.n_jobs)
        test(name, args.classifiers_dir, args.n_jobs)


if __name__ == "__main__":
    main()
