# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for functions related to classifier training and evaluation."""

import os

import numpy as np
import pytest

from sleepecg.classification import (
    _merge_sleep_stages,
    _pad_sequences,
    _to_categorical,
    load_classifier,
    prepare_data_keras,
    save_classifier,
)

# must be set before the first `import keras`, since Keras otherwise defaults to (and
# tries to import) the tensorflow backend
os.environ.setdefault("KERAS_BACKEND", "torch")
keras = pytest.importorskip("keras")


@pytest.mark.parametrize(
    ["mode", "output"],
    [
        ("wake-sleep", [0, 1, 1, 1, 1, 2]),
        ("wake-rem-nrem", [0, 1, 1, 1, 2, 3]),
        ("wake-rem-light-n3", [0, 1, 2, 2, 3, 4]),
        ("wake-rem-n1-n2-n3", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_merge_sleep_stages(mode, output):
    """Test if the sleep stage mapping works correctly."""
    stages = [np.array([0, 1, 2, 3, 4, 5])]
    assert (_merge_sleep_stages(stages, mode)[0] == np.array(output)).all()


def test_pad_sequences():
    """Sequences are pre-padded (i.e. padding is prepended) with `value`."""
    padded = _pad_sequences([np.array([1, 2, 3]), np.array([4, 5])], value=0)
    assert np.array_equal(padded, [[1, 2, 3], [0, 4, 5]])


def test_pad_sequences_2d():
    """2D feature matrices are padded along the first axis."""
    sequences = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])]
    padded = _pad_sequences(sequences, value=-1, dtype=float)
    assert np.array_equal(padded, [[[1, 2], [3, 4]], [[-1, -1], [5, 6]]])


def test_to_categorical():
    """One-hot encoding infers the number of classes from the maximum label."""
    onehot = _to_categorical(np.array([[0, 2], [1, 0]]))
    assert onehot.shape == (2, 2, 3)
    assert np.array_equal(onehot, [[[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0]]])


def test_prepare_data_keras():
    """Pad, mask, one-hot encode, and compute sample weights for training data."""
    features = [
        np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]]),
        np.array([[7.0, 8.0]]),
    ]
    stages = [np.array([5, 2, 3]), np.array([4])]  # WAKE, N2, N1 | REM

    features_padded, stages_onehot, sample_weight = prepare_data_keras(
        features, stages, "wake-rem-nrem", mask_value=-1
    )

    assert features_padded.shape == (2, 3, 2)
    assert stages_onehot.shape == (2, 3, 4)  # UNDEFINED, NREM, REM, WAKE
    assert sample_weight.shape == (2, 3)

    # short record is pre-padded with mask_value on both features and stages
    assert np.array_equal(features_padded[1, :2], [[-1, -1], [-1, -1]])
    assert stages_onehot[1, 0, 0] == 1  # UNDEFINED one-hot

    # non-finite feature values are also replaced with mask_value
    assert np.array_equal(features_padded[0, 2], [-1, 6.0])


@pytest.mark.parametrize(
    "name", ["ws-gru-mesa", "wrn-gru-mesa", "wrn-gru-mesa-weighted"]
)
def test_load_bundled_classifier(name):
    """Bundled classifiers load and expose the metadata stored in `info.yml`."""
    clf = load_classifier(name, "SleepECG")

    assert clf.model_type == "keras"
    assert clf.mask_value == -1
    assert clf.stages_mode in {"wake-sleep", "wake-rem-nrem"}
    assert "feature_selection" in clf.feature_extraction_params
    assert clf.source_file.name == f"{name}.zip"


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        (
            "ws-gru-mesa",
            [
                [0.010579, 0.308753, 0.680669],
                [0.003326, 0.220124, 0.776550],
                [0.002070, 0.234330, 0.763600],
            ],
        ),
        (
            "wrn-gru-mesa",
            [
                [0.000296, 0.148225, 0.017080, 0.834399],
                [0.000085, 0.120967, 0.011013, 0.867935],
                [0.000044, 0.085466, 0.007049, 0.907442],
            ],
        ),
        (
            "wrn-gru-mesa-weighted",
            [
                [0.000550, 0.073161, 0.029711, 0.896579],
                [0.000119, 0.033012, 0.020463, 0.946406],
                [0.000084, 0.033300, 0.018102, 0.948513],
            ],
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_bundled_classifier_predictions(name, reference):
    """Predictions on a fixed-seed input should match values captured pre-migration.

    This guards against a Keras/backend upgrade silently changing the numerical
    behavior of the bundled, pretrained models. Tolerance is loose since torch-backend
    GRU results can differ slightly (~1e-4) across platforms and Keras versions.
    """
    clf = load_classifier(name, "SleepECG")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 50, 36)).astype("float32")

    pred = clf.model.predict(x, verbose=0)

    assert np.allclose(pred[0, :3], reference, atol=1e-3)


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_save_and_load_classifier_roundtrip(tmp_path):
    """A saved Keras model can be loaded back with all metadata intact."""
    model = keras.Sequential(
        [keras.layers.Input((None, 3)), keras.layers.Dense(2, activation="softmax")]
    )
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    # fit once so the optimizer's variables are fully built before saving, matching how
    # real classifiers (trained via model.fit()) are saved
    rng = np.random.default_rng(0)
    model.fit(
        rng.standard_normal((2, 5, 3)).astype("float32"),
        _to_categorical(np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])),
        verbose=0,
    )

    feature_extraction_params = {
        "feature_selection": ["hrv-time"],
        "lookback": 0,
        "lookforward": 30,
    }

    save_classifier(
        name="test-clf",
        model=model,
        stages_mode="wake-sleep",
        feature_extraction_params=feature_extraction_params,
        mask_value=-1,
        classifiers_dir=tmp_path,
    )
    assert (tmp_path / "test-clf.zip").exists()

    clf = load_classifier("test-clf", classifiers_dir=tmp_path)

    assert clf.model_type == "keras"
    assert clf.mask_value == -1
    assert clf.stages_mode == "wake-sleep"
    assert clf.feature_extraction_params == feature_extraction_params
    assert clf.source_file == tmp_path / "test-clf.zip"

    x = np.random.default_rng(0).standard_normal((1, 5, 3)).astype("float32")
    assert clf.model.predict(x, verbose=0).shape == (1, 5, 2)
