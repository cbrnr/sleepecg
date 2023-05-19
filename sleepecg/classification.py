# © SleepECG developers
#
# License: BSD (3-clause)

"""Functions related to classifier training and evaluation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Protocol
from zipfile import ZipFile

import numpy as np
import yaml

from sleepecg.config import get_config
from sleepecg.feature_extraction import extract_features
from sleepecg.io.sleep_readers import SleepRecord, SleepStage
from sleepecg.utils import _STAGE_NAMES, _merge_sleep_stages


def prepare_data_keras(
    features: list[np.ndarray],
    stages: list[np.ndarray],
    stages_mode: str,
    mask_value: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mask and pad data and calculate sample weights for a Keras model.

    The following steps are performed:

    - Merge sleep stages in `stages` according to `stage_mode`.
    - Set features corresponding to `SleepStage.UNDEFINED` to `mask_value`.
    - Replace `np.nan` and `np.inf` in `features` with `mask_value`.
    - Pad to a common length, where `mask_value` is used for `features` and
      `SleepStage.UNDEFINED` (i.e `0`) is used for stages.
    - One-hot encode stages.
    - Calculate sample weights with class weights taken as `n_samples /
      (n_classes * np.bincount(y))`.

    Parameters
    ----------
    features : list[np.ndarray]
        Each 2D array in this list is a feature matrix of shape `(n_samples, n_features)`
        corresponding to a single record as returned by `extract_features()`.
    stages : list[np.ndarray]
        Each 1D array in this list contains the sleep stages of a single record as returned
        by `extract_features()`.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    mask_value : int, optional
        Value used to pad features and replace `np.nan` and `np.inf`, by default `-1`.
        Remember to pass the same value to `layers.Masking` in your model.

    Returns
    -------
    features_padded : np.ndarray
        A 3D array of shape `(n_records, max_n_samples, n_features)`, where `n_records` is
        the length of `features`/`stages` and `max_n_samples` is the maximum number of rows
        of all feature matrices in `features`.
    stages_padded_onehot : np.ndarray
        A 3D array of shape `(n_records, max_n_samples, n_classes+1)`, where `n_classes` is
        the number of classes remaining after merging sleep stages (excluding
        `SleepStage.UNDEFINED`).
    sample_weight : np.ndarray
        A 2D array of shape `(n_records, max_n_samples)`.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    stages_merged = _merge_sleep_stages(stages, stages_mode)
    stages_padded = pad_sequences(stages_merged, value=SleepStage.UNDEFINED)
    stages_padded_onehot = to_categorical(stages_padded)

    features_padded = pad_sequences(features, dtype=float, value=mask_value)
    features_padded[stages_padded == SleepStage.UNDEFINED, :] = mask_value
    features_padded[~np.isfinite(features_padded)] = mask_value

    stage_counts = stages_padded_onehot.sum(0).sum(0)
    # samples corresponding to SleepStage.UNDEFINED are ignored, so their count shouldn't
    # influence the class weights -> slice with [1:]
    class_weight = np.sum(stage_counts[1:]) / stage_counts
    sample_weight = class_weight[stages_padded]

    return features_padded, stages_padded_onehot, sample_weight


def print_class_balance(stages: np.ndarray, stages_mode: Optional[str] = None) -> None:
    """
    Print the number of samples and percentages of each class in `stages`.

    Parameters
    ----------
    stages : np.ndarray
        A 2D array of shape `(n_records, n_samples)` containing integer class labels or a
        3D array of shape `(n_records, n_samples, n_classes)` containing one-hot encoded
        class labels.
    stages_mode : str, optional
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`. If `None` (default), no class labels
        are printed.
    """
    if stages.ndim == 3:
        stages = stages.argmax(2)

    if stages_mode is not None:
        stage_names = ["UNDEFINED"] + _STAGE_NAMES[stages_mode]
    else:
        stage_names = [str(n) for n in range(6)]

    print("Class balance:")

    unique_stages, counts = np.unique(stages, return_counts=True)
    max_len_counts = len(str(max(counts)))
    max_len_stages = max(len(str(s)) for s in stage_names)
    total_count = counts.sum()
    for stage, count, fraction in zip(unique_stages, counts, counts / total_count):
        print(
            f"    {stage_names[stage]:>{max_len_stages}}: {count:{max_len_counts}} "
            f"({fraction:3.0%})"
        )


def save_classifier(
    name: str,
    model: Any,
    stages_mode: str,
    feature_extraction_params: dict[str, Any],
    mask_value: Optional[int] = None,
    classifiers_dir: Optional[str | Path] = None,
) -> None:
    """
    Save a trained classifier to disk.

    The `model` itself and a `.yml` file containing classifier metadata are stored as
    `<name>.zip` in `classifiers_dir`. Model serialization is performed as suggested by the
    respective package documentation. Currently only Keras models are supported.

    Parameters
    ----------
    name : str
        An identifier which is used as the filename.
    model : Any
        The classification model, should have `fit()` and `predict()` methods.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, or `'wake-rem-n1-n2-n3'`.
    feature_extraction_params : dict[str, typing.Any]
        The parameters passed to `extract_features()`, as a dictionary mapping string
        parameter names to values. Should not include `records` and `n_jobs`.
    mask_value : int, optional
        Only required for Keras models, as passed to `prepare_data_keras()` and
        `keras.layers.Masking`, by default `None`.
    classifiers_dir : str | pathlib.Path, optional
        Directory in which the `.zip` file is stored. If `None` (default), the value is
        taken from the configuration.

    See Also
    --------
    load_classifier : Load classifiers saved with this function.
    """
    if classifiers_dir is None:
        classifiers_dir = get_config("classifiers_dir")

    target_file = Path(classifiers_dir).expanduser() / name

    model_type = model.__module__.split(".")[0]
    classifier_info = {
        "model_type": model_type,
        "stages_mode": stages_mode,
        "feature_extraction_params": feature_extraction_params,
    }
    if mask_value is not None:
        classifier_info["mask_value"] = mask_value

    with TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/info.yml", "w") as infofile:
            yaml.dump(classifier_info, infofile)

        if model_type == "keras":
            model.save(f"{tmpdir}/classifier")
        else:
            raise ValueError(f"Saving model of type {type(model)} is not supported")

        shutil.make_archive(str(target_file), "zip", tmpdir)


class _Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class SleepClassifier:
    """
    Store a sleep classifier model and metadata.

    Attributes
    ----------
    model : _Model
        The classification model, should have `fit` and `predict` methods.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, or `'wake-rem-n1-n2-n3'`.
    feature_extraction_params : dict[str, typing.Any]
        The parameters passed to `extract_features()`, as a dictionary mapping string
        parameter names to values. Does not include `records` and `n_jobs`.
    model_type : str
        A string identifying the model type, e.g. `'keras'` or `'sklearn'`. This is used by
        `stage()` to determine how to perform sleep stage predictions.
    mask_value : int, optional
        Only required for models of type `'keras'`, as passed to `prepare_data_keras()` and
        `keras.layers.Masking`, by default `None`.
    source_file : pathlib.Path, optional
        The file from which the classifier was loaded using `load_classifier()`, by default
        `None`.
    """

    model: _Model
    stages_mode: str
    feature_extraction_params: dict[str, Any]
    model_type: str
    mask_value: Optional[int] = None
    source_file: Optional[Path] = None

    def __repr__(self) -> str:
        if self.source_file is not None:
            return (
                f"<SleepClassifier | {self.stages_mode}, {self.model_type}, "
                f"{self.source_file.name}>"
            )
        return f"<SleepClassifier | {self.stages_mode}, {self.model_type}>"

    def __str__(self) -> str:
        features = ", ".join(self.feature_extraction_params["feature_selection"])
        return (
            f"SleepClassifier for {self.stages_mode.upper()}\n"
            f"    features: {features}\n"
            f"    model type: {self.model_type}\n"
            f"    source file: {self.source_file}\n"
        )


def load_classifier(
    name: str,
    classifiers_dir: Optional[str | Path] = None,
    silence_tf_messages: bool = True,
) -> SleepClassifier:
    """
    Load a `SleepClassifier` from disk.

    This functions reads `.zip` files saved by `save_classifier()`. Pass `'SleepECG'` as the
    second argument to load a classifier bundled with SleepECG.

    Parameters
    ----------
    name : str
        The identifier of the classifier to load.
    classifiers_dir : str | pathlib.Path, optional
        Directory in which to look for `<name>.zip`. If `None` (default), the value is taken
        from the configuration. If `'SleepECG'`, load classifiers from
        `site-packages/sleepecg/classifiers`.
    silence_tf_messages : bool, optional
        Whether or not to silence messages from TensorFlow when loading a model. By default
        `True`.

    Returns
    -------
    SleepClassifier
        Contains the model and metadata required for feature extraction and preprocessing.
        Can be passed to `stage()`.

    See Also
    --------
    list_classifiers : Show information about available classifiers.
    """
    if classifiers_dir == "SleepECG":
        classifiers_dir = Path(__file__).parent / "classifiers"
    elif classifiers_dir is None:
        classifiers_dir = get_config("classifiers_dir")

    soure_file = Path(classifiers_dir).expanduser() / f"{name}.zip"

    with TemporaryDirectory() as tmpdir:
        shutil.unpack_archive(soure_file, tmpdir)

        with open(f"{tmpdir}/info.yml") as infofile:
            classifier_info = yaml.safe_load(infofile)

        if classifier_info["model_type"] == "keras":
            import os

            environ_orig = os.environ.copy()
            if silence_tf_messages:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

            from tensorflow import keras

            try:
                classifier = keras.models.load_model(f"{tmpdir}/classifier")
            finally:
                os.environ.clear()
                os.environ.update(environ_orig)

        else:
            raise ValueError(
                f"Loading model of type {classifier_info['model_type']} is not supported"
            )

    return SleepClassifier(
        model=classifier,
        source_file=soure_file,
        **classifier_info,
    )


def list_classifiers(classifiers_dir: Optional[str | Path] = None) -> None:
    """
    List available classifiers.

    Parameters
    ----------
    classifiers_dir : str | pathlib.Path, optional
        Directory in which to look for classifiers. If `None` (default), the value is taken
        from the configuration. If `'SleepECG'`, `site-packages/sleepecg/classifiers` is
        used.

    See Also
    --------
    load_classifier : Load classifiers.
    """
    if classifiers_dir == "SleepECG":
        classifiers_dir = Path(__file__).parent / "classifiers"
        print("Classifiers in SleepECG:")
    elif classifiers_dir is None:
        classifiers_dir = get_config("classifiers_dir")
        print(f"Classifiers in {classifiers_dir}:")
    else:
        print(f"Classifiers in {classifiers_dir}:")

    classifiers_dir = Path(classifiers_dir).expanduser()

    for classifier_filepath in classifiers_dir.glob("*.zip"):
        with ZipFile(classifier_filepath, "r") as zip_file:
            with zip_file.open("info.yml") as infofile:
                classifier_info = yaml.safe_load(infofile)
                features = ", ".join(
                    classifier_info["feature_extraction_params"]["feature_selection"]
                )
                print(
                    f"  {classifier_filepath.stem}\n"
                    f"      stages_mode: {classifier_info['stages_mode'].upper()}\n"
                    f"      model type: {classifier_info['model_type']}\n"
                    f"      features: {features}\n"
                )


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, N: int) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    N : int
        Number of unique classes.

    Returns
    -------
    np.ndarray
        Confusion matrix whose i-th row and j-th column entry indicates the number of
        samples with true label being i-th class and predicted label being j-th class.
    """
    return np.bincount(N * y_true + y_pred, minlength=N * N).reshape(N, N)


def _cohen_kappa(confmat: np.ndarray) -> float:
    """
    Compute Cohen's kappa (which measures inter-annotator agreement).

    Implementation modified from `sklearn.metrics.cohen_kappa_score`.

    Parameters
    ----------
    confmat : np.ndarray
        A confusion matrix, as returned by `confusion_matrix()`.

    Returns
    -------
    float
        The kappa statistic, which is a number between -1 and 1. The maximum value means
        complete agreement; zero or lower means chance agreement.
    """
    n_classes = confmat.shape[0]
    sum0 = np.sum(confmat, axis=0)
    sum1 = np.sum(confmat, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = 1 - np.eye(n_classes, dtype=int)
    k: float = np.sum(w_mat * confmat) / np.sum(w_mat * expected)
    return 1 - k


def evaluate(
    stages_true: np.ndarray,
    stages_pred: np.ndarray,
    stages_mode: str,
    show_undefined: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Evaluate the performance of a sleep stage classifier.

    Prints overall accuracy, Cohen's kappa, confusion matrix, per-class precision, recall,
    and F1 score.

    Parameters
    ----------
    stages_true : np.ndarray
        The annotated (ground truth) sleep stages as a 2D array of shape
        `(n_records, n_samples)` containing integer class labels, or a 3D array of shape
        `(n_records, n_samples, n_classes)` containing one-hot encoded class labels.
    stages_pred : np.ndarray
        The predicted sleep stages as a 2D array of shape `(n_records, n_samples)`
        containing integer class labels, or a 3D array of shape
        `(n_records, n_samples, n_classes)` containing one-hot encoded class labels.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    show_undefined : bool, optional
        If `True`, include `SleepStage.UNDEFINED` (i.e `0`) in the confusion matrix output.
        This can be helpful during debugging. By default `False`.

    Returns
    -------
    conf_mat : np.ndarray
        Confusion matrix.
    stage_names : list[str]
        Sleep stage names.
    """
    stage_names = _STAGE_NAMES[stages_mode]

    if stages_true.ndim == 3:
        stages_true = stages_true.argmax(2)
    if stages_pred.ndim == 3:
        stages_pred = stages_pred.argmax(2)

    confmat_full = _confusion_matrix(
        stages_true.ravel(),
        stages_pred.ravel(),
        len(stage_names) + 1,
    )
    confmat = confmat_full[1:, 1:]

    print(f"Confusion matrix ({stages_mode.upper()}):")
    if show_undefined:
        print(confmat_full)
    else:
        print(confmat)

    kappa = _cohen_kappa(confmat)

    acc = confmat.trace() / confmat.sum()
    tp = np.diag(confmat)
    fp = confmat.sum(1) - tp
    fn = confmat.sum(0) - tp
    precision = tp / (tp + fn)
    recall = tp / (tp + fp)
    f1 = 2 / (recall**-1 + precision**-1)
    support = confmat.sum(1)

    print(f"Accuracy: {acc:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print("       precision    recall  f1-score    support")
    for i, stage_name in enumerate(stage_names):
        print(
            f"{stage_name:>5}{precision[i]:11.2f}{recall[i]:10.2f}{f1[i]:10.2f}"
            f"{support[i]:11}"
        )
    print(f"{support.sum():47}")

    return confmat_full, stage_names


def stage(
    clf: SleepClassifier,
    record: SleepRecord,
    return_mode: str = "int",
) -> np.ndarray:
    """
    Predict sleep stages for a single record.

    Feature extraction and preprocessing are performed according to the information stored
    in `clf`.

    Parameters
    ----------
    clf : SleepClassifier
        A classifier object as loaded with `load_classifier()`.
    record : SleepRecord
        A single record (i.e. night).
    return_mode : str, optional
        If `'int'`, return the predicted sleep stages as a 1D array of integers. If
        `'prob'`, return a 2D array of probabilities. If `'str'`, return a 1D array of
        strings.

    Returns
    -------
    np.ndarray
        Array of sleep stages. Depending on `return_mode`, this takes different forms.

    Warnings
    --------
    Note that the returned labels depend on `clf.stages_mode`, so they do not necessarily
    follow the stage-to-integer mapping defined in `SleepStage`. See
    [classification](../classification.md) for details.
    """
    return_modes = {"int", "prob", "str"}
    if return_mode not in return_modes:
        raise ValueError(
            f"Invalid return_mode: {return_mode!r}. Possible options: {return_modes}"
        )

    stage_names = ["UNDEFINED"] + _STAGE_NAMES[clf.stages_mode]

    features = extract_features(records=[record], **clf.feature_extraction_params)[0][0]
    if clf.model_type == "keras":
        features[~np.isfinite(features)] = clf.mask_value
        stages_pred_proba: np.ndarray = clf.model.predict(features[np.newaxis, ...])[0]
        stages_pred: np.ndarray = stages_pred_proba.argmax(-1)
    else:
        raise ValueError(f"Staging with model of type {type(clf)} is not supported")

    if return_mode == "prob":
        return stages_pred_proba
    elif return_mode == "str":
        return np.array([stage_names[s] for s in stages_pred])
    return stages_pred
