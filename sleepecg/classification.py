# © SleepECG developers
#
# License: BSD (3-clause)

"""Functions related to classifier training and evaluation."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import yaml

from .config import get_config
from .feature_extraction import extract_features
from .io.sleep_readers import SleepRecord, SleepStage
from .utils import _time_to_sec

# Classifiers don't always discriminate between all sleep stages defined by the AASM
# guidelines. This dictionary is used to create a consistent mapping from groups of AASM
# sleep stages (as defined in `SleepStage`) to integers. `SleepStage.UNDEFINED` is always
# `0` and the actual stages' values increase with wakefulness, so they map correctly to the
# y-axis in a hypnogram plot. Gaps between stage values are avoided as non-existing classes
# in a one-hot encoding leads to issues when calculating class weights and losses.

_SLEEP_STAGE_MAPPING = {
    "wake-sleep": {
        SleepStage.WAKE: 2,
        SleepStage.REM: 1,
        SleepStage.N1: 1,
        SleepStage.N2: 1,
        SleepStage.N3: 1,
    },
    "wake-rem-nrem": {
        SleepStage.WAKE: 3,
        SleepStage.REM: 2,
        SleepStage.N1: 1,
        SleepStage.N2: 1,
        SleepStage.N3: 1,
    },
    "wake-rem-light-n3": {
        SleepStage.WAKE: 4,
        SleepStage.REM: 3,
        SleepStage.N1: 2,
        SleepStage.N2: 2,
        SleepStage.N3: 1,
    },
    "wake-rem-n1-n2-n3": {
        SleepStage.WAKE: 5,
        SleepStage.REM: 4,
        SleepStage.N1: 3,
        SleepStage.N2: 2,
        SleepStage.N3: 1,
    },
}

# These two dicts are used for plotting and labeling of evaluation results
_STAGE_INTS = {k: sorted(set(v.values())) for k, v in _SLEEP_STAGE_MAPPING.items()}
_STAGE_NAMES = {m: m.upper().split("-")[::-1] for m in _SLEEP_STAGE_MAPPING}


def _merge_sleep_stages(stages: List[np.ndarray], stages_mode: str) -> List[np.ndarray]:
    """
    Merge sleep stage labels into groups.

    Parameters
    ----------
    stages : list[np.ndarray]
        A list of 1d-arrays containing AASM sleep stages as defined by `SleepStage`, e.g. as
        returned by :func:`extract_features`.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.

    Returns
    -------
    list[np.ndarray]
        A list of 1d-arrays containing merged sleep stages.
    """
    if stages_mode not in _SLEEP_STAGE_MAPPING:
        options = list(_SLEEP_STAGE_MAPPING.keys())
        raise ValueError(f"Invalid stages_mode: {stages_mode}. Possible options: {options}")

    new_stages = []
    for array in stages:
        new_array = np.full_like(array, fill_value=SleepStage.UNDEFINED)
        for source_stage, target_stage in _SLEEP_STAGE_MAPPING[stages_mode].items():
            new_array[array == source_stage] = target_stage
        new_stages.append(new_array)
    return new_stages


def prepare_data_keras(
    features: List[np.ndarray],
    stages: List[np.ndarray],
    stages_mode: str,
    mask_value: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mask and pad data and calculate sample weights for a keras model.

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
        Each 2d-array in this list is a feature matrix of shape `(n_samples, n_features)`
        corresponding to a single record, as returned by :func:`extract_features`.
    stages : list[np.ndarray]
        Each 1d-array in this list contains the sleep stages of a single record, as returned
        by :func:`extract_features`.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    mask_value : int, optional
        Value used to pad features and replace `np.nan` and `np.inf`, by default `-1`.
        Remember to pass the same value to `layers.Masking` in your model.

    Returns
    -------
    features_padded : np.ndarray
        A 3d-array of shape `(n_records, max_n_samples, n_features)`, where `n_records` is
        the length of `features`/`stages` and `max_n_samples` is the maximum number of rows
        of all feature matrices in `features`.
    stages_padded_onehot : np.ndarray
        A 3d-array of shape `(n_records, max_n_samples, n_classes+1)`, where `n_classes` is
        the number of classes remaining after merging sleep stages (excluding
        `SleepStage.UNDEFINED`).
    sample_weight : np.ndarray
        A 2d-array of shape `(n_records, max_n_samples)`.
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
        A 2d-array of shape `(n_records, n_samples)` containing integer class labels or a
        3d-array of shape `(n_records, n_samples, n_classes)` containing one-hot encoded
        class labels.
    stages_mode : str, optional
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`. If `None` (default), no class labels
        are printed.
    """
    if stages.ndim == 3:
        stages = stages.argmax(2)

    if stages_mode is not None:
        stage_names = stage_names = ["UNDEFINED"] + _STAGE_NAMES[stages_mode]
    else:
        stage_names = np.arange(6)

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
    name: Union[str, Path],
    model: Any,
    stages_mode: str,
    feature_extraction_params: Dict[str, Any],
    mask_value: Optional[int] = None,
    classifiers_dir: Optional[Union[str, Path]] = None,
):
    """
    Save a trained classifier to disk.

    The `model` itself and a `.yml` file containing the classifier metadata are stored as
    `<name>.zip` in `classifiers_dir`. Model serialization is performed as suggested by the
    respective package documentation. Currently only keras models are supported.

    Parameters
    ----------
    name : str | pathlib.Path
        An identifier which is used as the filename.
    model : Any
        The classification model, should have `fit` and `predict` methods.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    feature_extraction_params : dict[str, typing.Any]
        The parameters passed to :func:`extract_features`, as a dictionary mapping string
        parameter names to values. Should not include `records` and `n_jobs`.
    mask_value : int, optional
        Only required for keras models, as passed to `prepare_data_keras` and
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

        shutil.make_archive(target_file, "zip", tmpdir)


@dataclass
class SleepClassifier:
    """
    Store a sleep classifier model and metadata.

    Attributes
    ----------
    model : typing.Any
        The classification model, should have `fit` and `predict` methods.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    feature_extraction_params : dict[str, typing.Any]
        The parameters passed to :func:`extract_features`, as a dictionary mapping string
        parameter names to values. Does not include `records` and `n_jobs`.
    model_type : str
        A string identifying the model type, e.g. `'keras'` or `'sklearn'`. This is used by
        :func:`stage` to determine how to perform sleep stage predictions.
    mask_value : int, optional
        Only required for models of type `'keras'`, as passed to `prepare_data_keras` and
        `keras.layers.Masking`, by default `None`.
    source_file : pathlib.Path, optional
        The file from which the classifier was loaded using :func:`load_classifier`, by
        default `None`.
    """

    model: Any
    stages_mode: str
    feature_extraction_params: Dict[str, Any]
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
    classifiers_dir: Optional[Union[str, Path]] = None,
) -> SleepClassifier:
    """
    Load a `SleepClassifier` from disk.

    This functions reads `.zip` files saved by :func:`save_classifier`. Pass `'SleepECG'` as
    a second argument to load a classifier bundled with SleepECG.

    Parameters
    ----------
    name : str
        The identifier of the classifier to load.
    classifiers_dir : str | pathlib.Path, optional
        Directory in which to look for `<name>.zip`. If `None` (default), the value is taken
        from the configuration. If `'SleepECG'`, load classifiers from
        `site-packages/sleepecg/classifiers`.

    Returns
    -------
    SleepClassifier
        Contains the model and metadata required for feature extraction and preprocessing.
        Can be passed to :func:`stage`.

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
            from tensorflow import keras

            classifier = keras.models.load_model(f"{tmpdir}/classifier")

        else:
            raise ValueError(
                f"Loading model of type {classifier_info['model_type']} is not supported"
            )

    return SleepClassifier(
        model=classifier,
        source_file=soure_file,
        **classifier_info,
    )


def list_classifiers(classifiers_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Show information about available classifiers.

    Pass `'SleepECG'` as a second argument to list the classifiers bundled with SleepECG.

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
        A confusion matrix, as returned by :func:`confusion_matrix`.

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
    k = np.sum(w_mat * confmat) / np.sum(w_mat * expected)
    return 1 - k


def _plot_confusion_matrix(confmat: np.ndarray, stage_names: List[str]):
    """
    Create a labeled plot of a confusion matrix.

    Parameters
    ----------
    confmat : np.ndarray
        A confusion matrix, as returned by :func:`confusion_matrix`.
    stage_names : list[str]
        Class labels which are used as tick labels.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the confusion matrix plot.
    """
    import matplotlib.pyplot as plt

    classes = np.arange(len(confmat))

    fig, ax = plt.subplots()
    ax.imshow(confmat, cmap="Blues", vmin=0, vmax=confmat[1:, 1:].max())
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            ax.text(j, i, f"{confmat[i, j]}", ha="center", va="center", color="k")

    ax.set_ylabel("Annotated Stage")
    ax.set_xlabel("Predicted Stage")
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(stage_names)
    ax.set_yticklabels(stage_names)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    return fig


def evaluate(
    stages_true: np.ndarray,
    stages_pred: np.ndarray,
    stages_mode: str,
    show_undefined: bool = False,
) -> None:
    """
    Evaluate the performance of a sleep stage classifier.

    Prints overall accuracy, Cohen's kappa, confusion matrix and per-class precision, recall
    and F1 score. In an interactive environment, the confusion matrix is additionally shown
    as a labeled plot.

    Parameters
    ----------
    stages_true : np.ndarray
        The annotated ('ground truth') sleep stages as a 2d-array of shape
        `(n_records, n_samples)` containing integer class labels or a 3d-array of shape
        `(n_records, n_samples, n_classes)` containing one-hot encoded class labels.
    stages_pred : np.ndarray
        The predicted sleep stages as a 2d-array of shape `(n_records, n_samples)`
        containing integer class labels or a 3d-array of shape
        `(n_records, n_samples, n_classes)` containing class probabilities.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    show_undefined : bool, optional
        If `True`, include `SleepStage.UNDEFINED` (i.e `0`) in the confusion matrix output
        and plot. This can be helpful during debugging. By default `False`.
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
        _plot_confusion_matrix(confmat_full, ["UNDEFINED"] + stage_names)
        print(confmat_full)
    else:
        _plot_confusion_matrix(confmat, stage_names)
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
        A classifier object as loaded with :func:`load_classifier`.
    record : SleepRecord
        A single record (i.e. night).
    return_mode : str, optional
        If `'int'`, return the predicted sleep stages as a 1d-array of integers. If
        `'prob'`, return a 2d-array of probabilities. If `'str'`, return a 1d-array of
        strings.

    Returns
    -------
    np.ndarray
        A array of sleep stages. Depending on `return_mode`, this takes different forms.

    Warnings
    --------
    Note that the returned labels depend on `clf.stages_mode`, so they do not necessarily
    follow the stage-to-integer mapping defined in :class:`SleepStage`. See
    :ref:`classification` for details.
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
        stages_pred_proba = clf.model.predict(features[np.newaxis, ...])[0]
        stages_pred = stages_pred_proba.argmax(-1)
    else:
        raise ValueError(f"Staging with model of type {type(clf)} is not supported")

    if return_mode == "prob":
        return stages_pred_proba
    elif return_mode == "str":
        return np.array([stage_names[s] for s in stages_pred])
    return stages_pred


def plot_hypnogram(
    record: SleepRecord,
    stages_pred: np.ndarray,
    stages_mode: str,
    stages_pred_duration: int = 30,
    merge_annotations: bool = False,
    show_bpm: bool = False,
):
    """
    Plot a hypnogram for a single record.

    Annotated sleep stages are included in the plot if available in `record`. If
    `stages_pred` contains probabilities, they are shown in an additional subplot.

    Parameters
    ----------
    record : SleepRecord
        A single record (i.e. night).
    stages_pred : np.ndarray
        The predicted stages, either as a 1d-array of integers or a 2d-array of
        probabilties.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.
    stages_pred_duration : int, optional
        Duration of the predicted sleep stages in seconds, by default `30`.
    merge_annotations : bool, optional
        If `True`, merge annotations according to `stages_mode`, otherwise plot original
        annotations. By default `False`.
    show_bpm : bool, optional
        If `True`, include a subplot of the heart rate in bpm. This can be helpful to find
        bad signal quality intervals, by default `False`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the hypnogram plot.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    stages_pred_probs = None
    num_subplots = 1
    if stages_pred.ndim == 2:
        num_subplots += 1
        stages_pred_probs = stages_pred
        stages_pred = stages_pred_probs.argmax(1)
    if record.sleep_stages is not None:
        num_subplots += 1
    if show_bpm:
        num_subplots += 1

    start_time = _time_to_sec(record.recording_start_time)

    fig, ax = plt.subplots(num_subplots, sharex=True, figsize=(7, 4))

    # predicted stages
    t_stages_pred = np.arange(len(stages_pred)) * stages_pred_duration + start_time
    t_stages_pred = t_stages_pred.astype("datetime64[s]")
    stages_pred = stages_pred.astype(float)
    stages_pred[stages_pred == SleepStage.UNDEFINED] = np.nan
    ax[0].plot(t_stages_pred, stages_pred)
    ax[0].set_yticks(_STAGE_INTS[stages_mode])
    ax[0].set_yticklabels(_STAGE_NAMES[stages_mode])
    ax[0].set_ylabel("predicted")
    ax[0].yaxis.tick_right()

    row = 1

    # predicted stage probabilities
    if stages_pred_probs is not None:
        ax[row].stackplot(
            t_stages_pred,
            stages_pred_probs[:, 1:].T,
            labels=_STAGE_NAMES[stages_mode],
        )
        ax[row].set_ylabel("probabilities")
        legend_handles, legend_labels = ax[row].get_legend_handles_labels()
        ax[row].legend(legend_handles[::-1], legend_labels[::-1], loc=(1.01, 0))
        ax[row].set_ylim(0, 1)
        ax[row].set_yticks([])
        row += 1

    # annotated stages
    if record.sleep_stages is not None:
        stages_true = record.sleep_stages
        t_stages_true = (
            np.arange(len(stages_true)) * record.sleep_stage_duration + start_time
        )
        t_stages_true = t_stages_true.astype("datetime64[s]")
        if merge_annotations:
            stages_true = _merge_sleep_stages([stages_true], stages_mode)[0]
            stages_mode_true = stages_mode
        else:
            stages_mode_true = "wake-rem-n1-n2-n3"
        stages_true = stages_true.astype(float)
        stages_true[stages_true == SleepStage.UNDEFINED] = np.nan

        ax[row].plot(t_stages_true, stages_true)
        ax[row].set_yticks(_STAGE_INTS[stages_mode_true])
        ax[row].set_yticklabels(_STAGE_NAMES[stages_mode_true])
        ax[row].set_ylabel("annotated")
        ax[row].yaxis.tick_right()

        row += 1

    # heartrate
    if show_bpm:
        t_ecg = (record.heartbeat_times[1:] + start_time).astype("datetime64[s]")
        ax[row].plot(t_ecg, 60 / np.diff(record.heartbeat_times))
        ax[row].set_ylabel("beats per minute")
        ax[row].yaxis.tick_right()

    # x axis ticks and label
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    if record.recording_start_time is None:
        ax[-1].set_xlabel("time since recording start in hours")
    else:
        ax[-1].set_xlabel("time of day in hours")
    ax[-1].set_xlim(t_stages_pred[0], t_stages_pred[-1])

    return fig
