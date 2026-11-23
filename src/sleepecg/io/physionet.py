# © SleepECG developers
#
# License: BSD (3-clause)

"""Simple interface for downloading PhysioNet data."""

from __future__ import annotations

import fnmatch
import warnings
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

from sleepecg.config import get_config_value
from sleepecg.io.utils import _download_file

_PHYSIONET_FILES_URL = "https://physionet.org/files"
_CHECKSUM_FILENAME = "SHA256SUMS.txt"
_RECORDS_FILENAME = "RECORDS"
_CHECKSUM_TYPE = "sha256"


def _list_physionet(
    data_dir: Path,
    db_slug: str,
    db_version: str = "1.0.0",
    pattern: str = "*",
) -> list[str]:
    """List record IDs for a PhysioNet database.

    IDs can be filtered using a glob-like `pattern`.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where all datasets are stored. Required to download the RECORDS file.
    db_slug : str
        Short identifier of a database, e.g. `'mitdb'`.
    db_version : str, optional
        Version of the database, by default `'1.0.0'`.
    pattern : str, optional
        Glob-like pattern to select record IDs, by default `'*'`.

    Returns
    -------
    list[str]
        List containing record IDs as strings.
    """
    data_dir = Path(data_dir)

    records_filepath = data_dir / db_slug / _RECORDS_FILENAME
    records_url = f"{_PHYSIONET_FILES_URL}/{db_slug}/{db_version}/{_RECORDS_FILENAME}"
    checksum = _get_physionet_checksums(data_dir, db_slug, db_version)[
        _RECORDS_FILENAME
    ]

    if not records_filepath.is_file():
        _download_file(records_url, records_filepath, checksum, _CHECKSUM_TYPE)

    all_records = records_filepath.read_text().splitlines()
    return fnmatch.filter(all_records, pattern)


def _list_capslpdb_records(data_dir: Path, pattern: str) -> list[str]:
    """Return CAP IDs with an ECG and warn at the reader or listing call site."""
    files = _list_physionet(
        data_dir=data_dir, db_slug="capslpdb", pattern=f"{pattern}.edf"
    )
    records = [Path(name).stem for name in files]
    if "n16" in records:
        warnings.warn(
            "Skipping n16 because its EDF does not contain an ECG channel.",
            RuntimeWarning,
            stacklevel=3,
        )
        records.remove("n16")
    return records


def list_physionet_records(
    db_slug: str,
    pattern: str = "*",
    data_dir: str | Path | None = None,
) -> list[str]:
    """List IDs accepted by the corresponding SleepECG PhysioNet reader.

    On first use, fetch the index and checksums for database version 1.0.0.

    Parameters
    ----------
    db_slug : {'capslpdb', 'ltdb', 'mitdb', 'slpdb'}
        Database identifier.
    pattern : str, optional
        Glob matched against record IDs, by default `'*'`.
    data_dir : str | pathlib.Path | None, optional
        Directory used to cache the index and checksums. Defaults to the configured
        `data_dir`.

    Returns
    -------
    list[str]
        IDs to pass as `records_pattern`. CAP IDs omit the EDF extension; record n16
        is skipped with a warning because it has no ECG channel.
    """
    if db_slug not in {"capslpdb", "ltdb", "mitdb", "slpdb"}:
        raise ValueError(f"No SleepECG PhysioNet reader for {db_slug!r}.")
    if data_dir is None:
        data_dir = get_config_value("data_dir")
    data_dir = Path(data_dir).expanduser()
    if db_slug == "capslpdb":
        return _list_capslpdb_records(data_dir, pattern)
    return _list_physionet(data_dir, db_slug, pattern=pattern)


def download_physionet(
    db_slug: str,
    requested_records: list[str],
    extensions: Iterable[str],
    db_version: str = "1.0.0",
    data_dir: str | Path = ".",
) -> None:
    """Download requested files from PhysioNet.

    All files with `extensions` for record IDs in `requested_records` are downloaded
    from the PhysioNet database `db_slug`.

    Parameters
    ----------
    db_slug : str
        Short identifier of a database, e.g. `'mitdb'`.
    requested_records : list[str]
        Records with those IDs are downloaded.
    extensions : Iterable[str]
        Files with those extensions are downloaded.
    db_version : str, optional
        Version of the database, by default `'1.0.0'`.
    data_dir : str | pathlib.Path, optional
        Directory where all datasets are stored, by default `'.'`.
    """
    data_dir = Path(data_dir)
    checksums = _get_physionet_checksums(data_dir, db_slug, db_version)
    db_url = f"{_PHYSIONET_FILES_URL}/{db_slug}/{db_version}"

    for record_id in tqdm(requested_records, desc=f"Downloading {db_slug}"):
        for extension in extensions:
            if not extension.startswith("."):
                extension = "." + extension
            filepath = (data_dir / db_slug / record_id).with_suffix(extension)
            _download_file(
                f"{db_url}/{filepath.name}",
                filepath,
                checksums[filepath.name],
                checksum_type=_CHECKSUM_TYPE,
            )


def _get_physionet_checksums(
    data_dir: Path,
    db_slug: str,
    db_version: str = "1.0.0",
) -> dict[str, str]:
    """Parse PhysioNet checksums into a dictionary.

    Reads a PhysioNet checksum file and parses it into a dictionary mapping filenames to
    checksums. Tries to download the checksum file if it is not available on disk.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where all datasets are stored.
    db_slug : str
        Short identifier of a database, e.g. `'mitdb'`.
    db_version : str, optional
        Version of the database, by default `'1.0.0'`.

    Returns
    -------
    dict[str, str]
        Mapping of filenames to checksums.
    """
    checksum_url = f"{_PHYSIONET_FILES_URL}/{db_slug}/{db_version}/{_CHECKSUM_FILENAME}"
    checksum_filepath = data_dir / db_slug / _CHECKSUM_FILENAME

    if not checksum_filepath.is_file():
        _download_file(checksum_url, checksum_filepath)

    checksums = {}
    for line in checksum_filepath.read_text().splitlines():
        checksum, filename = line.split()
        checksums[filename] = checksum
    return checksums
