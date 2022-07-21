# © SleepECG developers
#
# License: BSD (3-clause)

"""Simple interface for downloading PhysioNet data."""

import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from .utils import _download_file

_PHYSIONET_FILES_URL = "https://physionet.org/files/"
_CHECKSUM_FILENAME = "SHA256SUMS.txt"
_RECORDS_FILENAME = "RECORDS"
_CHECKSUM_TYPE = "sha256"


def _list_physionet(
    data_dir: Path,
    db_slug: str,
    db_version: Optional[str] = "1.0.0",
    pattern="*",
) -> List[str]:
    """
    List record IDs for a PhysioNet database.

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
    checksum = _get_physionet_checksums(data_dir, db_slug, db_version)[_RECORDS_FILENAME]

    if not records_filepath.is_file():
        _download_file(records_url, records_filepath, checksum, _CHECKSUM_TYPE)

    all_records = records_filepath.read_text().splitlines()
    return fnmatch.filter(all_records, pattern)


def download_physionet(
    data_dir: Path,
    db_slug: str,
    requested_records: List[str],
    extensions: Iterable[str],
    db_version: Optional[str] = "1.0.0",
) -> None:
    """
    Download requested files from PhysioNet.

    All files with `extensions` for record IDs in `requested_records` are downloaded from
    the PhysioNet database `db_slug`.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where all datasets are stored.
    db_slug : str
        Short identifier of a database, e.g. `'mitdb'`.
    requested_records : list[str]
        Records with those IDs are downloaded.
    extensions : Iterable[str]
        Files with those extensions are downloaded.
    db_version : str, optional
        Version of the database, by default `'1.0.0'`.
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
    db_version: Optional[str] = "1.0.0",
) -> Dict[str, str]:
    """
    Parse PhysioNet checksums into a dictionary.

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
