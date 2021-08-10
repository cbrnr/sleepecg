# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""I/O- and download-related utilities."""

import hashlib
from pathlib import Path
from typing import Optional

import requests

__all__ = [
    'download_file',
]

_HASH_FUNCTIONS = {
    'md5': hashlib.md5,
    'sha256': hashlib.sha256,
}


def _calculate_checksum(filepath: Path, checksum_type: str) -> str:
    """
    Calculate the checksum for a file.

    Parameters
    ----------
    filepath : pathlib.Path
        Location of the file.
    checksum_type : {'md5', 'sha256'}
        Type of the checksum to calculate.

    Returns
    -------
    str
        The hexdigest of the checksum.
    """
    computed_hash = _HASH_FUNCTIONS[checksum_type]()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            computed_hash.update(chunk)
    return computed_hash.hexdigest()


def download_file(
    url: str,
    target_filepath: Path,
    checksum: Optional[str] = None,
    checksum_type: Optional[str] = None,
    verbose: bool = False,
):
    """
    Download a single file from `url` to `target_filepath`.

    In case `checksum` and `checksum_type` are provided, the downloaded
    file is verified. Raises a `RuntimeError` in case verification fails.

    Parameters
    ----------
    url : str
        URL to download from.
    target_filepath : pathlib.Path
        Location where the downloaded file will be stored.
    checksum : str, optional
        Checksum to verify the file against, by default `None`.
    checksum_type : str, optional
        Type of the checksum, by default `None`.
    verbose : bool, optional
        If `True`, output information during download. By default `False`.
    """
    if target_filepath.is_file():
        if checksum is not None and checksum_type is not None:
            calculated_checksum = _calculate_checksum(target_filepath, checksum_type)
            if calculated_checksum == checksum:
                if verbose:
                    print(f'Skipping {url}, already downloaded.')
                return

    target_filepath.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'Downloading {url}...')

    response = requests.get(url)
    response.raise_for_status()

    with open(target_filepath, 'wb') as file:
        file.write(response.content)

    if checksum is not None and checksum_type is not None:
        calculated_checksum = _calculate_checksum(target_filepath, checksum_type)
        if calculated_checksum != checksum:
            raise RuntimeError(
                f'Checksum mismatch for {target_filepath}:\n'
                f'    {checksum!r} (expected)\n'
                f'    {calculated_checksum!r} (calculated)',
            )
