# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Interface for listing and downloading NSRR (sleepdata.org) data."""

from fnmatch import fnmatch
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List, Tuple, Union

import requests
from tqdm import tqdm

from .utils import download_file

__all__ = [
    'set_nsrr_token',
    'get_nsrr_download_url',
    'list_nsrr_files',
    'download_nsrr_files',
]

_nsrr_token = None


def set_nsrr_token(token: str) -> None:
    """
    Set and verify the NSRR (sleepdata.org) download token.

    Implemented according to the NSRR API documentation:
    https://github.com/nsrr/sleepdata.org/wiki/api-v1-account

    Parameters
    ----------
    token : str
        NSRR (sleepdata.org) download token (get it from
        https://sleepdata.org/token).

    """
    response = requests.get(
        'https://sleepdata.org/api/v1/account/profile.json',
        params={'auth_token': token},
    )
    authenticated = response.json()['authenticated']
    if authenticated:
        username = response.json()['username']
        email = response.json()['email']
        print(f'Authenticated at sleepdata.org as {username} ({email})')
        global _nsrr_token
        _nsrr_token = token
    else:
        raise RuntimeError('Authentication at sleepdata.org failed, verify token!')


def get_nsrr_download_url(db_slug: str) -> str:
    """
    Get the download URL for a given NSRR database.

    The download token is a part of the URL, so it needs to be already set.

    Parameters
    ----------
    db_slug : str
        Short identifier of a database, e.g. `'mesa'`.

    Returns
    -------
    download_url : str
        The download URL.
    """
    if _nsrr_token is None:
        raise RuntimeError('NSRR token not set, use `sleepecg.io.set_nsrr_token(<token>)`!')
    return f'https://sleepdata.org/datasets/{db_slug}/files/a/{_nsrr_token}/m/sleepecg/'


def list_nsrr_files(
    db_slug: str,
    subfolder: str = '',
    pattern: str = '*',
    shallow: bool = False,
) -> List[Tuple[str, str]]:
    """
    Recursively list filenames and checksums for a dataset.

    Specify a subfolder and/or a filename-pattern to filter results.

    Implemented according to the NSRR API documentation:
    https://github.com/nsrr/sleepdata.org/wiki/api-v1-datasets#list-files-in-folder

    Parameters
    ----------
    db_slug : str
        Short identifier of a database, e.g. `'mesa'`.
    subfolder : str, optional
        The folder at which to start the search, by default `''` (i.e. the
        root folder).
    pattern : str, optional
        Glob-like pattern to select files (only applied to the basename,
        not the dirname), by default `'*'`.
    shallow : bool, optional
        If `True`, only search in the given subfolder (i.e. no recursion),
        by default `False`.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples `(<filename>, <checksum>)`; `<filename>` is the
        full filename (i.e. dirname and basename) and `<checksum>` the
        MD5 checksum.
    """
    api_url = f'https://sleepdata.org/api/v1/datasets/{db_slug}/files.json'

    response = requests.get(api_url, params={'path': subfolder})
    try:
        response_json = response.json()
    except JSONDecodeError:
        raise RuntimeError(f'No API response for dataset {db_slug}.') from None

    files = []
    for item in response_json:
        if not item['is_file'] and not shallow:
            files.extend(list_nsrr_files(db_slug, item['full_path'], pattern))
        elif fnmatch(item['file_name'], pattern):
            files.append((item['full_path'], item['file_checksum_md5']))
    return files


def download_nsrr_files(
    db_slug: str,
    subfolder: str = '',
    pattern: str = '*',
    shallow: bool = False,
    data_dir: Union[str, Path] = '.',
) -> None:
    """
    Recursively download files from NSRR (sleepdata.org).

    Specify a subfolder and/or a filename-pattern to filter results.

    Implemented according to the NSRR API documentation:
    https://github.com/nsrr/sleepdata.org/wiki/api-v1-datasets#download-a-file

    Parameters
    ----------
    db_slug : str
        Short identifier of a database, e.g. `'mesa'`.
    subfolder : str, optional
        The folder at which to start the search, by default `''` (i.e. the
        root folder).
    pattern : str, optional
        Glob-like pattern to select files (only applied to the basename,
        not the dirname), by default `'*'`.
    shallow : bool, optional
        If `True`, only download files in the given subfolder (i.e. no
        recursion), by default `False`.
    data_dir : str | pathlib.Path, optional
        Directory where all datasets are stored, by default `'.'`.
    """
    db_dir = Path(data_dir) / db_slug

    download_url = get_nsrr_download_url(db_slug)
    files_to_download = list_nsrr_files(db_slug, subfolder, pattern, shallow)
    tqdm_description = f'Downloading {db_slug}/{subfolder or "."}/{pattern}'

    for filepath, checksum in tqdm(files_to_download, desc=tqdm_description):
        target_filepath = db_dir / filepath
        url = download_url + filepath
        try:
            download_file(url, target_filepath, checksum, 'md5')
        except RuntimeError as error:
            # If the token is invalid for the requested dataset, the
            # request is redirected to a files-overview page. The response
            # is an HTML-page which doesn't have a "content-disposition"
            # header.
            response = requests.get(url, stream=True)
            if 'content-disposition' not in response.headers:
                raise RuntimeError(f'Make sure you have access to {db_slug}!') from error
            else:
                raise
