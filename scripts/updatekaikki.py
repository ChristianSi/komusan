#!/usr/bin/env python3
"""Update the kaikki.org dump if needed.

Requires requests.
"""

from datetime import datetime
from email import utils
import gzip
from os import path
import sys
from typing import Tuple

import requests

import util


def get_last_modified_date_and_size(url: str) -> Tuple[datetime, int]:
    """Make a HEAD request and get the last-modified date and file size of an URL."""
    # We send an empty Accept-Encoding header since otherwise the server sends the size
    # of a compressed version of the file
    response = requests.head(url, headers={'Accept-Encoding': ''})
    response.raise_for_status()
    last_modified_raw = response.headers.get('Last-Modified')
    content_length_raw = response.headers.get('Content-Length')
    return (datetime(*utils.parsedate(last_modified_raw)[:6]),  # type: ignore
            int(content_length_raw))                            # type: ignore


def download_and_compress_file(url: str, output_filename: str, file_size: int) -> None:
    """Download a file and store it locally, compressed using gzip.

    A progress indicator is printed to stdout while the download is in progress, based on the
    'file_size' argument.
    """
    chunk_size = 1024**2  # 1 MB
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Open the output file in gzip mode
    with gzip.open(output_filename, 'wb') as outfile:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            outfile.flush()
            downloaded_size += len(chunk)

            # Calculate and print progress
            progress = int(60 * downloaded_size / file_size)
            percent = int(100 * downloaded_size / file_size)
            sys.stdout.write(f"\r[{'=' * progress}{' ' * (60 - progress)}] {percent}%")
            sys.stdout.flush()

    print(f'\nFile downloaded and compressed as {output_filename}.')


def update_kaikki_if_needed() -> None:
    """Check if the kaikki.org dump has changed and download the latest version if needed."""
    url = f'{util.KAIKKI_EN_DIR}/{util.KAIKKI_EN_FILE}'
    datefile = f'{util.KAIKKI_EN_FILE}.date'
    download = False

    if path.exists(datefile):
        old_moddate = datetime.fromisoformat(util.read_file(datefile).strip())
        new_moddate, file_size = get_last_modified_date_and_size(url)

        if old_moddate < new_moddate:
            print(f'Redownloading {url} since it has changed...')
            download = True
        else:
            print(f"{url} hasn't changed")
    else:
        new_moddate, file_size = get_last_modified_date_and_size(url)
        print(f"Downloading {url} since it doesn't yet exist locally...")
        download = True

    if download:
        # Download file
        download_and_compress_file(url, f'{util.KAIKKI_EN_FILE}.gz', file_size)
        # Store last-modified date
        util.dump_file(str(new_moddate) + "\n", datefile)


if __name__ == '__main__':
    update_kaikki_if_needed()
