import os
import ssl
import typing as t
import lyricsgenius
import requests
import time
from pathlib import Path

from datasets import load_dataset
from lyricsgenius.types.artist import Artist

from huggingartists.datasets.genius import create_dataset as create_dataset_from_genius_lyrics

GENIUS_API_TOKEN = "q_JK_BFy9OMiG7fGTzL-nUto9JDv3iXI24aYRrQnkOvjSCSbY4BuFIindweRsr5I"
HUGGINGFACE_USER = 'huggingartists-app'
HUGGINGFACE_NAMESPACE = 'huggingartists'


def get_rawdata(params: t.Optional[t.Dict] = None) -> None:
    # Check input parameters and their default values.
    params = params or {}
    params['artist_name'] = params.get('artist_name', None) or 'Eminem'
    params['genius_token'] = params.get('genius_token', os.environ.get('GENIUS_API_TOKEN', None)) or GENIUS_API_TOKEN
    params['hf_namespace'] = params.get('hf_namespace', None) or HUGGINGFACE_NAMESPACE
    params['hf_datasets_cache'] = params.get('hf_datasets_cache', None)
    params['workspace'] = params.get('workspace', (Path.cwd().resolve() / 'workspace').as_posix())

    # If dataset has been cached locally, return it.
    rawdata_path = Path(params['workspace']) / params['artist_name'].lower().replace(" ", "_") / 'rawdata'
    if rawdata_path.exists():
        return

    # Fetch artist brief info from `Genius` (at this point in time we need only to indentify model name).
    try:
        genius = lyricsgenius.Genius(params['genius_token'])
        artist: t.Optional[Artist] = genius.search_artist(params['artist_name'], max_songs=0, get_full_info=False)
    except ssl.SSLCertVerificationError:
        print("If you are behind a firewall, try setting http_proxy and https_proxy environmental variables.")
        raise

    if artist is None:
        raise Exception("Artist does not exist!")

    time.sleep(0.1)
    artist_url = str(genius.artist(artist.id)['artist']['url'])
    model_name = artist_url[artist_url.rfind('/') + 1:].lower()

    # Try to find this dataset on Hugging Face before building it from scratch. The `status_code` maybe 401 which
    # means not authorized (e.g., dataset is private), but this could also mean there's no such dataset at all.
    # Probably safe to check for 200  (OK: Indicates that the related request has been successful).
    namespace = params['hf_namespace']
    hf_response = requests.get(f"https://huggingface.co/datasets/{namespace}/{model_name}/tree/main")
    if hf_response.status_code == 200:
        dataset = load_dataset(f"{namespace}/{model_name}", cache_dir=params['hf_datasets_cache'])
        dataset.save_to_disk(rawdata_path)
        return

    # Need to build dataset from scratch
    create_dataset_from_genius_lyrics(artist.id, params['genius_token'], rawdata_path)


def run_task() -> None:
    get_rawdata()


if __name__ == '__main__':
    run_task()
