import logging
import os
import ssl
import time
import typing as t
from pathlib import Path

import click
import datasets
import lyricsgenius
from datasets import load_dataset, load_dataset_builder
from datasets.config import HF_DATASETS_CACHE
from furl import furl
from lyricsgenius.types.artist import Artist

from huggingartists.dataset.genius import \
    create_dataset as create_dataset_from_genius_lyrics
from huggingartists.utils import (ParameterSource, artist_workspace,
                                  default_param_file, get_params, get_path,
                                  init_loggers, load_mlcube_parameters)

__all__ = ["create_dataset"]
logger = logging.getLogger("create_dataset")


GENIUS_ACCESS_TOKEN = "q_JK_BFy9OMiG7fGTzL-nUto9JDv3iXI24aYRrQnkOvjSCSbY4BuFIindweRsr5I"
HUGGINGFACE_NAMESPACE = "huggingartists"


def create_dataset(
    params: t.Optional[ParameterSource] = None,
    workspace_dir: t.Optional[t.Union[str, Path]] = None,
    cache_dir: t.Optional[t.Union[str, Path]] = None,
) -> None:
    # Check input parameters and their default values.
    params = get_params(
        params,
        defaults={
            "artist_name": "Eminem",
            "genius_access_token": os.environ.get("GENIUS_API_TOKEN", None)
            or GENIUS_ACCESS_TOKEN,
            "huggingface_namespace": HUGGINGFACE_NAMESPACE,
        },
    )
    os.environ.update(params.get("env", {}))
    logger.info(
        "Task inputs: params=%s, workspace_dir=%s, cache_dir=%s",
        params,
        workspace_dir,
        cache_dir,
    )

    artist_workspace_dir = artist_workspace(workspace_dir, params["artist_name"])
    cache_dir = get_path(cache_dir, datasets.config.HF_DATASETS_CACHE)
    logger.info(
        "Artist working directory: %s. HF dataset cache directory: %s",
        artist_workspace_dir,
        cache_dir,
    )

    # If dataset has been cached locally, return it.
    rawdata_dir = artist_workspace_dir / "rawdata"
    if rawdata_dir.exists():
        logger.warning(
            "Output directory exists (%s). Delete this directory to rerun this task.",
            rawdata_dir.as_posix(),
        )
        return

    # Fetch artist brief info from `Genius` (at this point in time we need only to indentify model name).
    try:
        genius = lyricsgenius.Genius(params["genius_access_token"])
        artist: t.Optional[Artist] = genius.search_artist(
            params["artist_name"], max_songs=0, get_full_info=False
        )
    except ssl.SSLCertVerificationError:
        logger.error(
            "Cannot access remote Genius API. If you are behind a corporate firewall, try setting `http_proxy` and "
            "`https_proxy` environment variables."
        )
        raise

    if artist is None:
        logger.error("Artist (%s) has not been found.", params["artist_name"])
        raise Exception(f"Artist ({params['artist_name']}) has not been found!")
    logger.info(
        "Artist (%s) has been found: name=%s, url=%s",
        params["artist_name"],
        artist.name,
        artist.url,
    )

    # Fetch details on artist, and use the last path segment as HF's repository name within `hf_namespace` org. The
    # `genius.artist` returns a dictionary. The `url` is a string.
    time.sleep(0.1)
    repo_name = (
        furl(genius.artist(artist.id)["artist"]["url"]).path.segments[-1].lower()
    )

    # Try to find this dataset on Hugging Face before building it from scratch. The `status_code` maybe 401 which
    # means not authorized (e.g., dataset is private), but this could also mean there's no such dataset at all.
    # Probably safe to check for 200  (OK: Indicates that the related request has been successful).
    repo_id: str = f"{params['huggingface_namespace']}/{repo_name}"
    try:
        # If this function does not raise any exceptions, the dataset has been found.
        _ = load_dataset_builder(repo_id)
        dataset = load_dataset(repo_id, cache_dir=cache_dir)
        dataset.save_to_disk(rawdata_dir)
        logger.info(
            "Artist (%s) dataset has been fetched form HF hub (%s) and has been cached at %s.",
            params["artist_name"],
            repo_id,
            cache_dir,
        )
    except FileNotFoundError:
        # Need to build dataset from scratch using lyrics from genius website.
        create_dataset_from_genius_lyrics(
            artist.id, params["genius_access_token"], rawdata_dir
        )
        logger.info(
            "Artist (%s) dataset has been created using lyrics from Genius service. PS - dataset has not been pushed "
            "to HF hub yet.",
            params["artist_name"],
        )


@click.command()
@click.option("--params", required=False, type=str, default=default_param_file())
@click.option("--workspace_dir", required=False, type=str)
@click.option("--cache_dir", required=False, type=str)
def run_task(
    params: str,
    workspace_dir: t.Optional[str] = None,
    cache_dir: t.Optional[str] = None,
) -> None:
    init_loggers(workspace_dir)
    create_dataset(
        load_mlcube_parameters(params, "create_dataset"), workspace_dir, cache_dir
    )


if __name__ == "__main__":
    run_task()
