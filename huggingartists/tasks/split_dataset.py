import logging
import os
import random
import typing as t
from pathlib import Path

import click
import numpy as np
from datasets import Dataset, DatasetDict

from huggingartists.utils import (ParameterSource, artist_workspace,
                                  default_param_file, get_params, init_loggers,
                                  load_mlcube_parameters)

__all__ = ["split_dataset"]
logger = logging.getLogger("split_dataset")


def split_dataset(
    params: t.Optional[ParameterSource] = None,
    workspace_dir: t.Optional[t.Union[str, Path]] = None,
) -> None:
    params = get_params(
        params,
        defaults={"artist_name": "Eminem", "random_seed": 100, "train_size": 0.85},
    )
    os.environ.update(params.get("env", {}))
    logger.info("Task inputs: params=%s, workspace_dir=%s", params, workspace_dir)

    artist_workspace_dir = artist_workspace(workspace_dir, params["artist_name"])
    logger.info("Artist working directory: %s.", artist_workspace_dir)

    dataset_dir = artist_workspace_dir / "dataset"
    if dataset_dir.exists():
        logger.warning(
            "Output directory exists (%s). Delete this directory to rerun this task.",
            dataset_dir.as_posix(),
        )
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # This dataset should contain one `train` split. That's a table with one column (text) containing one song as one
    # example
    rawdata_dir = artist_workspace_dir / "rawdata"
    rawdata: DatasetDict = DatasetDict.load_from_disk(rawdata_dir)

    # TODO: fix this
    train_size: float = params["train_size"]
    valid_size = 1.0 - train_size
    test_size = 0.0

    songs: t.List[str] = rawdata["train"]["text"]
    random.seed(params["random_seed"])
    random.shuffle(songs)
    train, validation, test = np.split(
        songs,
        [int(len(songs) * train_size), int(len(songs) * (train_size + valid_size))],
    )

    datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": list(train)}),
            "validation": Dataset.from_dict({"text": list(validation)}),
            "test": Dataset.from_dict({"text": list(test)}),
        }
    )
    datasets.save_to_disk(dataset_dir)


@click.command()
@click.option("--params", required=False, type=str, default=default_param_file())
@click.option("--workspace_dir", required=False, type=str)
def run_task(params: str, workspace_dir: t.Optional[str] = None) -> None:
    init_loggers(workspace_dir)
    split_dataset(load_mlcube_parameters(params, "split_dataset"), workspace_dir)


if __name__ == "__main__":
    run_task()
