import typing as t
import random
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict


def get_dataset(params: t.Optional[t.Dict], rawdata_dir: str) -> None:
    params = params or {}
    params['artist_name'] = params.get('artist_name', None) or 'Eminem'
    params['train_size'] = params.get('train_size', 0.85)
    params['workspace'] = params.get('workspace', (Path.cwd().resolve() / 'workspace').as_posix())

    dataset_path = Path(params['workspace']) / params['artist_name'].lower().replace(" ", "_") / 'dataset'
    if dataset_path.exists():
        return
    dataset_path.mkdir(parents=True, exist_ok=True)

    rawdata: DatasetDict = DatasetDict.load_from_disk(rawdata_dir)

    train_size: float = params['train_size']
    valid_size = 1.0 - train_size
    test_size = 0.0

    songs: t.List[str] = rawdata['train']['text']
    random.shuffle(songs)
    train, validation, test = np.split(
        songs,
        [
            int(len(songs) * train_size),
            int(len(songs) * (train_size + valid_size))
        ]
    )

    datasets = DatasetDict({
        'train': Dataset.from_dict({'text': list(train)}),
        'validation': Dataset.from_dict({'text': list(validation)}),
        'test': Dataset.from_dict({'text': list(test)})
    })
    datasets.save_to_disk(dataset_path)


def run_task() -> None:
    get_dataset(
        params=None,
        rawdata_dir=(Path.cwd().resolve() / 'workspace' / 'eminem' / 'rawdata').as_posix()
    )


if __name__ == '__main__':
    run_task()
