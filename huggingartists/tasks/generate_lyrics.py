import os
import typing as t
from pathlib import Path
import logging
import re
import click
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import default_cache_path

from huggingartists.utils import ParameterSource, get_params, default_param_file, artist_workspace, init_loggers, \
    load_mlcube_parameters, get_path

__all__ = ["generate_lyrics"]
logger = logging.getLogger("generate_lyrics")


def generate_lyrics(
    params: t.Optional[ParameterSource],
    workspace_dir: t.Optional[t.Union[str, Path]] = None,
    cache_dir: t.Optional[t.Union[str, Path]] = None,
) -> None:
    params = get_params(
        params,
        defaults={
            "artist_name": "Eminem",
            "base_model": "gpt2",
            "use_gpu": True,
        },
    )
    artist_workspace_dir = artist_workspace(workspace_dir, params["artist_name"])
    cache_dir = get_path(cache_dir, default_cache_path)
    logger.info(
        "Artist working directory: %s. HF transformers cache directory: %s. Prompts file: %s",
        artist_workspace_dir,
        cache_dir,
    )

    lyrics_dir = artist_workspace_dir / "lyrics"
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint: t.Optional[str] = params.get('checkpoint', None)
    if checkpoint is None:
        checkpoint_dir = artist_workspace_dir / 'model'
        checkpoints = [
            d for d in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir / d) and d.startswith("checkpoint-")
        ]
        latest_checkpoint_idx = np.argmax([int(_checkpoint[11:]) for _checkpoint in checkpoints])
        checkpoint = checkpoints[latest_checkpoint_idx]

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        params["base_model"], cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(artist_workspace_dir / 'model' / checkpoint)
    if params["use_gpu"]:
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            logger.warning("The `use_gpu` is true but cuda is not available. Falling back to CPU device.")

    lyrics: t.List[t.Dict] = []
    for prompt in params.get('prompts', []):
        # shape = (1, prompt_length)
        encoded_prompt: torch.Tensor = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        # shape = (num_return_sequences, min(actual_length, max_length))
        encoded_sequences: torch.Tensor = model.generate(
            input_ids=encoded_prompt.to(model.device),
            **params.get('generate_config', {})
        )
        sequences: t.List[t.List[str]] = _decode_sequences(tokenizer, encoded_sequences)
        lyrics.append({"input": prompt, "output": sequences})

    with open(lyrics_dir / 'lyrics.yaml', 'wt') as fp:
        yaml.dump(lyrics, fp)


def _decode_sequences(tokenizer: PreTrainedTokenizerBase, encoded_sequences: torch.Tensor) -> t.List[t.List[str]]:
    sequences: t.List[t.List[str]] = []
    max_repeat = 2

    # decode and clean prediction
    for encoded_sequences in encoded_sequences:
        sequence: str = tokenizer.decode(
            encoded_sequences.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        sequence = re.sub(r"\n+", "\n", sequence.strip())
        sequences.append(sequence.split('\n'))
        # Not sure what the following code does (raises index of range error sometimes).
        """
        lines: t.List[str] = sequence.split('\n')
        i = max_repeat
        while i != len(lines):
            remove_count = 0
            for index in range(0, max_repeat):
                if lines[i - index - 1] == lines[i - index]:
                    remove_count += 1
            if remove_count == max_repeat:
                lines.pop(i)
                i -= 1
            else:
                i += 1
        sequences.append('\n'.join(lines))
        """

    return sequences


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
    try:
        generate_lyrics(
            load_mlcube_parameters(params, "generate_lyrics"), workspace_dir, cache_dir
        )
    except Exception as err:
        logger.exception("Exception while executing `generate_song` task")
        print(f"Exception while executing `generate_song` task: {err}")
        raise


if __name__ == "__main__":
    run_task()
