import logging
import typing as t
from functools import partial
from pathlib import Path

import click
import torch
import yaml
from datasets import DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainingArguments,
                          get_cosine_schedule_with_warmup)
from transformers.utils.hub import default_cache_path

from huggingartists.utils import (ParameterSource, artist_workspace,
                                  default_param_file, get_params, get_path,
                                  init_loggers, load_mlcube_parameters)

__all__ = ['finetune_model']
logger = logging.getLogger("finetune_model")


def finetune_model(
    params: t.Optional[ParameterSource],
    workspace_dir: t.Optional[t.Union[str, Path]] = None,
    cache_dir: t.Optional[t.Union[str, Path]] = None,
) -> None:
    params = get_params(
        params,
        defaults={
            "artist_name": "Eminem",
            "random_seed": 100,
            "base_model": "gpt2",
            "train_arg.num_epochs": 1,
            "train_arg.learning_rate": 1.372e-4,
            "train_arg.weight_decay": 0.01,
            "train_env.use_gpu": True,
        },
    )
    logger.info(
        "Task inputs: params=%s, workspace_dir=%s, cache_dir=%s",
        params,
        workspace_dir,
        cache_dir,
    )

    artist_workspace_dir = artist_workspace(workspace_dir, params["artist_name"])
    cache_dir = get_path(cache_dir, default_cache_path)
    logger.info(
        "Artist working directory: %s. HF hub cache directory: %s",
        artist_workspace_dir,
        cache_dir,
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        params["base_model"], cache_dir=cache_dir
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        params["base_model"], cache_dir=cache_dir
    )
    dataset: DatasetDict = DatasetDict.load_from_disk(artist_workspace_dir / "dataset")

    if params["train_env.use_gpu"]:
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            logger.warning(
                "The `train_env.use_gpu` is true but cuda is not available. Falling back to CPU device."
            )

    # Tokenized dataset will contain two features - `input_ids` and `attention_mask`.
    dataset: DatasetDict = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        desc="Tokenizing datasets.",
    )

    dataset: DatasetDict = dataset.map(
        partial(_group_texts, block_size=int(tokenizer.model_max_length / 4)),
        batched=True,
        batch_size=1000,
        num_proc=1,
    )

    model_dir = artist_workspace_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_dir.as_posix(),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=params["train_arg.learning_rate"],
        weight_decay=params["train_arg.weight_decay"],
        num_train_epochs=params["train_arg.num_epochs"],
        save_total_limit=10,
        save_strategy="epoch",
        save_steps=1,
        report_to=None,
        seed=params["random_seed"],
        logging_steps=5,
        do_eval=True,
        eval_steps=1,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    num_train_steps = len(trainer.get_train_dataloader())
    trainer.create_optimizer_and_scheduler(num_train_steps)
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    trainer.model.config.task_specific_params["text-generation"] = {
        "do_sample": True,
        "min_length": 100,
        "max_length": 200,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    if torch.has_cuda:
        torch.cuda.empty_cache()
    _ = trainer.train()

    evaluation: t.Dict = trainer.evaluate()
    with open(model_dir / "evaluation.txt", "wt") as fp:
        yaml.dump(evaluation, fp)


def _group_texts(examples, block_size: int) -> t.Dict:
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, v in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


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
    finetune_model(
        load_mlcube_parameters(params, "finetune_model"), workspace_dir, cache_dir
    )


if __name__ == "__main__":
    run_task()
