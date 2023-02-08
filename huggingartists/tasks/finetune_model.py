import logging
import os
import typing as t
from functools import partial
from pathlib import Path

import click
import torch
from datasets import DatasetDict
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainingArguments,
                          get_cosine_schedule_with_warmup)
from transformers.integrations import is_mlflow_available
from transformers.utils.hub import default_cache_path

from huggingartists.utils import (ParameterSource, artist_workspace,
                                  default_param_file, get_params, get_path,
                                  init_loggers, load_mlcube_parameters)

__all__ = ["finetune_model"]
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
            "base_model": "gpt2",
            "use_gpu": True,
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
    cache_dir = get_path(cache_dir, default_cache_path)
    logger.info(
        "Artist working directory: %s. HF transformers cache directory: %s",
        artist_workspace_dir,
        cache_dir,
    )

    model_dir = artist_workspace_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_dir.as_posix(), **params.get("training_arguments", {})
    )

    if "mlflow" in training_args.report_to and is_mlflow_available():
        if not os.environ.get("MLFLOW_TRACKING_URI", None):
            os.environ["MLFLOW_TRACKING_URI"] = (model_dir / "mlruns").as_uri()
        logger.info(
            "MLflow integration is enabled (tracking_uri=%s).",
            os.environ["MLFLOW_TRACKING_URI"],
        )
    else:
        logger.warning("MLflow integration is not enabled.")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        params["base_model"], cache_dir=cache_dir
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        params["base_model"], cache_dir=cache_dir
    )
    if params["use_gpu"]:
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            logger.warning("The `use_gpu` is true but cuda is not available. Falling back to CPU device.")

    # Preprocess dataset
    dataset: DatasetDict = DatasetDict.load_from_disk(artist_workspace_dir / "dataset")

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

    if params.get("task_specific_params", None):
        task_specific_params: t.Dict = params["task_specific_params"]
        for name, params in task_specific_params.items():
            trainer.model.config.task_specific_params[name] = params

    if torch.has_cuda:
        torch.cuda.empty_cache()

    logger.info("Training a model.")
    _ = trainer.train()

    if params.get("evaluate", False) is True:
        logger.info("Evaluating a model.")
        _ = trainer.evaluate()


def _group_texts(examples, block_size: int) -> t.Dict:
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [v[i: i + block_size] for i in range(0, total_length, block_size)]
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
    try:
        finetune_model(
            load_mlcube_parameters(params, "finetune_model"), workspace_dir, cache_dir
        )
    except Exception as err:
        logger.exception("Exception while executing `finetune_model` task")
        print(f"Exception while executing `finetune_model` task: {err}")
        raise


if __name__ == "__main__":
    run_task()
