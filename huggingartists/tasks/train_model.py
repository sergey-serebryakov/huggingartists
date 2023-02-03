import yaml
import torch
import typing as t
from pathlib import Path
from functools import partial
import random
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    get_cosine_schedule_with_warmup


def train_model(params: t.Optional[t.Dict], dataset_dir: str) -> None:
    params = params or {}
    params['artist_name'] = params.get('artist_name', None) or 'Eminem'
    params['base_model'] = params.get('base_model', 'gpt2')
    params['train_arg.num_epochs'] = params.get('num_epochs', 1)
    params['train_arg.learning_rate'] = params.get('learning_rate', 1.372e-4)
    params['train_arg.weight_decay'] = params.get('weight_decay', 0.01)
    params['train_arg.seed'] = params.get('weight_decay', random.randint(0, 2**32-1))
    params['workspace'] = params.get('workspace', (Path.cwd().resolve() / 'workspace').as_posix())

    tokenizer = AutoTokenizer.from_pretrained(params['base_model'])
    model = AutoModelForCausalLM.from_pretrained(params['base_model'])
    dataset = DatasetDict.load_from_disk(dataset_dir)

    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples["text"]), batched=True, num_proc=1, remove_columns=["text"]
    )

    lm_datasets = tokenized_datasets.map(
        partial(_group_texts, block_size=int(tokenizer.model_max_length / 4)),
        batched=True, batch_size=1000, num_proc=1,
    )

    model_dir = Path(params['workspace']) / params['artist_name'].lower().replace(" ", "_") / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_dir.as_posix(),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=params['train_arg.learning_rate'],
        weight_decay=params['train_arg.weight_decay'],
        num_train_epochs=params['train_arg.num_epochs'],
        save_total_limit=10,
        save_strategy='epoch',
        save_steps=1,
        report_to=None,
        seed=params['train_arg.seed'],
        logging_steps=5,
        do_eval=True,
        eval_steps=1,
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=lm_datasets["train"], eval_dataset=lm_datasets["validation"]
    )

    num_train_steps = len(trainer.get_train_dataloader())
    trainer.create_optimizer_and_scheduler(num_train_steps)
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    trainer.model.config.task_specific_params['text-generation'] = {
        'do_sample': True, 'min_length': 100, 'max_length': 200, 'temperature': 1., 'top_p': 0.95,
    }

    if torch.has_cuda:
        torch.cuda.empty_cache()
    data = trainer.train()
    print(type(data))

    evaluation: t.Dict = trainer.evaluate()
    with open(model_dir / 'evaluation.txt', 'w') as fp:
        yaml.dump(evaluation, fp)


def _group_texts(examples, block_size):
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


def run_task() -> None:
    train_model(
        params={},
        dataset_dir=(Path.cwd().resolve() / 'workspace' / 'eminem' / 'dataset').as_posix()
    )


if __name__ == '__main__':
    run_task()
