import torch
import os
import time
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    logits = logits[0]
    labels = labels.reshape(-1, 1)
    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) /
                                   (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def dtype_from_str(dtype_str):
    """
    Converts the string representation of a dtype to a torch dtype.
    """
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        item["labels"] = float(item["labels"])
        return item

    def __len__(self):
        return len(self.labels)


def is_main(args):
    """
    Returns True if the process is the main process.
    """
    return args.local_rank in [-1, 0]


def init_wandb(args):
    import wandb
    wandb_name = None
    if not os.getenv("WANDB_NAME"):
        date = time.strftime("%Y-%m-%d-%H-%M")
        model_name = args.model.rstrip("/").split("/")[-1]
        dataset_name = args.dataset_name.rstrip("/").split("/")[-1]
        wandb_name = f"{model_name}_{dataset_name}_{date}"
    try:
        wandb.init(name=wandb_name)
    except Exception as e:
        print(
            f"Failed to initialize wandb -- Can disable it with the `--no_wandb` option.\nError: {e}")
        raise e


def model_load_extra_kwargs(args):
    model_extra_kwargs = {}
    if args.fa2:
        # need to set dtype to either float16 or bfloat16
        if args.bf16:
            model_extra_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_extra_kwargs["torch_dtype"] = torch.float16

    if args.torch_dtype:  # overrides everything else
        model_extra_kwargs["torch_dtype"] = dtype_from_str(args.torch_dtype)

    if args.attention_dropout is not None:  # some models dont support this
        model_extra_kwargs["attention_dropout"] = args.attention_dropout

    return model_extra_kwargs


def load_datasets(args, tokenizer):
    dataset = datasets.load_dataset(args.dataset, split='train')
    if not args.no_shuffle_train:
        dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=args.eval_ratio)

    train_encodings = tokenizer(
        dataset['train'][args.content_col], truncation=True, padding=True, max_length=args.seq_len)
    valid_encodings = tokenizer(
        dataset['test'][args.content_col], truncation=True, padding=True, max_length=args.seq_len)

    train_dataset = RegressionDataset(
        train_encodings, dataset['train'][args.score_col])
    valid_dataset = RegressionDataset(
        valid_encodings, dataset['test'][args.score_col])
    return train_dataset, valid_dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, valid_dataset = load_datasets(args, tokenizer)
    has_eval = len(valid_dataset) > 0

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        report_to="wandb" if not args.no_wandb else None,
        logging_steps=10,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        save_total_limit=args.epochs,
        evaluation_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=(not args.no_fp16),
        deepspeed=args.deepspeed,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        use_flash_attention_2=args.fa2,
        use_cache=not args.no_gradient_checkpointing,
        trust_remote_code=True,
        **model_load_extra_kwargs(args)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_eval else None,
        compute_metrics=compute_metrics_for_regression,
    )

    init_wandb(args)

    trainer.train()
    if has_eval:
        trainer.evaluate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1000,
                        help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="Batch size.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay.")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory to save the model checkpoints.")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Dataset name.")
    parser.add_argument("--score_col", type=str,
                        default="score", help="Column name for the score.")
    parser.add_argument("--content_col", type=str,
                        default="content", help="Column name for the content.")
    parser.add_argument("--model", type=str,
                        default="bigcode/starencoder", help="Model name.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16.")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Do not use fp16.")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Ratio of the dataset to use for evaluation.")
    parser.add_argument("--no_shuffle_train", action="store_true",
                        help="Do not shuffle the training set.")
    parser.add_argument("--attention_dropout", type=float, default=None,
                        help="Attention dropout -- may not be supported by all models.")
    parser.add_argument("--num_warmup_steps", type=int,
                        default=10, help="Number of warmup steps.")
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        help="Disable gradient checkpointing.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=[
        "cosine", "linear", "constant"], help="Learning rate scheduler type.")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed configuration file.")
    parser.add_argument("--fa2", action="store_true",
                        help="Use FlashAttention2.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Do not use wandb.")
    args = parser.parse_args()
    main(args)
