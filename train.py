import torch
import threading
import os
import time
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import IterableDataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self.tokenizer.save_pretrained(checkpoint_folder)


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    # if logits is a tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.squeeze()
    labels = labels.reshape(-1)

    mse = mean_squared_error(labels, logits)
    rmse = root_mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) /
                                   (np.abs(labels) + np.abs(logits)) * 100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def compute_metrics_for_classification(eval_pred):
    # f-1 score, precision, recall, accuracy
    logits, labels = eval_pred
    # if logits is a tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.squeeze()
    labels = labels.reshape(-1)

    # convert logits to predictions
    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


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


class IterableClassificationDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, content_col, label_col, max_length=4096):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.content_col = content_col
        self.label_col = label_col
        self.max_length = max_length

    def __iter__(self):
        for ex in self.dataset:
            inputs = self.tokenizer(
                ex[self.content_col], truncation=True, padding=False, max_length=self.max_length)
            inputs['labels'] = ex[self.label_col]
            yield inputs

    def __len__(self):
        return len(self.dataset)


def is_main(args):
    """
    Returns True if the process is the main process.
    """
    return args.local_rank in [-1, 0]


def get_rank(args):
    """
    Returns the rank of the process.
    """
    return args.local_rank if args.local_rank != -1 else 0


def init_wandb(args):
    if args.no_wandb:
        return
    import wandb
    wandb_name = None
    if not os.getenv("WANDB_NAME"):
        date = time.strftime("%Y-%m-%d-%H-%M")
        model_name = args.model.rstrip("/").split("/")[-1]
        dataset_name = args.dataset.rstrip("/").split("/")[-1]
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

    if args.num_labels > 1:
        model_extra_kwargs["id2label"] = {
            i: str(i) for i in range(args.num_labels)}
        model_extra_kwargs["label2id"] = {
            str(i): i for i in range(args.num_labels)}

    return model_extra_kwargs


def load_datasets(args, tokenizer):
    dataset = datasets.load_dataset(args.dataset, split=args.train_split)
    if not args.no_shuffle_train:
        dataset = dataset.shuffle(seed=42)

    if args.eval_dataset:
        train = dataset
        val = datasets.load_dataset(args.eval_dataset, split=args.eval_split)
    else:
        dataset = dataset.train_test_split(test_size=args.eval_ratio)
        train = dataset['train']
        val = dataset['test']

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=args.seq_len)
    train_dataset = IterableClassificationDataset(
        tokenizer, train, args.content_col, args.score_col, args.seq_len)
    valid_dataset = IterableClassificationDataset(
        tokenizer, val, args.content_col, args.score_col, args.seq_len)

    return train_dataset, valid_dataset, data_collator


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch._dynamo.config.optimize_ddp = False

    if tokenizer.pad_token is None:  # default to eos token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Process loaded with rank:", get_rank(args))

    train_dataset, valid_dataset, collator = load_datasets(args, tokenizer)
    has_eval = len(valid_dataset) > 0

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        report_to=["wandb"] if not args.no_wandb else [],
        logging_steps=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.no_gradient_checkpointing,
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
        ddp_find_unused_parameters=False,
        torch_compile_backend="inductor" if args.compile else None,
        push_to_hub=args.push != None,
        push_to_hub_model_id=args.push,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        use_flash_attention_2=args.fa2,
        use_cache=not args.no_gradient_checkpointing,
        trust_remote_code=True,
        **model_load_extra_kwargs(args)
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not args.deepspeed:
        model = model.to(torch.device("cuda")
                         if torch.cuda.is_available() else torch.device("cpu"))

    if args.linear_probe:
        freeze_model(model)

    if is_main(args):
        init_wandb(args)

    if args.num_labels > 1:
        compute_metrics = compute_metrics_for_classification
    else:
        compute_metrics = compute_metrics_for_regression

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_eval else None,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[SaveTokenizerCallback(tokenizer)]
    )

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
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the model checkpoints.")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Dataset name.")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Evaluation dataset name.")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name for --dataset.")
    parser.add_argument("--eval_split", type=str, default="test",
                        help="Test split name for --eval_dataset.")
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
    parser.add_argument("--linear-probe", action="store_true",
                        help="Trains only the linear layer. Freezes the rest of the model.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed configuration file.")
    parser.add_argument("--compile", action="store_true",
                        help="Compiles the model with PyTorch.")
    parser.add_argument("--torch_dtype", type=str, default=None, choices=[
        "float16", "bfloat16", "float32"], help="Force the model to use a certain dtype.")
    parser.add_argument("--fa2", action="store_true",
                        help="Use FlashAttention2.")
    parser.add_argument("--push", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Do not use wandb.")
    args = parser.parse_args()
    main(args)
