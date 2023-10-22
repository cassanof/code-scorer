import torch
import wandb
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seq_len", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--save_dir", type=str, default="./results")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--score_col", type=str, default="score")
parser.add_argument("--content_col", type=str, default="content")
parser.add_argument("--model", type=str, default="bigcode/starencoder")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--no_fp16", action="store_true")
parser.add_argument("--eval_ratio", type=float, default=0.05)
args = parser.parse_args()


dataset = datasets.load_dataset(args.dataset, split='train')
dataset = dataset.train_test_split(test_size=args.eval_ratio)

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=1).to("cuda")

train_encodings = tokenizer(
    dataset['train'][args.content_col], truncation=True, padding=True, max_length=args.seq_len)
valid_encodings = tokenizer(
    dataset['test'][args.content_col], truncation=True, padding=True, max_length=args.seq_len)


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


train_dataset = RegressionDataset(
    train_encodings, dataset['train'][args.score_col])
valid_dataset = RegressionDataset(
    valid_encodings, dataset['test'][args.score_col])


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


training_args = TrainingArguments(
    output_dir=args.save_dir,
    report_to="wandb",
    logging_steps=10,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    weight_decay=args.weight_decay,
    learning_rate=args.lr,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='rmse',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    bf16=args.bf16,
    fp16=(not args.no_fp16),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics_for_regression,
)

wandb.init(project="roblox")

trainer.train()
trainer.evaluate()

# save model and tokenizer
trainer.save_model(args.save_dir + "/best")
tokenizer.save_pretrained(args.save_dir + "/best")
