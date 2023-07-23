import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--push", type=str, required=True)
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

model.push_to_hub(args.push)
tokenizer.push_to_hub(args.push)
