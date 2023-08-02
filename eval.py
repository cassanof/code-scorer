import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import datasets
from torch.utils import data
from inference import score_model_factory
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model", type=str,
                        default="nuprl/code-scorer-edu-v1")
arg_parser.add_argument("--model_type", type=str, default="regression")
arg_parser.add_argument("--dataset", type=str, required=True)
arg_parser.add_argument("--split", type=str, default="train")
arg_parser.add_argument("--batch_size", type=int, default=32)
args = arg_parser.parse_args()

model = score_model_factory(args.model_type, args.model)
eval_ds = datasets.load_dataset(args.dataset, split=args.split)


datasets.set_caching_enabled(False)
eval_dl = data.DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
preds = []
for batch in tqdm(eval_dl):
    batch_preds = model.score(batch["content"])
    preds.extend(batch_preds)

real_scores = [float(x) for x in eval_ds["score"]]
print(f"Predicted {len(preds)} scores")
print(f"Pearson correlation: {pearsonr(preds, real_scores)}")
print(f"Mean squared error: {mean_squared_error(preds, real_scores)}")
print(
    f"Root mean squared error: {mean_squared_error(preds, real_scores, squared=False)}")
print(f"R2 score: {r2_score(preds, real_scores)}")
