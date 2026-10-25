"""Train LightGBM on California Housing and explain a few predictions.

Run: python experiments/01_california_baseline.py experiments/configs/california_baseline.yaml
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import yaml
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from explainkit import explain_prediction

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())

X, y = fetch_california_housing(return_X_y=True, as_frame=True, data_home="data")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"])

model = lgb.LGBMRegressor(random_state=cfg["seed"], verbose=-1, **cfg["model"]).fit(X_tr, y_tr)
mae = mean_absolute_error(y_te, model.predict(X_te))
print(f"Test MAE: {mae:.3f} (units: $100k)")

results = {"mae": mae, "explanations": []}
for i in cfg["explain_rows"]:
    exp = explain_prediction(model, X_te.iloc[[i]], feature_groups=cfg["feature_groups"])
    print(f"\nRow {i}: predicted {exp.prediction:.2f} (average {exp.base_value:.2f})")
    for name, val in exp.top(3):
        print(f"  {name:>12}: {val:+.2f}")
    results["explanations"].append({"row": i, "prediction": exp.prediction, "contributions": exp.contributions})

if cfg["use_wandb"]:
    import wandb
    wandb.init(project="explainkit", config=cfg)
    wandb.log({"mae": mae})
    wandb.finish()

out = Path("results") / f"{Path(sys.argv[1]).stem}.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out}")
