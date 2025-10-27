import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from src.utils.logger import setup_logging

logger = setup_logging()

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    c = np.corrcoef(y_true, y_pred)
    if np.isnan(c).any():
        return 0.0
    return float(c[0,1])

def ensure_splits(metadata_csv="data/processed/refined_dataset_metadata.csv",
                  splits_dir="data/processed/splits",
                  seed=SEED,
                  ratios=(0.8, 0.1, 0.1)):
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_file = splits_dir / "train_ids.txt"
    val_file = splits_dir / "val_ids.txt"
    test_file = splits_dir / "test_ids.txt"

    if train_file.exists() and val_file.exists() and test_file.exists():
        logger.info("Splits already available")
        return str(train_file), str(val_file), str(test_file)

    logger.info("No split files, creating new one")
    df_meta = pd.read_csv(metadata_csv)
    ids = list(df_meta["complex_id"].astype(str).tolist())

    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    train_file.write_text("\n".join(train_ids))
    val_file.write_text("\n".join(val_ids))
    test_file.write_text("\n".join(test_ids))

    logger.info(f"new splits: train->{len(train_ids)}, val->{len(val_ids)}, test->{len(test_ids)}")
    return str(train_file), str(val_file), str(test_file)

def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            out = out.numpy().reshape(-1)
            y = batch.y.numpy().reshape(-1)
            preds.append(out)
            ys.append(y)
    if len(preds) == 0:
        return {}
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    metrics = {
        "rmse": rmse(ys, preds),
        "r2": float(r2_score(ys, preds)) if len(ys) > 1 else 0.0,
        "pearson": pearson_r(ys, preds)
    }
    return metrics