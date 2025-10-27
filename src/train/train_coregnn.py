import math
import time
import json
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

from src.models.core_gnn import CoreGNN
from src.data.pyg_dataset import PDBDataset
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

def train_one_epoch(model, loader, optim):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    
    for batch in loader:
        optim.zero_grad()
        out = model(batch)
        target = batch.y.view(-1).to(out.dtype)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / (total_graphs + 1e-12)

def train(
    metadata_csv="data/processed/refined_dataset_metadata.csv",
    complex_dir="data/processed/complex_graphs",
    split_dir="data/processed/splits",
    output_dir="experiments/runs/coregnn",
    epochs=50,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-5,
    hidden_dim=128,
    encoder_layers=3,
    seed=SEED,
    patience=10
):
    set_seed(seed)
    writer = SummaryWriter(log_dir=output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ensure_splits(metadata_csv=metadata_csv, splits_dir=split_dir, seed=seed)

    train_ds = PDBDataset(metadata_csv=metadata_csv,
                                     complex_dir=complex_dir,
                                     split_dir=split_dir,
                                     root="data/processed/complex_dataset",
                                     split="train",
                                     force_rebuild=False)
    val_ds = PDBDataset(metadata_csv=metadata_csv,
                                   complex_dir=complex_dir,
                                   split_dir=split_dir,
                                   root="data/processed/complex_dataset",
                                   split="val",
                                   force_rebuild=False)
    test_ds = PDBDataset(metadata_csv=metadata_csv,
                                    complex_dir=complex_dir,
                                    split_dir=split_dir,
                                    root="data/processed/complex_dataset",
                                    split="test",
                                    force_rebuild=False)

    logger.info(f"Datasets: train-> {len(train_ds)}, val-> {len(val_ds)}, test-> {len(test_ds)}")

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # infer dims from a sample
    sample = train_ds.get(0)
    in_dim = sample.x.shape[1]
    # infer edge_dim if present
    edge_dim = sample.edge_attr.shape[1] if hasattr(sample, "edge_attr") and sample.edge_attr is not None else 0

    model = CoreGNN(in_dim, edge_dim, hidden_dim=hidden_dim, num_layers=encoder_layers)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3, verbose=True)

    best_val = float("inf")
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optim)
        val_metrics = evaluate(model, val_loader)
        test_metrics = evaluate(model, test_loader)

        logger.info(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_rmse {val_metrics.get('rmse',float('nan')):.4f} | test_rmse {test_metrics.get('rmse',float('nan')):.4f} | time {time.time()-t0:.1f}s")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("rmse/val", val_metrics.get("rmse", np.nan), epoch)
        writer.add_scalar("rmse/test", test_metrics.get("rmse", np.nan), epoch)
        writer.add_scalar("r2/val", val_metrics.get("r2", np.nan), epoch)
        writer.add_scalar("pearson/val", val_metrics.get("pearson", np.nan), epoch)
        scheduler.step(val_metrics.get("rmse", math.nan))

        val_rmse = val_metrics.get("rmse", float("inf"))
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            early_stop_counter = 0
            ckpt_path = Path(output_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "val_rmse": val_rmse
            }, ckpt_path)
            logger.info(f"Best model -> {ckpt_path} (val_rmse={val_rmse:.4f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            break

    ckpt_path = Path(output_dir) / "best_model.pt"
    if ckpt_path.exists():
        logger.info("Loading model for eval...")
        with torch.serialization.safe_globals([]):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model_state"])

    final_test_metrics = evaluate(model, test_loader)
    logger.info(f"Test metrics: RMSE->{final_test_metrics.get('rmse',np.nan):.4f}, r2->{final_test_metrics.get('r2',np.nan):.4f}, Pearson->{final_test_metrics.get('pearson',np.nan):.4f}")

    Path(output_dir).joinpath("results.json").write_text(json.dumps({
        "final_test": final_test_metrics,
        "best_val_rmse": best_val,
        "best_epoch": best_epoch
    }, indent=2))

    writer.close()
    logger.info("Training done")

# train(epochs=1, batch_size=4, lr=1e-4, hidden_dim=128, encoder_layers=3, patience=10)