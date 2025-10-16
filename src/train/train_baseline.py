import torch
import random
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import root_mean_squared_error, r2_score

from src.models.gnn import BaselineGNN
from src.utils.logger import setup_logging
from src.data.pyg_dataset import PDBBindLigandDataset


logger = setup_logging()

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=SEED):
    num = len(dataset)
    idx = list(range(num))
    random.Random(seed).shuffle(idx)
    ntrain = int(frac_train * num)
    nval = int(frac_val * num)
    train_idx = idx[:ntrain]
    val_idx = idx[ntrain:ntrain+nval]
    test_idx = idx[ntrain+nval:]
    return train_idx, val_idx, test_idx

def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        pred = model(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y.view(-1).to(pred.dtype))
        loss.backward()
        optim.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            ys.append(batch.y.cpu().numpy().reshape(-1))
            preds.append(out.cpu().numpy().reshape(-1))
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    rmse = root_mean_squared_error(y, p)
    r2 = r2_score(y, p)
    return {"rmse": float(rmse), "r2": float(r2)}

def train(batch_size=32, epochs=10, lr=1e-3, device=None):
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="experiments/runs/ligand_baseline")
    logger.info(f"Starting ligand-only baseline training on device={device}")

    # Load dataset (process & cache)
    dataset = PDBBindLigandDataset(metadata_csv="data/processed/refined_dataset_metadata.csv", 
                                   root="data/processed/graphs", force_rebuild=False)
    logger.info(f"Dataset size: {len(dataset)}")

    train_idx, val_idx, test_idx = scaffold_split(dataset, seed=SEED)
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sample = dataset.get(0)
    in_channels = sample.x.shape[1]
    model = BaselineGNN(in_channels=in_channels, hidden_channels=128, num_layers=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optim, device)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        logger.info(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_rmse {val_metrics['rmse']:.4f} | test_rmse {test_metrics['rmse']:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("rmse/val", val_metrics["rmse"], epoch)
        writer.add_scalar("rmse/test", test_metrics["rmse"], epoch)
        writer.add_scalar("r2/val", val_metrics["r2"], epoch)
        writer.add_scalar("r2/test", test_metrics["r2"], epoch)

        # save best
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            torch.save(model.state_dict(), "experiments/runs/ligand_baseline/best_model.pt")
            logger.info(f"Saved best model (val_rmse={best_val:.4f})")

    writer.close()
    logger.info("Training complete. Best val rmse: %.4f" % best_val)

if __name__ == "__main__":
    train()
