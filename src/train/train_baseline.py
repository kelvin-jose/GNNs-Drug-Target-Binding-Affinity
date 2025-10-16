import torch
import random
import numpy as np
from utils.logger import setup_logging
from sklearn.metrics import root_mean_squared_error, r2_score

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
    rmse = root_mean_squared_error(y, p, squared=False)
    r2 = r2_score(y, p)
    return {"rmse": float(rmse), "r2": float(r2)}