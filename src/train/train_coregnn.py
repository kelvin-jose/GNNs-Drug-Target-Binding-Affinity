import torch
import random
import numpy as np

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