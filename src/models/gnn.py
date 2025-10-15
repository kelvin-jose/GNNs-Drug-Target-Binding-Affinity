import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, global_mean_pool
from utils.logger import setup_logging

from utils.logger import setup_logging

logger = setup_logging()

class BaselineGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.pool = global_mean_pool

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 1)
        )