import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from src.utils.logger import setup_logging

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
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.pool(x, batch)
        out = self.head(h).squeeze(-1)
        return out