import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, global_mean_pool

class CoreGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            GATConv(
                in_channels=node_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim,
                heads=4,
                concat=False
            )
            for i in range(num_layers)
        ])
        self.encoder_norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])

        self.cross_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_dim,
                heads=4,
                concat=False
            )
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
    