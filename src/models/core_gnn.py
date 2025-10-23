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
        
    def forward(self, data):
        x, edge_index, edge_attr, batch, node_type = (
            data.x, data.edge_index, data.edge_attr, data.batch, data.node_type
        )
        src, dst = edge_index
        same_mask = (node_type[src] == node_type[dst])
        cross_mask = (node_type[src] != node_type[dst])

        intra_edge_index = edge_index[:, same_mask]
        intra_edge_attr = edge_attr[same_mask]
        cross_edge_index = edge_index[:, cross_mask]
        cross_edge_attr = edge_attr[cross_mask]

        # ligand + protein separately
        for conv, norm in zip(self.encoder_layers, self.encoder_norms):
            x = conv(x, intra_edge_index, intra_edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        
