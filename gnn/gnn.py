import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class Gnn(nn.Module):
    def __init__(self, node_feature_keys, edge_feature_keys, hidden_dim=32, out_dim=16):
        super().__init__()
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.conv1 = GATv2Conv(
            len(node_feature_keys), hidden_dim, edge_dim=len(edge_feature_keys)
        )
        self.conv2 = GATv2Conv(hidden_dim, out_dim, edge_dim=len(edge_feature_keys))

    def forward(self, data):
        pyg_data = data
        x = torch.stack([feat for feat in pyg_data.x], dim=0)
        edge_index = pyg_data.edge_index
        edge_attr = torch.stack([feat for feat in pyg_data.edge_attr], dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        obs = x.flatten()
        return obs
