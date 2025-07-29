import logging
from torch_geometric.nn import NNConv
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GCNConv

class CustomGNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        num_nodes,
        node_feature_dim,
        num_edges,
        edge_feature_dim,
        edge_index,
        features_dim=128,
    ):
        super().__init__(observation_space, features_dim)

        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.num_edges = num_edges
        self.edge_feature_dim = edge_feature_dim

        self.register_buffer("edge_index", edge_index)

        # Define edge MLP for edge-conditioned convolution
        self.edge_mlp1 = Sequential(
            Linear(edge_feature_dim, 64),
            ReLU(),
            Linear(64, node_feature_dim * 64)
        )
        self.conv1 = NNConv(node_feature_dim, 64, self.edge_mlp1, aggr='mean')

        self.edge_mlp2 = Sequential(
            Linear(edge_feature_dim, 64),
            ReLU(),
            Linear(64, 64 * 64)
        )
        self.conv2 = NNConv(64, 64, self.edge_mlp2, aggr='mean')

        self.linear = nn.Linear(64, features_dim)

# Configure logger once (you can move this to your __init__ or module setup)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Change to INFO or ERROR in production
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
    def forward(self, observations):
        """
        Forward pass for a single observation (batch size = 1).

        Args:
            observations: Tensor of shape (1, flattened graph data)

        Returns:
            features: Tensor of shape (1, features_dim)
        """
        # Remove batch dimension: (1, obs_dim) → (obs_dim,)
        observations = observations[0]

        # Calculate expected sizes
        node_feat_size = self.num_nodes * self.node_feature_dim
        edge_feat_size = self.num_edges * self.edge_feature_dim
        total_expected_size = node_feat_size + edge_feat_size
        print("node_feats: " , len(observations))
        print("node_feats_size: " , node_feat_size)
        print("edge_feats: " , edge_feat_size )
        print("node_feats: " , total_expected_size)
        # Debug: Check if input is as expected
        if observations.shape[0] != total_expected_size:
            raise ValueError(
                f"[GNNExtractor] Mismatched observation size: "
                f"Expected {total_expected_size}, got {observations.shape[0]}. "
                f"Check num_nodes, node_feature_dim, num_edges, edge_feature_dim."
            )

        # Split node features
        node_feats = observations[:node_feat_size].reshape(self.num_nodes, self.node_feature_dim)
        # Split and reshape edge features
        edge_feats_flat = observations[node_feat_size:]
        
        edge_feats = edge_feats_flat.reshape(self.num_edges, self.edge_feature_dim)

        # Run GNN layers
        h = torch.relu(self.conv1(node_feats, self.edge_index, edge_feats))
        h = torch.relu(self.conv2(h, self.edge_index, edge_feats))

        print("c")
        # Global mean pooling over nodes → graph-level feature
        graph_feat = h.mean(dim=0)

        # Project to feature_dim and return with batch dimension
        return self.linear(graph_feat).unsqueeze(0)

