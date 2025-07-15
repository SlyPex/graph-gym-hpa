import torch
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
        """
        GNN Feature Extractor for SB3 (single observation version).

        Args:
            observation_space: Gym space, expects a flat tensor (node and edge features concatenated).
            num_nodes: Number of nodes in the graph.
            node_feature_dim: Features per node.
            num_edges: Number of edges in the graph.
            edge_feature_dim: Features per edge.
            edge_index: Static edge connectivity (shape: [2, num_edges]).
            features_dim: Output dimension of the feature extractor.
        """
        super().__init__(observation_space, features_dim)

        # Save graph metadata
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.num_edges = num_edges
        self.edge_feature_dim = edge_feature_dim

        # Register static edge_index so it's on the correct device
        self.register_buffer("edge_index", edge_index)

        # Define GNN layers
        self.conv1 = GCNConv(node_feature_dim, 64)
        self.conv2 = GCNConv(64, 64)

        # Final linear layer for projection to features_dim
        self.linear = nn.Linear(64, features_dim)

    def forward(self, observations):
        """
        Forward pass for a **single observation (batch size = 1)**.

        Args:
            observations: Tensor of shape (1, flattened graph data)

        Returns:
            features: Tensor of shape (1, features_dim)
        """

        # Remove batch dimension to simplify (input is (1, obs_dim), take the first element)
        observations = observations[0]

        # Split the flat observation into node and edge features
        node_feat_size = self.num_nodes * self.node_feature_dim
        edge_feat_size = self.num_edges * self.edge_feature_dim

        # Reshape to (num_nodes, node_feature_dim)
        node_feats = observations[:node_feat_size].reshape(
            self.num_nodes, self.node_feature_dim
        )

        # Reshape edge features (ignored in this GCN, but available if needed later)
        # edge_feats = observations[
        # node_feat_size : node_feat_size + edge_feat_size
        # ].reshape(self.num_edges, self.edge_feature_dim)

        # Run GCN layers
        h = torch.relu(self.conv1(node_feats, self.edge_index))
        h = torch.relu(self.conv2(h, self.edge_index))

        # Global mean pooling over nodes → graph-level feature
        graph_feat = h.mean(dim=0)

        # Apply final linear layer and add batch dimension back (SB3 expects (1, features_dim))
        return self.linear(graph_feat).unsqueeze(0)
