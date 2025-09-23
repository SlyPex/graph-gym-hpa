import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GCNConv
import logging
from torch_geometric.nn import GATv2Conv, GCNConv




def flatten_graph_data(data):
    node_feats_flat = data.x.flatten()
    edge_feats_flat = data.edge_attr.flatten()
    return torch.cat([node_feats_flat, edge_feats_flat])



class AdvancedGNNExtractor(BaseFeaturesExtractor):
    """
    An advanced GNN Feature Extractor for SB3, using simple print statements for debugging.

    This architecture avoids global pooling and uses GATv2Conv to incorporate
    both node and edge features dynamically.

    Args:
        observation_space: Gym space.
        num_nodes (int): Number of nodes (microservices) in the graph.
        node_feature_dim (int): Number of features per node.
        num_edges (int): Number of edges in the graph.
        edge_feature_dim (int): Number of features per edge.
        edge_index (torch.Tensor): Static edge connectivity (shape: [2, num_edges]).
        features_dim (int): The final output dimension of the feature extractor.
    """

    def __init__(
        self,
        observation_space,
        num_nodes: int,
        node_feature_dim: int,
        num_edges: int,
        edge_feature_dim: int,
        edge_index: torch.Tensor,
        features_dim: int = 256,
    ):
        gnn_out_dim = 64
        super().__init__(observation_space, features_dim)

        # print("--- Initializing AdvancedGNNExtractor ---")

        # Graph metadata
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.num_edges = num_edges
        self.edge_feature_dim = edge_feature_dim

        self.register_buffer("edge_index", edge_index)
        # print(f"Registered static edge_index with shape: {self.edge_index.shape}")

        # --- GNN Layers ---
        self.conv1 = GATv2Conv(
            in_channels=node_feature_dim,
            out_channels=128,
            heads=4,
            concat=True,
            edge_dim=edge_feature_dim
        )
        self.norm1 = nn.LayerNorm(128 * 4)
        self.conv2 = GATv2Conv(
            in_channels=128 * 4,
            out_channels=gnn_out_dim,
            heads=4,
            concat=True,
            edge_dim=edge_feature_dim
        )
        self.norm2 = nn.LayerNorm(gnn_out_dim * 4)
        # print("GNN layers (GATv2Conv) and LayerNorm initialized.")

        # --- MLP Head ---
        flattened_dim = (gnn_out_dim * 4) * self.num_nodes
        self.linear_head = nn.Sequential(
            nn.Linear(flattened_dim, features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
        )
        # print(f"MLP head initialized with input dim {flattened_dim} and output dim {features_dim}")
        # print("--- Initialization complete ---\n")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of observations.

        Args:
            observations (torch.Tensor): Tensor of shape (batch_size, flattened_graph_data)

        Returns:
            features (torch.Tensor): Tensor of shape (batch_size, features_dim)
        """
        # Uncomment the line below to see output for every forward pass
        # print(f"\n--- Forward Pass ---")
        # print(f"Received observations with shape: {observations.shape}")
        
        batch_features = []
        for i, obs in enumerate(observations):
            # --- 1. Deconstruct the flat observation vector ---
            node_feat_size = self.num_nodes * self.node_feature_dim
            edge_feat_size = self.num_edges * self.edge_feature_dim

            node_feats = obs[:node_feat_size].reshape(
                self.num_nodes, self.node_feature_dim
            )
            edge_feats = obs[
                node_feat_size : node_feat_size + edge_feat_size
            ].reshape(self.num_edges, self.edge_feature_dim)
            
            # --- 2. GNN message passing ---
            h = self.conv1(node_feats, self.edge_index, edge_attr=edge_feats)
            h = torch.relu(self.norm1(h))
            h = self.conv2(h, self.edge_index, edge_attr=edge_feats)
            h = self.norm2(h)
            
            # --- 3. Flatten node features (NO global pooling) ---
            graph_feat = h.flatten()
            batch_features.append(graph_feat)

        # Stack the features and process through the MLP head
        stacked_features = torch.stack(batch_features)
        final_features = self.linear_head(stacked_features)
        
        # print(f"Final output features shape: {final_features.shape}")
        # print(graph_feat.shape)
        # print("--- End Forward Pass ---")
        return final_features