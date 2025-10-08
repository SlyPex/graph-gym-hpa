import networkx as nx
import torch
from gnn import Gnn
from torch_geometric.utils import from_networkx


def test_gnn_module():
    """
    Tests the GNN module by creating a sample graph,
    passing it through the GNN, and printing the output.
    """
    print("--- Testing GNN Module ---")

    # 1. Define the feature keys that the GNN expects.
    # These must match the attributes you add to your graph nodes and edges.
    node_feature_keys = ["cpu_usage", "mem_usage", "num_pods"]
    edge_feature_keys = ["traffic"]

    # 2. Create a sample microservice dependency graph using networkx.
    G = nx.DiGraph()

    # Add nodes (representing microservices) with their features.
    G.add_node(0, cpu_usage=0.6, mem_usage=0.5, num_pods=3)  # Frontend service
    G.add_node(1, cpu_usage=0.4, mem_usage=0.7, num_pods=2)  # Backend service
    G.add_node(2, cpu_usage=0.2, mem_usage=0.3, num_pods=1)  # Database service

    # Add edges (representing service-to-service calls) with their features.
    G.add_edge(0, 1, traffic=150.5)  # Frontend calls Backend
    G.add_edge(1, 2, traffic=90.0)  # Backend calls Database
    G.add_edge(1, 0, traffic=10.0)  # Health check from Backend to Frontend

    print(
        f"Created a sample graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    graph_data = from_networkx(
        G=G, group_node_attrs=node_feature_keys, group_edge_attrs=edge_feature_keys
    )

    # 3. Instantiate the GNN model.
    # The parameters should match the features defined above.
    gnn_model = Gnn(
        node_feature_keys=node_feature_keys,
        edge_feature_keys=edge_feature_keys,
        hidden_dim=32,
        out_dim=16,  # Each node will have an 8-dimensional embedding
    )
    print("GNN model instantiated successfully.")

    # 4. Pass the graph through the GNN to get the embeddings.
    # This calls the forward() method of your Gnn class.
    # The model is in eval mode because we are not training here.
    gnn_model.eval()
    with torch.no_grad():
        observation_embedding = gnn_model(graph_data)

    # 5. Print the results.
    print("\n--- GNN Output ---")
    print(f"Shape of the final observation embedding: {observation_embedding.shape}")
    print("Final observation embedding vector:")
    print(observation_embedding)
    print("\nTest finished successfully!")


if __name__ == "__main__":
    test_gnn_module()
