# imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import matplotlib.patches as mpatches
import torch
from torch_geometric.data import Data
import requests
import time
# from gym_hpa.envs.online_boutique import OnlineBoutique

PROMETHEUS_URL = "http://localhost:9090"

NUM_FEATURES_PER_SERVICE = 4  # cpu, ram, num_pods, desired_pods
service_names = [
    "recommendationservice",
    "productcatalogservice",
    "cartservice",
    "adservice",
    "paymentservice",
    "shippingservice",
    "currencyservice",
    "redis-cart",
    "checkoutservice",
    "frontend",
    "emailservice",
]


# Define which services interact (caller → callee)
service_dependencies = {
    "frontend": [
        "recommendationservice",
        "productcatalogservice",
        "cartservice",
        "checkoutservice",
        "currencyservice",
        "shippingservice",
        "adservice",
    ],
    "recommendationservice": ["productcatalogservice"],
    # "cartservice": ["redis-cart"],
    "checkoutservice": [
        "cartservice",
        "paymentservice",
        "shippingservice",
        "currencyservice",
        "productcatalogservice",
        "emailservice",
    ],
    "paymentservice": [],
    "shippingservice": [],
    "emailservice": [],
    "adservice": [],
    "currencyservice": [],
    "productcatalogservice": [],
    "redis-cart": [],
}

## to build random traffic data
traffic_metrics_list = []
for source, destinations in service_dependencies.items():
    for destination in destinations:
        traffic_metrics_list.append(
            {
                "source": source,
                "destination": destination,
                "traffic": random.uniform(0, 100),
            }
        )
traffic_metrics_df = pd.DataFrame(traffic_metrics_list)


def build_graph(
    metrics, traffic_metrics=traffic_metrics_df, num_features=NUM_FEATURES_PER_SERVICE
):
    # Step 1: build node feature map
    num_services = len(service_names)
    nodes = []
    for i in range(num_services):
        start = i * num_features
        features = metrics[start : start + num_features]
        nodes.append(
            {
                # 'id': i,
                "name": service_names[i],
                "cpu_usage": features[0],
                "mem_usage": features[1],
                "pod_count": features[2],
                "desired_pod_count": features[3],
            }
        )

    # Step 2: build edges
    edges = []
    for i in service_dependencies.keys():
        if service_dependencies[i] != []:
            for j in service_dependencies[i]:
                traffic = float(
                    traffic_metrics[
                        (traffic_metrics["source"] == i)
                        & (traffic_metrics["destination"] == j)
                    ]["traffic"].iloc[0]
                )
                # print(traffic)
                edges.append({"source": i, "target": j, "traffic": traffic})

    # Step 3: return graph as dict
    graph = {"nodes": nodes, "edges": edges}
    # print(graph)
    return graph


def visualize_service_graph(
    graph_dict,
    figsize=(12, 8),
    node_colors=None,
    edge_colors=None,
    show_weights=True,
    random_colors=True,
    color_seed=None,
):
    """
    Visualizes a service graph with circular layout and styling.

    Args:
        graph_dict: Dictionary with 'nodes' and 'edges' keys
        figsize: Figure size tuple
        node_colors: Dict mapping node names to colors, or single color (overrides random_colors)
        edge_colors: Dict mapping edges to colors, or single color
        show_weights: Whether to show edge weights as labels
        random_colors: Whether to use random colors for nodes (default True)
        color_seed: Seed for random color generation (None for truly random)
    """

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in graph_dict["nodes"]:
        node_type = node.get("type", "service")
        G.add_node(
            node["name"],
            label=node["name"],
            type=node_type,
            size=node.get("size", 1000),
        )

    # Add edges with attributes
    for edge in graph_dict["edges"]:
        weight = edge.get("traffic", 1)
        G.add_edge(
            edge["source"], edge["target"], weight=weight, label=edge.get("label", "")
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout algorithm

    pos = nx.circular_layout(G)

    # Determine node colors
    if node_colors is None and random_colors:
        # Generate random colors for each node
        if color_seed is not None:
            np.random.seed(color_seed)  # For reproducible colors

        node_colors = []
        for _ in G.nodes():
            # Generate bright, vibrant colors
            hue = np.random.random()
            saturation = 0.7 + np.random.random() * 0.3  # 70-100% saturation
            value = 0.8 + np.random.random() * 0.2  # 80-100% value

            # Convert HSV to RGB
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            node_colors.append(hex_color)
    elif node_colors is None:
        # Default blue color if random_colors is False
        node_colors = ["#45B7D1"] * len(G.nodes())

    # Determine node sizes
    node_sizes = [G.nodes[node].get("size", 1000) for node in G.nodes()]

    # Draw nodes with enhanced styling
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="black",
        linewidths=1.5,
    )

    # Draw node labels with better positioning
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, font_size=10, font_weight="bold", font_color="black"
    )

    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 3 for w in edge_weights]

    edge_color = edge_colors if edge_colors else "#666666"
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_color,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=edge_widths,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )

    # Add edge labels if requested
    if show_weights:
        edge_labels = {}
        for u, v in G.edges():
            weight = G[u][v]["weight"]
            label = G[u][v].get("label", "")
            if label:
                edge_labels[(u, v)] = f"{label}\n({weight})"
            else:
                edge_labels[(u, v)] = f"{weight:.2f} kbps"

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add title and styling
    plt.title("Microservice Interaction Graph", fontsize=16, fontweight="bold", pad=20)

    # Remove type-based legend since we're using random colors
    # Legend is now optional and only shown if explicitly using typed nodes
    if node_colors is not None and isinstance(node_colors, dict):
        legend_elements = []
        for node_type, color in node_colors.items():
            legend_elements.append(mpatches.Patch(color=color, label=node_type.title()))
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G, pos


def graph_to_data(graph):
    node_list = graph["nodes"]
    node_names = [node["name"] for node in node_list]
    node_idx_map = {name: idx for idx, name in enumerate(node_names)}

    x = torch.tensor(
        [
            [
                node["cpu_usage"],
                node["mem_usage"],
                node["pod_count"],
                node["desired_pod_count"],
            ]
            for node in node_list
        ],
        dtype=torch.float,
    )

    edge_index = []
    edge_attr = []

    for edge in graph["edges"]:
        src = node_idx_map[edge["source"]]
        tgt = node_idx_map[edge["target"]]
        edge_index.append([src, tgt])
        edge_attr.append([edge["traffic"]])  # shape = [num_edges, 1]

    edge_index = (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )  # shape: [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def fetch_prom(query, prometheus_url=PROMETHEUS_URL, retry_sleep=5, max_retries=3):
    """
    Fetch data from Prometheus API with retry logic.

    Args:
        query: PromQL query string
        prometheus_url: Prometheus server URL
        retry_sleep: Sleep time between retries in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        List of query results or None if failed
    """
    retries = 0

    while retries < max_retries:
        try:
            print(query)
            response = requests.get(
                f"{prometheus_url}/api/v1/query", params={"query": query}, timeout=30
            )  # Add timeout
            data = response.json()
            timestamp, value_str = data['data']['result'][0]["value"]
            print(query, value_str)
            # exit()
            # Check if request was successful
            response.raise_for_status()

            # Parse JSON response
            json_data = response.json()

            if json_data.get("status") != "success":
                error_msg = json_data.get("error", "Unknown error")
                print(f"Prometheus query failed: {error_msg}")
                print(f"Query: {query}")

                retries += 1
                if retries < max_retries:
                    print(
                        f"Retrying in {retry_sleep}s... (attempt {retries}/{max_retries})"
                    )
                    time.sleep(retry_sleep)
                    continue
                else:
                    print("Max retries reached, returning None")
                    return None

            # result = json_data["data"]["result"]
            return float(value_str)

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            retries += 1
            if retries < max_retries:
                print(
                    f"Retrying in {retry_sleep}s... (attempt {retries}/{max_retries})"
                )
                time.sleep(retry_sleep)
            else:
                print("Max retries reached, returning None")
                return None

        except (ValueError, KeyError) as e:
            print(f"Error parsing Prometheus response: {e}")
            return None

    return None


def _get_traffic_from_prometheus(prometheus_url=PROMETHEUS_URL):
    """Get traffic data from Prometheus metrics using the fixed fetch_prom function"""
    try:
        traffic_data = []

        for source_name, destinations in service_dependencies.items():
            for dest_name in destinations:
                # Query network traffic between services using Istio metrics
                query = f'rate(istio_requests_total{{source_app="{source_name}",destination_service_name="{dest_name}"}}[1m])'

                # Use the fixed fetch_prom function
                result = fetch_prom(query, prometheus_url)

                # if result and len(result) > 0:
                #     try:
                #         # Extract traffic value from Prometheus result
                #         traffic_value = (
                #             float(result[0]["value"][1]) * 1000
                #         )  # Convert to appropriate scale
                #     except (IndexError, KeyError, ValueError) as e:
                #         print(
                #             f"Error parsing result for {source_name}->{dest_name}: {e}"
                #         )
                #         traffic_value = np.random.uniform(10, 80)  # Fallback
                # else:
                #     print(
                #         f"No data found for {source_name}->{dest_name}, using fallback"
                #     )
                #     traffic_value = np.random.uniform(10, 80)  # Fallback

                traffic_data.append(
                    {
                        "source": source_name,
                        "destination": dest_name,
                        "traffic": round(result, 2),
                    }
                )

        return pd.DataFrame(traffic_data)

    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return _get_simulated_traffic()


def _get_simulated_traffic(seed=42):
    """Fallback function for simulated traffic when Prometheus is unavailable"""
    np.random.seed(seed)
    traffic_data = []

    for source_name, destinations in service_dependencies.items():
        for dest_name in destinations:
            traffic = np.random.uniform(10, 90)
            traffic_data.append(
                {
                    "source": source_name,
                    "destination": dest_name,
                    "traffic": round(traffic, 2),
                }
            )

    return pd.DataFrame(traffic_data)


def get_traffic_between_services(
    method="simulated", prometheus_url=PROMETHEUS_URL, **kwargs
):
    """
    Main function to get traffic data between services.

    Args:
        method: 'prometheus' or 'simulated'
        prometheus_url: Prometheus server URL
        **kwargs: Additional arguments for specific methods

    Returns:
        pandas.DataFrame: Traffic data
    """
    if method == "prometheus":
        print("Attempting to fetch traffic data from Prometheus...")
        print("##########################||||||||||||||||||||||||||||||||############")
        return _get_traffic_from_prometheus(prometheus_url)
    else:
        print("Using simulated traffic data...")
        return _get_simulated_traffic(**kwargs)

def build_graph_with_sim_traffic(metrics):
    ## to build random traffic data
    traffic_metrics_list = []
    for source, destinations in service_dependencies.items():
        for destination in destinations:
            traffic_metrics_list.append(
                {
                    "source": source,
                    "destination": destination,
                    "traffic": random.uniform(0, 100),
                }
            )
    traffic_metrics_df = pd.DataFrame(traffic_metrics_list)
    return build_graph(metrics = metrics, traffic_metrics=traffic_metrics_df)


## add get_real_traffic_df
