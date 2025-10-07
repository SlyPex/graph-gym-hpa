import torch
import pandas as pd
import requests
import time
from torch_geometric.data import Data

# ---------------------------
# Constants
# ---------------------------
NUM_FEATURES_PER_SERVICE = 4  # cpu_ratio, mem_ratio, pod_count, desired_pod_count

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

PROMETHEUS_URL = "http://localhost:31090"

# ---------------------------
# Fetch traffic
# ---------------------------
def fetch_prom(query, prometheus_url=PROMETHEUS_URL, retry_sleep=5, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(
                f"{prometheus_url}/api/v1/query", params={"query": query}, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                # print(f"[WARN] Prometheus query failed: {query}, retry {retries+1}")
                retries += 1
                time.sleep(retry_sleep)
                continue
            result = data["data"]["result"]
            if not result:
                raise ValueError(f"No result from Prometheus for query: {query}")
            _, value_str = result[0]["value"]
            # print(f"[DEBUG] Prometheus query success: {query} -> {value_str}")
            return float(value_str)
        except Exception as e:
            # print(f"[ERROR] Exception fetching Prometheus query: {query}, retry {retries+1}, {e}")
            retries += 1
            time.sleep(retry_sleep)
    raise RuntimeError(f"Prometheus query failed after {max_retries} retries: {query}")

def get_traffic_between_services(prometheus_url=PROMETHEUS_URL):
    traffic_data = []
    for source_name, destinations in service_dependencies.items():
        for dest_name in destinations:
            query = (
                f'rate(istio_requests_total{{source_app="{source_name}",'
                f'destination_service_name="{dest_name}"}}[1m])'
            )
            val = fetch_prom(query, prometheus_url)
            traffic_data.append({
                "source": source_name,
                "destination": dest_name,
                "traffic": val,
            })
            # print(f"[DEBUG] Traffic {source_name}->{dest_name} = {val}")
    df = pd.DataFrame(traffic_data)
    # print("[INFO] Traffic DataFrame built:\n", df)
    return df
# ---------------------------
# Build graph from metrics dict
# ---------------------------
def build_graph(metrics_dict, traffic_metrics):
    nodes = []
    for name in service_names:
        m = metrics_dict.get(name)
        if not m:
            raise ValueError(f"Missing metrics for service {name}")
        node = {
            "name": name,
            "cpu_ratio": float(m.get("cpu_ratio", 0.0)),
            "mem_ratio": float(m.get("mem_ratio", 0.0)),
            "pod_count": int(m.get("pod_count", 0)),
            "desired_pod_count": int(m.get("desired_pod_count", 0)),
        }
        # print("[DEBUG] Node built:", node)
        nodes.append(node)

    edges = []
    for source, destinations in service_dependencies.items():
        for dest in destinations:
            traffic_row = traffic_metrics[
                (traffic_metrics["source"] == source) &
                (traffic_metrics["destination"] == dest)
            ]
            if traffic_row.empty:
                raise ValueError(f"No traffic data for edge {source} -> {dest}")
            traffic_val = float(traffic_row["traffic"].iloc[0])
            edges.append({"source": source, "target": dest, "traffic": traffic_val})
            # print(f"[DEBUG] Edge built: {source} -> {dest}, traffic={traffic_val}")

    return {"nodes": nodes, "edges": edges}

# ---------------------------
# Convert graph to PyG Data
# ---------------------------
def graph_to_data(graph):
    node_list = graph["nodes"]
    node_names = [node["name"] for node in node_list]
    node_idx_map = {name: idx for idx, name in enumerate(node_names)}

    # Node features
    x = torch.tensor(
        [[n["cpu_ratio"], n["mem_ratio"], n["pod_count"], n["desired_pod_count"]] for n in node_list],
        dtype=torch.float
    )

    # Edges
    edge_index = []
    edge_attr_list = []
    for edge in graph["edges"]:
        src = node_idx_map[edge["source"]]
        tgt = node_idx_map[edge["target"]]
        edge_index.append([src, tgt])
        edge_attr_list.append([edge["traffic"]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # print("[DEBUG] Node features (x):\n", x)
    # print("[DEBUG] Edge index:\n", edge_index)
    # print("[DEBUG] Edge features (edge_attr):\n", edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------------------------
# Flatten graph for RL
# ---------------------------
def flatten_graph_data(data):
    return torch.cat([data.x.flatten(), data.edge_attr.flatten()])
