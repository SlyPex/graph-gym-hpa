import csv
import logging
import time
from datetime import datetime

import requests
from kubernetes import client, config

# -------------------
# Config
# -------------------
PROM_URL = "http://localhost:9090/api/v1/query"
NAMESPACE = "onlineboutique"
REPLICA_FILE = "replicas.csv"
REPLICA_METRICS_FILE = "replicas_metrics.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------
# Kubernetes setup
# -------------------
try:
    config.load_kube_config()
except Exception:
    config.load_incluster_config()
apps_api = client.AppsV1Api()


# -------------------
# Prometheus helpers
# -------------------
def query_prometheus(query: str):
    try:
        resp = requests.get(PROM_URL, params={"query": query}, timeout=5)
        resp.raise_for_status()
        results = resp.json()["data"]["result"]
        return results
    except Exception as e:
        logging.error(f"Prometheus query failed: {e}")
        return []


def get_cpu_usage():
    # CPU in cores
    query = f'rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",container!=""}}[1m])'
    results = query_prometheus(query)
    return {
        r["metric"].get("pod"): float(r["value"][1])
        for r in results
        if "pod" in r["metric"]
    }


def get_mem_usage():
    query = (
        f'container_memory_working_set_bytes{{namespace="{NAMESPACE}",container!=""}}'
    )
    results = query_prometheus(query)
    return {
        r["metric"].get("pod"): float(r["value"][1])
        for r in results
        if "pod" in r["metric"]
    }


# -------------------
# Kubernetes helpers
# -------------------
def get_replicas():
    deployments = apps_api.list_namespaced_deployment(namespace=NAMESPACE)
    replica_data = {}
    for dep in deployments.items:
        replica_data[dep.metadata.name] = (
            dep.spec.replicas or 0,
            dep.status.available_replicas or 0,
        )
    return replica_data


# -------------------
# Writers
# -------------------
def init_csv_files():
    with (
        open(REPLICA_FILE, "w", newline="") as f1,
        open(REPLICA_METRICS_FILE, "w", newline="") as f2,
    ):
        writer1 = csv.DictWriter(
            f1,
            fieldnames=[
                "timestamp",
                "deployment",
                "desired_replicas",
                "available_replicas",
            ],
        )
        writer2 = csv.DictWriter(
            f2,
            fieldnames=[
                "timestamp",
                "deployment",
                "desired_replicas",
                "available_replicas",
                "pod",
                "cpu_cores",
                "memory_bytes",
            ],
        )
        writer1.writeheader()
        writer2.writeheader()


def log_metrics():
    timestamp = datetime.utcnow().isoformat()
    replica_data = get_replicas()
    cpu_usage = get_cpu_usage()
    mem_usage = get_mem_usage()

    with (
        open(REPLICA_FILE, "a", newline="") as f1,
        open(REPLICA_METRICS_FILE, "a", newline="") as f2,
    ):
        writer1 = csv.DictWriter(
            f1,
            fieldnames=[
                "timestamp",
                "deployment",
                "desired_replicas",
                "available_replicas",
            ],
        )
        writer2 = csv.DictWriter(
            f2,
            fieldnames=[
                "timestamp",
                "deployment",
                "desired_replicas",
                "available_replicas",
                "pod",
                "cpu_cores",
                "memory_bytes",
            ],
        )

        for dep, (desired, available) in replica_data.items():
            # log replicas only
            writer1.writerow(
                {
                    "timestamp": timestamp,
                    "deployment": dep,
                    "desired_replicas": desired,
                    "available_replicas": available,
                }
            )

            # log replicas + pod metrics (filter pods by prefix match to avoid mixing)
            for pod, cpu in cpu_usage.items():
                if pod.startswith(dep):
                    row = {
                        "timestamp": timestamp,
                        "deployment": dep,
                        "desired_replicas": desired,
                        "available_replicas": available,
                        "pod": pod,
                        "cpu_cores": round(cpu, 4),
                        "memory_bytes": mem_usage.get(pod, 0),
                    }
                    writer2.writerow(row)


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    init_csv_files()
    logging.info("Starting metrics collection loop ...")
    try:
        while True:
            log_metrics()
            time.sleep(10)
    except KeyboardInterrupt:
        logging.info("Stopped.")
