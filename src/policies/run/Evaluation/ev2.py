#!/usr/bin/python3
# IMPORTANT: monkey-patch before any stdlib imports that use ssl/urllib3/requests
from gevent import monkey
monkey.patch_all()

import csv
import time
import random
from datetime import datetime
import datetime as dt
import requests

from faker import Faker
from locust import FastHttpUser, TaskSet, between
from locust.env import Environment
from kubernetes import client, config

# ---------------- Locust workload ----------------
fake = Faker()
products = [
    '0PUK6V6EV0','1YMWWN1N4O','2ZYFJ3GM2N','66VCHSJNUP',
    '6E92ZMYYFZ','9SIQT8TOJO','L9ECAV7KIM','LS4PSXUNUM','OLJCESPC7Z'
]

def index(l): l.client.get("/")
def setCurrency(l): l.client.post("/setCurrency", {'currency_code': random.choice(['EUR','USD','JPY','CAD','GBP','TRY'])})
def browseProduct(l): l.client.get("/product/" + random.choice(products))
def viewCart(l): l.client.get("/cart")
def addToCart(l):
    product = random.choice(products)
    l.client.get("/product/" + product)
    l.client.post("/cart", {'product_id': product, 'quantity': random.randint(1,10)})
def empty_cart(l): l.client.post('/cart/empty')
def checkout(l):
    addToCart(l)
    current_year = dt.datetime.now().year+1
    l.client.post("/cart/checkout", {
        'email': fake.email(),
        'street_address': fake.street_address(),
        'zip_code': fake.zipcode(),
        'city': fake.city(),
        'state': fake.state_abbr(),
        'country': fake.country(),
        'credit_card_number': fake.credit_card_number(card_type="visa"),
        'credit_card_expiration_month': random.randint(1, 12),
        'credit_card_expiration_year': random.randint(current_year, current_year+10),
        'credit_card_cvv': f"{random.randint(100, 999)}",
    })
def logout(l): l.client.get('/logout')

class UserBehavior(TaskSet):
    tasks = {
        index: 1,
        setCurrency: 2,
        browseProduct: 10,
        addToCart: 2,
        viewCart: 3,
        checkout: 1,
    }
    def on_start(self): index(self)

class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 10)
    host = "http://localhost:8080"

# ---------------- Config ----------------
config.load_kube_config()
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()

PROM_URL = "http://localhost:9090/api/v1/query"
NAMESPACE = "onlineboutique"

LOCUST_FILE = "locust_metrics.csv"
REPLICAS_FILE = "replicas_metrics.csv"

RUN_ID = "run1"
STRATEGY = "hpa-cpu-70"
LOAD_USERS = 50
SPAWN_RATE = 10
DURATION = 60  # seconds
INTERVAL = 10  # seconds

# ---------------- Helpers ----------------
def query_prometheus(query):
    resp = requests.get(PROM_URL, params={"query": query})
    results = resp.json()["data"]["result"]
    return results

def get_cpu_mem_usage():
    cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",container!="POD"}}[1m])'
    mem_query = f'container_memory_working_set_bytes{{namespace="{NAMESPACE}",container!="POD"}}'

    cpu_results = query_prometheus(cpu_query)
    mem_results = query_prometheus(mem_query)

    cpu_usage = {r["metric"]["pod"]: float(r["value"][1]) for r in cpu_results}
    mem_usage = {r["metric"]["pod"]: float(r["value"][1]) for r in mem_results}

    return cpu_usage, mem_usage

def get_replicas():
    deployments = apps_v1.list_namespaced_deployment(NAMESPACE).items
    replica_data = {}
    for dep in deployments:
        name = dep.metadata.name
        desired = dep.spec.replicas or 0
        available = dep.status.available_replicas or 0
        replica_data[name] = (desired, available)
    return replica_data

def get_pod_to_deployment():
    pods = core_v1.list_namespaced_pod(NAMESPACE).items
    pod_to_dep = {}
    for pod in pods:
        pod_name = pod.metadata.name
        dep_name = pod.metadata.labels.get("app", "unknown")
        pod_to_dep[pod_name] = dep_name
    return pod_to_dep

# ---------------- Main ----------------
if __name__ == "__main__":
    # Locust environment
    env = Environment(user_classes=[WebsiteUser])
    env.create_local_runner()
    env.runner.start(LOAD_USERS, spawn_rate=SPAWN_RATE)

    with open(LOCUST_FILE, "w", newline="") as f1, open(REPLICAS_FILE, "w", newline="") as f2:
        writer1 = csv.DictWriter(f1, fieldnames=[
            "timestamp", "run_id", "strategy", "load_users", "spawn_rate",
            "endpoint", "rps", "total_requests", "failures",
            "avg_latency_ms", "p95_latency_ms"
        ])
        writer1.writeheader()

        writer2 = csv.DictWriter(f2, fieldnames=[
            "timestamp", "deployment", "desired_replicas", "available_replicas",
            "pod", "cpu_cores", "memory_bytes"
        ])
        writer2.writeheader()

        start_time = time.time()
        while time.time() - start_time < DURATION:
            timestamp = datetime.utcnow().isoformat()

            # ---- Locust metrics ----
            for (method, name), stats in env.runner.stats.entries.items():
                row = {
                    "timestamp": timestamp,
                    "run_id": RUN_ID,
                    "strategy": STRATEGY,
                    "load_users": LOAD_USERS,
                    "spawn_rate": SPAWN_RATE,
                    "endpoint": f"{method} {name}",
                    "rps": round(stats.current_rps, 2),
                    "total_requests": stats.num_requests,
                    "failures": stats.num_failures,
                    "avg_latency_ms": round(stats.avg_response_time or 0, 2),
                    "p95_latency_ms": round(stats.get_response_time_percentile(0.95) or 0, 2),
                }
                writer1.writerow(row)
            f1.flush()

            # ---- Replica + CPU/Mem ----
            replica_data = get_replicas()
            cpu_usage, mem_usage = get_cpu_mem_usage()
            pod_to_dep = get_pod_to_deployment()

            for pod, cpu in cpu_usage.items():
                dep = pod_to_dep.get(pod, "unknown")
                desired, available = replica_data.get(dep, (0, 0))
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
            f2.flush()

            time.sleep(INTERVAL)

    env.runner.quit()
    print("Finished run. Data written to:", LOCUST_FILE, "and", REPLICAS_FILE)
