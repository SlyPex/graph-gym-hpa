#!/usr/bin/env python3
import csv
import datetime
import logging
import os
import random
import time
import gevent

from faker import Faker
from kubernetes import client, config
from locust import FastHttpUser, TaskSet, between, events, stats

# ----------------------
# Setup
# ----------------------
fake = Faker()
RUN_ID = os.getenv("RUN_ID", "run1")
CSV_FILE = "autoscaling_eval.csv"
NAMESPACE = "onlineboutique"
LOG_INTERVAL = 10  # seconds

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Kubernetes client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()
apps_api = client.AppsV1Api()


# ----------------------
# Locust Tasks
# ----------------------
products = [
    "0PUK6V6EV0",
    "1YMWWN1N4O",
    "2ZYFJ3GM2N",
    "66VCHSJNUP",
    "6E92ZMYYFZ",
    "9SIQT8TOJO",
    "L9ECAV7KIM",
    "LS4PSXUNUM",
    "OLJCESPC7Z",
]


def index(l):
    l.client.get("/")


def setCurrency(l):
    currencies = ["EUR", "USD", "JPY", "CAD", "GBP", "TRY"]
    l.client.post("/setCurrency", {"currency_code": random.choice(currencies)})


def browseProduct(l):
    l.client.get("/product/" + random.choice(products))


def viewCart(l):
    l.client.get("/cart")


def addToCart(l):
    product = random.choice(products)
    l.client.get("/product/" + product)
    l.client.post("/cart", {"product_id": product, "quantity": random.randint(1, 10)})


def checkout(l):
    addToCart(l)
    current_year = datetime.datetime.now().year + 1
    l.client.post(
        "/cart/checkout",
        {
            "email": fake.email(),
            "street_address": fake.street_address(),
            "zip_code": fake.zipcode(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "country": fake.country(),
            "credit_card_number": fake.credit_card_number(card_type="visa"),
            "credit_card_expiration_month": random.randint(1, 12),
            "credit_card_expiration_year": random.randint(
                current_year, current_year + 5
            ),
            "credit_card_cvv": f"{random.randint(100, 999)}",
        },
    )


class UserBehavior(TaskSet):
    def on_start(self):
        index(self)

    tasks = {
        index: 1,
        setCurrency: 2,
        browseProduct: 10,
        addToCart: 2,
        viewCart: 3,
        checkout: 1,
    }


class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)
    host = os.getenv("TARGET_HOST", "http://frontend:80")


# ----------------------
# Metrics + CSV logging
# ----------------------
def get_replica_counts():
    """Fetch replica counts for all deployments in namespace."""
    replicas = {}
    try:
        deployments = apps_api.list_namespaced_deployment(NAMESPACE)
        for dep in deployments.items:
            replicas[dep.metadata.name] = {
                "desired": dep.spec.replicas,
                "available": dep.status.available_replicas or 0,
            }
    except Exception as e:
        logging.error(f"Error fetching replicas: {e}")
    return replicas


def ensure_csv_header():
    """Ensure CSV file has header only once."""
    header = [
        "timestamp",
        "run_id",
        "strategy",
        "load_users",
        "spawn_rate",
        "rps",
        "total_requests",
        "failures",
        "avg_latency_ms",
        "p95_latency_ms",
        "endpoint",
        "deployment",
        "desired_replicas",
        "available_replicas",
    ]
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def log_results(row: dict):
    """Append row to CSV."""
    ensure_csv_header()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row.get(col, "")
                for col in [
                    "timestamp",
                    "run_id",
                    "strategy",
                    "load_users",
                    "spawn_rate",
                    "rps",
                    "total_requests",
                    "failures",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "endpoint",
                    "deployment",
                    "desired_replicas",
                    "available_replicas",
                ]
            ]
        )


def aggregate_and_log():
    """Aggregate Locust stats + replica counts, log to CSV."""
    replicas = get_replica_counts()
    for stat in stats.get_current().entries.values():
        for dep, rep in replicas.items():
            row = {
                "timestamp": datetime.datetime.now().isoformat(),
                "run_id": RUN_ID,
                "strategy": os.getenv("STRATEGY", "hpa-cpu-70"),
                "load_users": WebsiteUser.user_count
                if hasattr(WebsiteUser, "user_count")
                else "",
                "spawn_rate": os.getenv("SPAWN_RATE", ""),
                "rps": stat.current_rps,
                "total_requests": stat.num_requests,
                "failures": stat.num_failures,
                "avg_latency_ms": stat.avg_response_time,
                "p95_latency_ms": stat.get_response_time_percentile(0.95),
                "endpoint": stat.name,
                "deployment": dep,
                "desired_replicas": rep["desired"],
                "available_replicas": rep["available"],
            }
            log_results(row)
    logging.info(f"Snapshot logged with {len(replicas)} deployments.")


@events.test_start.add_listener
def on_test_start(environment, **kw):
    """Start periodic logging worker."""

    def _log_worker():
        while True:
            aggregate_and_log()
            time.sleep(LOG_INTERVAL)

    gevent.spawn(_log_worker)
