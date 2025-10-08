import csv
import time
from datetime import datetime
import random
from faker import Faker
import datetime as dt

from locust import FastHttpUser, TaskSet, between
from locust.env import Environment
from kubernetes import client, config

# ---------------- Locust workload (from your script) ----------------
fake = Faker()
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
    l.client.post(
        "/setCurrency",
        {"currency_code": random.choice(["EUR", "USD", "JPY", "CAD", "GBP", "TRY"])},
    )


def browseProduct(l):
    l.client.get("/product/" + random.choice(products))


def viewCart(l):
    l.client.get("/cart")


def addToCart(l):
    product = random.choice(products)
    l.client.get("/product/" + product)
    l.client.post("/cart", {"product_id": product, "quantity": random.randint(1, 10)})


def empty_cart(l):
    l.client.post("/cart/empty")


def checkout(l):
    addToCart(l)
    current_year = dt.datetime.now().year + 1
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
                current_year, current_year + 10
            ),
            "credit_card_cvv": f"{random.randint(100, 999)}",
        },
    )


def logout(l):
    l.client.get("/logout")


class UserBehavior(TaskSet):
    tasks = {
        index: 1,
        setCurrency: 2,
        browseProduct: 10,
        addToCart: 2,
        viewCart: 3,
        checkout: 1,
    }

    def on_start(self):
        index(self)


class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 10)
    host = "http://localhost:8080"


# ---------------- K8s setup ----------------
config.load_kube_config()
apps_v1 = client.AppsV1Api()

# ---------------- Logging setup ----------------
OUT_FILE = "autoscaling_eval.csv"
FIELDNAMES = [
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

RUN_ID = "run1"
STRATEGY = "hpa-cpu-70"
LOAD_USERS = 100
SPAWN_RATE = 10
DURATION = 30  # 5 mins
INTERVAL = 10  # sample interval

NAMESPACE = "onlineboutique"


# ---------------- Helper: fetch replicas ----------------
def get_replicas():
    deployments = apps_v1.list_namespaced_deployment(NAMESPACE).items
    replica_data = {}
    for dep in deployments:
        name = dep.metadata.name
        desired = dep.spec.replicas or 0
        available = dep.status.available_replicas or 0
        replica_data[name] = (desired, available)
    return replica_data


# ---------------- Main ----------------
if __name__ == "__main__":
    # Locust environment
    env = Environment(user_classes=[WebsiteUser])
    env.create_local_runner()
    env.runner.start(LOAD_USERS, spawn_rate=SPAWN_RATE)

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        start_time = time.time()
        while time.time() - start_time < DURATION:
            timestamp = datetime.utcnow().isoformat()

            # Get detailed Locust stats per endpoint
            for (method, name), stats in env.runner.stats.entries.items():
                rps = stats.current_rps
                total_reqs = stats.num_requests
                fails = stats.num_failures
                avg_latency = stats.avg_response_time or 0
                p95_latency = stats.get_response_time_percentile(0.95) or 0

                # Get replica data
                replica_data = get_replicas()

                for dep, (desired, available) in replica_data.items():
                    row = {
                        "timestamp": timestamp,
                        "run_id": RUN_ID,
                        "strategy": STRATEGY,
                        "load_users": LOAD_USERS,
                        "spawn_rate": SPAWN_RATE,
                        "rps": rps,
                        "total_requests": total_reqs,
                        "failures": fails,
                        "avg_latency_ms": round(avg_latency, 2),
                        "p95_latency_ms": round(p95_latency, 2),
                        "endpoint": f"{method} {name}",
                        "deployment": dep,
                        "desired_replicas": desired,
                        "available_replicas": available,
                    }
                    writer.writerow(row)
                    f.flush()

            time.sleep(INTERVAL)

    env.runner.quit()
    print("Finished run. Data written to", OUT_FILE)
