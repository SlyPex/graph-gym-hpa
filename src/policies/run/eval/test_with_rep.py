from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
import gevent
import time
import csv
from kubernetes import client, config


FRONTEND_URL = "http://localhost:8080"
NAMESPACE = "onlineboutique"
CSV_FILE = "replica_counts.csv"


class WebsiteUser(HttpUser):
    wait_time = between(1, 3)
    host = FRONTEND_URL

    @task
    def index(self):
        self.client.get("/")


def collect_replicas(stop_event, interval=10):
    """
    Periodically collect replica counts and write to CSV.
    """
    # Load kubeconfig (works outside cluster)
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Open CSV file for writing
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "deployment", "desired", "available"])

        while not stop_event.is_set():
            timestamp = time.time()
            deployments = apps_v1.list_namespaced_deployment(namespace=NAMESPACE)
            for deploy in deployments.items:
                writer.writerow(
                    [
                        timestamp,
                        deploy.metadata.name,
                        deploy.spec.replicas,
                        deploy.status.available_replicas or 0,
                    ]
                )
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    # Create environment and runner
    env = Environment(user_classes=[WebsiteUser])
    env.create_local_runner()

    # Start periodic stats printing
    gevent.spawn(stats_printer(env.stats))
    gevent.spawn(stats_history, env.runner)

    # Start replica collector
    stop_event = gevent.event.Event()
    collector_greenlet = gevent.spawn(collect_replicas, stop_event, interval=10)

    # Start the test
    env.runner.start(user_count=20, spawn_rate=5)

    # Run for 60s for now
    gevent.sleep(60)

    # Stop everything
    env.runner.quit()
    stop_event.set()
    collector_greenlet.join()

    print(f"Replica counts saved to {CSV_FILE}")
