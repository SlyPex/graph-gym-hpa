from locust import HttpUser, task, between
from locust.env import Environment
from locust.stats import stats_printer, stats_history
import gevent
import time

# 👇 replace this with your actual frontend URL
FRONTEND_URL = "http://localhost:8080"

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)
    host = FRONTEND_URL  # fixes "must specify host" error

    @task
    def index(self):
        self.client.get("/")


if __name__ == "__main__":
    # Create environment and runner
    env = Environment(user_classes=[WebsiteUser])
    env.create_local_runner()

    # Start a greenlet that periodically outputs stats
    gevent.spawn(stats_printer(env.stats))
    gevent.spawn(stats_history, env.runner)

    # Start the test: 20 users, spawn rate 5/sec
    env.runner.start(user_count=20, spawn_rate=5)

    # Run for 30s
    gevent.sleep(30)

    # Stop the test
    env.runner.quit()

    # Print final stats
    print("Final request stats:")
    for s in env.stats.entries.values():
        print(f"{s.method} {s.name} - {s.num_requests} requests, failures={s.num_failures}, avg_response_time={s.avg_response_time:.2f}ms")

