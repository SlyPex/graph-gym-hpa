import csv
import time
import requests

# --- Constants ---
LATENCY_SLO = 300
PROMETHEUS_URL = "http://localhost:31090"
QUERY_LATENCY = "max(locust_requests_current_response_time_percentile_95)"

# --- Functions ---


def save_to_csv(file_name, episode, avg_pods, avg_latency, reward, execution_time):
    """Appends a summary of a completed episode to a CSV file."""
    try:
        with open(file_name, "r") as f:
            has_header = bool(f.read(1024))
    except FileNotFoundError:
        has_header = False

    with open(file_name, "a", newline="") as file:
        fields = ["episode", "avg_pods", "avg_latency", "reward", "execution_time"]
        writer = csv.DictWriter(file, fieldnames=fields)
        if not has_header:
            writer.writeheader()
        writer.writerow(
            {
                "episode": episode,
                "avg_pods": f"{avg_pods:.2f}",
                "avg_latency": f"{avg_latency:.4f}",
                "reward": f"{reward:.2f}",
                "execution_time": f"{execution_time:.2f}",
            }
        )


def get_num_pods(deployment_list):
    """Calculates the total number of pods across all deployments."""
    return sum(d.num_pods for d in deployment_list)


def fetch_prom(query, retry_delay=0.2):
    """Fetches data from Prometheus with a simple retry mechanism."""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "success":
            print(
                f"Prometheus query failed with status '{data['status']}': {data.get('error', 'No error message')}"
            )
            return []
        return data["data"]["result"]
    except requests.exceptions.RequestException as e:
        print(f"Request to Prometheus failed: {e}. Retrying in {retry_delay}s...")
        time.sleep(retry_delay)
        # *** FIX: Added 'return' to pass the result of the retry call back up ***
        return fetch_prom(query, retry_delay)


def calculate_cost_distance(deployment_list):
    """Calculates a normalized cost distance based on the number of running pods."""
    MAX_TOTAL_PODS = 44
    total_pods = get_num_pods(deployment_list)
    return min(total_pods / MAX_TOTAL_PODS, 1.0)


def calculate_latency_distance(global_latency):
    """Calculates a normalized distance from the latency SLO."""
    if global_latency <= LATENCY_SLO:
        return 0.0
    else:
        violation_amount = global_latency - LATENCY_SLO
        normalized_latency_distance = min(violation_amount / LATENCY_SLO, 1.0)
        return normalized_latency_distance


def calculate_system_distance(
    deployment_list, previous_system_distance, w_latency=2.0, w_cost=1.0
):
    """Calculates a weighted distance metric for the system state."""
    latency_result = fetch_prom(QUERY_LATENCY)
    current_latency = (
        float(latency_result[0]["value"][1]) if latency_result else LATENCY_SLO
    )

    latency_dist = calculate_latency_distance(current_latency)
    cost_dist = calculate_cost_distance(deployment_list)

    current_system_distance = (w_latency * latency_dist) + (w_cost * cost_dist)
    return current_system_distance
