import logging
import math
import random
import time
import requests
from kubernetes import client , config
import numpy as np
# Constants
MAX_CPU = 10000  # cpu in m
MAX_MEM = 10000  # memory in MiB
MAX_TRAFFIC = 20000  # MAX Number of requests (in Kbit/s)

CPU_WEIGHT = 0.7
MEM_WEIGHT = 0.3

# port-forward in k8s cluster
PROMETHEUS_URL = "http://localhost:31090/"

# Endpoint of your Kube cluster: kube proxy enabled
HOST = "http://localhost:8080"

# TODO: Add the TOKEN from your cluster!
TOKEN = ""



def get_redis_deployment_list(k8s, min, max):
    deployment_list = [
        DeploymentStatus(
            k8s,
            "redis-leader",
            "redis",
            "leader",
            "docker.io/redis:6.0.5",
            max,
            min,
            250,
            500,
            250,
            500,
        ),
        DeploymentStatus(
            k8s,
            "redis-follower",
            "redis",
            "follower",
            "gcr.io/google_samples/gb-redis-follower:v2",
            max,
            min,
            250,
            500,
            250,
            500,
        ),
    ]
    return deployment_list


def get_online_boutique_deployment_list(k8s, min, max):
    deployment_list = [
        # 1
        DeploymentStatus(
            k8s,
            "recommendationservice",
            "onlineboutique",
            "recommendationservice",
            "quay.io/signalfuse/microservices-demo-recommendationservice:433c23881a",
            max,
            min,
            100,
            200,
            220,
            450,
        ),
        # 2
        DeploymentStatus(
            k8s,
            "productcatalogservice",
            "onlineboutique",
            "productcatalogservice",
            "quay.io/signalfuse/microservices-demo-productcatalogservice:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 3
        DeploymentStatus(
            k8s,
            "cartservice",
            "onlineboutique",
            "cartservice",
            "quay.io/signalfuse/microservices-demo-cartservice:433c23881a",
            max,
            min,
            200,
            300,
            64,
            128,
        ),
        # 4
        DeploymentStatus(
            k8s,
            "adservice",
            "onlineboutique",
            "adservice",
            "quay.io/signalfuse/microservices-demo-adservice:433c23881a",
            max,
            min,
            200,
            300,
            180,
            300,
        ),
        # 5
        DeploymentStatus(
            k8s,
            "paymentservice",
            "onlineboutique",
            "paymentservice",
            "quay.io/signalfuse/microservices-demo-paymentservice:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 6
        DeploymentStatus(
            k8s,
            "shippingservice",
            "onlineboutique",
            "shippingservice",
            "quay.io/signalfuse/microservices-demo-shippingservice:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 7
        DeploymentStatus(
            k8s,
            "currencyservice",
            "onlineboutique",
            "currencyservice",
            "quay.io/signalfuse/microservices-demo-currencyservice:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 8
        DeploymentStatus(
            k8s,
            "redis-cart",
            "onlineboutique",
            "redis-cart",
            "redis:alpine",
            max,
            min,
            70,
            125,
            200,
            256,
        ),
        # 9
        DeploymentStatus(
            k8s,
            "checkoutservice",
            "onlineboutique",
            "checkoutservice",
            "quay.io/signalfuse/microservices-demo-checkoutservice:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 10
        DeploymentStatus(
            k8s,
            "frontend",
            "onlineboutique",
            "frontend",
            "quay.io/signalfuse/microservices-demo-frontend:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
        # 11
        DeploymentStatus(
            k8s,
            "emailservice",
            "onlineboutique",
            "emailservice",
            "quay.io/signalfuse/microservices-demo-frontend:433c23881a",
            max,
            min,
            100,
            200,
            64,
            128,
        ),
    ]
    return deployment_list


def get_max_cpu():
    return MAX_CPU


def get_max_mem():
    return MAX_MEM


def get_max_traffic():
    return MAX_TRAFFIC


def convert_to_milli_cpu(value):
    new_value = int(value[:-1])
    if value[-1] == "n":
        new_value = int(value[:-1])
        new_value = int(new_value / 1000000)

    return new_value


def change_usage(min, max, max_threshold):
    if max > max_threshold:
        max = max_threshold

    if min < 0:
        min = 0

    return random.randint(min, max)


def convert_to_mega_memory(value):
    last_two = value[-2:]
    new_value = 0

    if last_two == "Ki":
        size = len(value)
        # Slice string to remove last 2 characters
        new_value = int(value[: size - 2])
        new_value = int(new_value / 1000)

    return new_value

def parse_cpu(cpu_str: str) -> float:
    """Convert Kubernetes CPU string (e.g. '500m', '2') to millicores."""
    if cpu_str.endswith("m"):
        return float(cpu_str[:-1])  # already in millicores
    return float(cpu_str) * 1000   # assume in cores → convert to millicores


def parse_memory(mem_str: str) -> float:
    """Convert Kubernetes memory string (e.g. '512Mi', '2Gi') to Mi."""
    mem_str = mem_str.lower()
    if mem_str.endswith("ki"):
        return float(mem_str[:-2]) / 1024
    elif mem_str.endswith("mi"):
        return float(mem_str[:-2])
    elif mem_str.endswith("gi"):
        return float(mem_str[:-2]) * 1024
    elif mem_str.endswith("ti"):
        return float(mem_str[:-2]) * 1024 * 1024
    elif mem_str.endswith("pi"):
        return float(mem_str[:-2]) * 1024 * 1024 * 1024
    elif mem_str.endswith("ei"):
        return float(mem_str[:-2]) * 1024 * 1024 * 1024 * 1024
    else:
        # bytes → convert to Mi
        return float(mem_str) / (1024 * 1024)
class DeploymentStatus:  # Deployment Status (Workload)
    def __init__(
        self,
        k8s,
        name,
        namespace,
        container_name,
        container_image,
        max_pods,
        min_pods,
        cpu_request,
        cpu_limit,
        mem_request,
        mem_limit,
        threshold=0.75,
    ):
        self.name = name
        # namespace
        self.namespace = namespace
        # container_name
        self.container_name = container_name
        # container image
        self.container_image = container_image

        # CPU & MEM threshold
        self.threshold = threshold
        # CPU weight for replica calculation
        self.cpu_weight = CPU_WEIGHT
        # MEM weight for replica calculation
        self.mem_weight = MEM_WEIGHT

        # Pod Names
        self.pod_names = ["pod-1"]
        # MAX Number of Pods
        self.max_pods = max_pods
        # MIN Number of Pods
        self.min_pods = min_pods
        # Number of Pods
        self.num_pods = 1  # Initialize as 1
        # Number of Pods in previous step
        self.num_previous_pods = 1  # Initialize as 1
        # Number of desired replicas
        self.desired_replicas = 1

        # CPU request (in m)
        self.cpu_request = cpu_request
        # CPU limit (in m)
        self.cpu_limit = cpu_limit

        # MEM request (in MiB)
        self.mem_request = mem_request
        # MEM limit (in MiB)
        self.mem_limit = mem_limit

        # CPU Target (in m)
        self.cpu_target = int(self.threshold * self.cpu_request)

        # MEM Target (in MiB)
        self.mem_target = int(self.threshold * self.mem_request)

        self.MAX_CPU = MAX_CPU  # cpu in m
        self.MAX_MEM = MAX_MEM  # memory in MiB
        self.MAX_TRAFFIC = MAX_TRAFFIC  # MAX Number of requests

        # Get dataset
        # self.version = 'v1'
        # self.df = pd.read_csv(
        #     "../../datasets/real/" + self.namespace + "/" + self.version +
        #     "/" + self.namespace + '_' + self.name + '.csv')

        # CPU Usage Aggregated (in m)
        self.cpu_usage = random.randint(1, get_max_cpu())  # sample['cpu'].values[0]

        # MEM Usage Aggregated (in MiB)
        self.mem_usage = random.randint(1, get_max_mem())  # sample['mem'].values[0]

        # Current Requests
        self.received_traffic = random.randint(
            1, get_max_traffic()
        )  # sample['traffic_in'].values[0]
        self.transmit_traffic = random.randint(
            1, get_max_traffic()
        )  # sample['traffic_out'].values[0]

        # Throughput PING INLINE
        # self.ping = 0

        # K8s enabled?
        self.k8s = k8s

        # csv file
        self.csv = self.namespace + "_" + self.name + ".csv"

        # time between API calls if failure happens
        self.sleep = 0.2

        # App. Latency
        self.latency = 0
        
        self.obs = None
        if self.k8s:  # Real env: consider a k8s cluster    
            logging.info("[Deployment] Consider a real k8s cluster ... ")
            # out of cluster!
            # config.load_kube_config()

            # In cluster config!
            # config.load_incluster_config()

            # token for VWall cluster
            self.token = TOKEN

            # Create a configuration object
            self.config = client.Configuration()
            self.config.verify_ssl = False
            self.config.api_key = {"authorization": "Bearer " + self.token}

            # Specify the endpoint of your Kube cluster: kube proxy enabled
            self.config.host = HOST

            # Create a ApiClient with our config
            self.client = client.ApiClient(self.config)

            # v1 api
            self.v1 = client.CoreV1Api(self.client)
            # apps v1 api
            self.apps_v1 = client.AppsV1Api(self.client)

            # metrics api
            # self.metrics_api = client.CustomObjectsApi(self.client)
            # Get deployment object
            self.deployment_object = self.apps_v1.read_namespaced_deployment(
                name=self.name, namespace=self.namespace
            )
                # Get the initial replica count and store it
            initial_deployment = self.apps_v1.read_namespaced_deployment(
                name=self.name, namespace=self.namespace
            )
            self.num_previous_pods = initial_deployment.spec.replicas
            # Update number of Pods
            self.num_pods = self.deployment_object.spec.replicas
            self.num_previous_pods = self.deployment_object.spec.replicas

            # update obs
            self.update_obs_k8s()

        # else: # Simulation Environment
        # Update Desired replicas


    def get_resource_limits(self):
        """
        Fetch CPU & memory limits for this deployment via the K8s API.
        Returns:
            (cpu_limit, mem_limit, total_cpu_limit, total_mem_limit)
            - cpu_limit, mem_limit: limits for the main app container only
            - total_cpu_limit, total_mem_limit: sum of all container limits in the pod (including sidecars)
        """
        cpu_limit = None
        mem_limit = None
        total_cpu_limit = 0
        total_mem_limit = 0

        deployment = self.apps_v1.read_namespaced_deployment(
            name=self.name, namespace=self.namespace
        )

        for container in deployment.spec.template.spec.containers:
            limits = container.resources.limits or {}

            # App-only: pick the one matching the deployment name
            if container.name == self.name:
                if "cpu" in limits:
                    cpu_limit = parse_cpu(limits["cpu"])
                if "memory" in limits:
                    mem_limit = parse_memory(limits["memory"])

            # Total: sum all containers
            if "cpu" in limits:
                total_cpu_limit += parse_cpu(limits["cpu"])
            if "memory" in limits:
                total_mem_limit += parse_memory(limits["memory"])

        return total_cpu_limit or None, total_mem_limit or None



    def update_obs_k8s(self):
        """
        Observes the current state (pods, CPU/Mem ratios) and calculates the
        desired replica count in a single pass.
        """
        try:
            self.deployment_object = self.apps_v1.read_namespaced_deployment(
                name=self.name, namespace=self.namespace
            )
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"app={self.name}"
            )
            self.pod_names = [p.metadata.name for p in pods.items if p.status.phase == "Running"]
            self.num_pods = len(self.pod_names)
        except Exception as e:
            print(f"Error fetching Kubernetes data for '{self.name}': {e}")
            return self.obs # Return last known observation on failure

        # Collect container-level resource limits
        container_limits = {}
        for container in self.deployment_object.spec.template.spec.containers:
            cpu_lim = parse_cpu(container.resources.limits.get("cpu"))
            mem_lim = parse_memory(container.resources.limits.get("memory"))
            container_limits[container.name] = {"cpu": cpu_lim, "mem": mem_lim}

        # Collect container-level usage and compute ratios from Prometheus
        cpu_ratios = []
        mem_ratios = []
        for pod in self.pod_names:
            for container_name, limits in container_limits.items():
                if limits["cpu"] > 0:
                    query_cpu = f'sum(irate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod="{pod}", container="{container_name}"}}[2m]))'
                    results_cpu = self.fetch_prom(query_cpu)
                    if results_cpu:
                        cpu_usage_millicores = float(results_cpu[0]["value"][1]) * 1000
                        cpu_ratios.append(cpu_usage_millicores / limits["cpu"])

                if limits["mem"] > 0:
                    query_mem = f'sum(container_memory_working_set_bytes{{namespace="{self.namespace}", pod="{pod}", container="{container_name}"}})'
                    results_mem = self.fetch_prom(query_mem)
                    if results_mem:
                        mem_usage_mib = float(results_mem[0]["value"][1]) / (1024 * 1024)
                        mem_ratios.append(mem_usage_mib / limits["mem"])

        # Average and clamp the utilization ratios
        self.cpu_ratio = min(max(np.mean(cpu_ratios) if cpu_ratios else 0, 0.0), 1.0)
        self.mem_ratio = min(max(np.mean(mem_ratios) if mem_ratios else 0, 0.0), 1.0)

        # Calculate the desired number of replicas based on the new ratios
        self.update_replicas()

        # Create the final observation array based on CURRENT and DESIRED state
        current_pod_ratio = self.num_pods / self.max_pods if self.max_pods > 0 else 0.0
        desired_pod_ratio = self.desired_replicas / self.max_pods if self.max_pods > 0 else 0.0
        self.obs = np.array([self.cpu_ratio, self.mem_ratio, current_pod_ratio, desired_pod_ratio], dtype=np.float32)

        # Debug print
        #print(f"[Deployment: {self.name}] Pods={self.num_pods}, Desired={self.desired_replicas}")
        #print(f"  CPU Ratio={self.cpu_ratio:.2f}, Mem Ratio={self.mem_ratio:.2f} -> OBS={self.obs}")

        return self.obs







    def update_replicas(self):
        """
        Calculates the desired number of replicas based on the current average
        resource utilization ratio versus a target utilization ratio.
        """
        # Ensure target values are not zero to avoid division errors
        #if not hasattr(self, 'cpu_target') or self.cpu_target <= 0:
        self.cpu_target = 0.8  # Default to 80% if not set

        # if not hasattr(self, 'mem_target') or self.mem_target <= 0:
        self.mem_target = 0.8  # Default to 80% if not set

        # Calculate desired replicas for each metric based on target utilization
        # Formula: desired = current * ( current_ratio / target_ratio )
        desired_replicas_cpu = math.ceil(
            self.num_pods * (self.cpu_ratio / self.cpu_target)
        )
        desired_replicas_mem = math.ceil(
            self.num_pods * (self.mem_ratio / self.mem_target)
        )

        # In HPA logic, you typically scale up based on whichever metric needs it most.
        # Taking the maximum is a common and effective strategy.
        self.desired_replicas = max(desired_replicas_cpu, desired_replicas_mem)

        # --- Clamping the result to min/max boundaries ---

        # Ensure a minimum of 1 replica (or self.min_pods if defined)
        min_pods = getattr(self, 'min_pods', 1)
        if self.desired_replicas < min_pods:
            self.desired_replicas = min_pods

        # Ensure the number of replicas does not exceed the maximum
        if hasattr(self, 'max_pods') and self.desired_replicas > self.max_pods:
            self.desired_replicas = self.max_pods

        return self.desired_replicas

    def fetch_prom(self, query):
        try:
            response = requests.get(
                PROMETHEUS_URL + "/api/v1/query", params={"query": query}
            )

        except requests.exceptions.RequestException as e:
            print(e)
            print("Retrying in {}...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        if response.json()["status"] != "success":
            print("Error processing the request: " + response.json()["status"])
            print("The Error is: " + response.json()["error"])
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.fetch_prom(query)

        result = response.json()["data"]["result"]
        return result

    def print_deployment(self):
        logging.info("[Deployment] Name: " + str(self.name))
        logging.info("[Deployment] Namespace: " + str(self.namespace))
        logging.info("[Deployment] Number of pods: " + str(self.num_pods))
        logging.info("[Deployment] Desired Replicas: " + str(self.desired_replicas))
        logging.info("[Deployment] Pod Names: " + str(self.pod_names))
        logging.info("[Deployment] MAX Pods: " + str(self.max_pods))
        logging.info("[Deployment] MIN Pods: " + str(self.min_pods))
        logging.info("[Deployment] CPU Usage (in m): " + str(self.cpu_usage))
        logging.info("[Deployment] MEM Usage (in Mi): " + str(self.mem_usage))
        logging.info(
            "[Deployment] Received traffic (in Kbit/s): " + str(self.received_traffic)
        )
        logging.info(
            "[Deployment] Transmit traffic (in Kbit/s): " + str(self.transmit_traffic)
        )
        logging.info("[Deployment] latency (in ms): " + str(self.latency))

    def update_deployment(self, new_replicas):
        # Get deployment object
        self.deployment_object = self.apps_v1.read_namespaced_deployment(
            name=self.name, namespace=self.namespace
        )
        # logging.info(self.deployment_object)

        # Update previous number of pods
        self.num_previous_pods = self.deployment_object.spec.replicas

        # Update replicas
        self.deployment_object.spec.replicas = new_replicas

        # try to patch the deployment
        self.patch_deployment(new_replicas)

    def patch_deployment(self, new_replicas):
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=self.name, namespace=self.namespace, body=self.deployment_object
            )
        except Exception as e:
            print(e)
            print("Retrying in {}s...".format(self.sleep))
            time.sleep(self.sleep)
            return self.update_deployment(new_replicas)

    def deploy_pod_replicas(self, n, env):
        # Deploy pods if possible
        replicas = self.num_pods + n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if replicas <= self.max_pods:
            # logging.info("[Take Action] Add {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            # logging.info("Constraint: MAX Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_max_pod_replicas = True

    def terminate_pod_replicas(self, n, env):
        # Terminate pods if possible
        replicas = self.num_pods - n

        # logging.info("Deployment name: " + str(self.name))
        # logging.info("Current replicas: " + str(self.num_pods))
        # logging.info("New replicas: " + str(replicas))

        if replicas >= self.min_pods:
            # logging.info("[Take Action] Terminate {} Replicas".format(str(n)))
            if self.k8s:  # patch deployment on k8s cluster
                self.update_deployment(replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = replicas
            return
        else:
            # logging.info("Constraint: MIN Pod Replicas! Desired replicas: " + str(replicas))
            env.constraint_min_pod_replicas = True
