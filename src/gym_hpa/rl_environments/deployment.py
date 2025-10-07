import logging
import math
import random
import time
import numpy as np
from kubernetes import client

# Local imports
from gym_hpa.rl_environments.util import fetch_prom

# --- Constants ---
HOST = "http://localhost:8080"
TOKEN = ""

MAX_CPU = 10000
MAX_MEM = 10000
MAX_TRAFFIC = 20000

CPU_WEIGHT = 0.7
MEM_WEIGHT = 0.3

# --- Helper Functions ---

def get_online_boutique_deployment_list(k8s, min_pods, max_pods):
    """Factory function to create a list of DeploymentStatus objects for the Online Boutique app."""
    deployment_configs = [
        ("recommendationservice", 100, 200, 220, 450),
        ("productcatalogservice", 100, 200, 64, 128),
        ("cartservice", 200, 300, 64, 128),
        ("adservice", 200, 300, 180, 300),
        ("paymentservice", 100, 200, 64, 128),
        ("shippingservice", 100, 200, 64, 128),
        ("currencyservice", 100, 200, 64, 128),
        ("redis-cart", 70, 125, 200, 256),
        ("checkoutservice", 100, 200, 64, 128),
        ("frontend", 100, 200, 64, 128),
        ("emailservice", 100, 200, 64, 128),
    ]
    
    deployment_list = []
    for name, cpu_req, cpu_lim, mem_req, mem_lim in deployment_configs:
        deployment_list.append(
            DeploymentStatus(
                k8s=k8s, name=name, namespace="onlineboutique", container_name=name,
                max_pods=max_pods, min_pods=min_pods,
                cpu_request=cpu_req, cpu_limit=cpu_lim,
                mem_request=mem_req, mem_limit=mem_lim
            )
        )
    return deployment_list

def get_max_cpu():
    return MAX_CPU

def get_max_mem():
    return MAX_MEM

def get_max_traffic():
    return MAX_TRAFFIC

def parse_cpu(cpu_str: str) -> float:
    if not cpu_str: return 0.0
    if cpu_str.endswith("m"):
        return float(cpu_str[:-1])
    return float(cpu_str) * 1000

def parse_memory(mem_str: str) -> float:
    if not mem_str: return 0.0
    mem_str = mem_str.lower()
    if mem_str.endswith("ki"):
        return float(mem_str[:-2]) / 1024
    if mem_str.endswith("mi"):
        return float(mem_str[:-2])
    if mem_str.endswith("gi"):
        return float(mem_str[:-2]) * 1024
    return float(mem_str) / (1024 * 1024)

# --- Main Class ---

class DeploymentStatus:
    def __init__(
        self, k8s, name, namespace, container_name, max_pods, min_pods,
        cpu_request, cpu_limit, mem_request, mem_limit, threshold=0.8
    ):
        self.k8s = k8s
        self.name = name
        self.namespace = namespace
        self.container_name = container_name
        self.threshold = threshold
        self.cpu_weight = CPU_WEIGHT
        self.mem_weight = MEM_WEIGHT
        self.max_pods = max_pods
        self.min_pods = min_pods
        self.num_pods = min_pods
        self.num_previous_pods = min_pods
        self.desired_replicas = min_pods
        self.cpu_request = cpu_request
        self.cpu_limit = cpu_limit
        self.mem_request = mem_request
        self.mem_limit = mem_limit
        self.cpu_target = self.threshold
        self.mem_target = self.threshold
        self.cpu_usage = random.randint(1, get_max_cpu())
        self.mem_usage = random.randint(1, get_max_mem())
        self.received_traffic = random.randint(1, get_max_traffic())
        self.transmit_traffic = random.randint(1, get_max_traffic())
        self.latency = 0
        self.pod_names = []
        self.sleep = 0.2
        self.obs = np.array([0.0, 0.0, self.min_pods / self.max_pods, self.min_pods / self.max_pods], dtype=np.float32)

        if self.k8s:
            logging.info(f"[Deployment {self.name}] Connecting to Kubernetes cluster...")
            try:
                k8s_config = client.Configuration()
                k8s_config.verify_ssl = False
                k8s_config.api_key = {"authorization": f"Bearer {TOKEN}"}
                k8s_config.host = HOST
                api_client = client.ApiClient(k8s_config)
                self.v1 = client.CoreV1Api(api_client)
                self.apps_v1 = client.AppsV1Api(api_client)
                deployment = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
                self.num_pods = deployment.spec.replicas
                self.num_previous_pods = deployment.spec.replicas
                self.update_obs_k8s()
            except Exception as e:
                logging.error(f"Failed to connect to Kubernetes or find deployment '{self.name}': {e}")
                self.k8s = False

    def update_obs_k8s(self):
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
            pods = self.v1.list_namespaced_pod(namespace=self.namespace, label_selector=f"app={self.name}")
            self.pod_names = [p.metadata.name for p in pods.items if p.status.phase == "Running"]
            self.num_pods = len(self.pod_names)
        except Exception as e:
            logging.error(f"Error fetching Kubernetes data for '{self.name}': {e}")
            return self.obs

        container_limits = {}
        for c in deployment.spec.template.spec.containers:
            limits = c.resources.limits or {}
            container_limits[c.name] = {
                "cpu": parse_cpu(limits.get("cpu")),
                "mem": parse_memory(limits.get("memory"))
            }

        cpu_ratios, mem_ratios = [], []
        for pod in self.pod_names:
            for c_name, limits in container_limits.items():
                if limits["cpu"] > 0:
                    query = f'sum(irate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod="{pod}", container="{c_name}"}}[2m]))'
                    # *** FIX: Call fetch_prom once and check the result safely ***
                    cpu_results = fetch_prom(query)
                    if cpu_results:
                        cpu_usage_cores = float(cpu_results[0]["value"][1])
                        cpu_ratios.append((cpu_usage_cores * 1000) / limits["cpu"])
                if limits["mem"] > 0:
                    query = f'sum(container_memory_working_set_bytes{{namespace="{self.namespace}", pod="{pod}", container="{c_name}"}})'
                    # *** FIX: Call fetch_prom once and check the result safely ***
                    mem_results = fetch_prom(query)
                    if mem_results:
                        mem_usage_bytes = float(mem_results[0]["value"][1])
                        mem_ratios.append((mem_usage_bytes / (1024*1024)) / limits["mem"])

        self.cpu_ratio = np.mean(cpu_ratios) if cpu_ratios else 0
        self.mem_ratio = np.mean(mem_ratios) if mem_ratios else 0

        self.update_replicas()

        pod_ratio = self.num_pods / self.max_pods if self.max_pods > 0 else 0.0
        desired_ratio = self.desired_replicas / self.max_pods if self.max_pods > 0 else 0.0
        self.obs = np.array([self.cpu_ratio, self.mem_ratio, pod_ratio, desired_ratio], dtype=np.float32)
        return self.obs

    def update_replicas(self):
        if self.num_pods == 0:
             self.desired_replicas = self.min_pods
             return self.desired_replicas
             
        desired_cpu = math.ceil(self.num_pods * (self.cpu_ratio / self.cpu_target)) if self.cpu_target > 0 else self.min_pods
        desired_mem = math.ceil(self.num_pods * (self.mem_ratio / self.mem_target)) if self.mem_target > 0 else self.min_pods
        
        self.desired_replicas = max(desired_cpu, desired_mem)
        self.desired_replicas = max(self.min_pods, min(self.max_pods, self.desired_replicas))
        return self.desired_replicas

    def print_deployment(self):
        logging.info(f"[Deployment] Name: {self.name}, Namespace: {self.namespace}, Pods: {self.num_pods}, Desired: {self.desired_replicas}")

    def update_deployment_replicas(self, new_replicas):
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.name, namespace=self.namespace)
            self.num_previous_pods = deployment.spec.replicas
            deployment.spec.replicas = new_replicas
            self.apps_v1.patch_namespaced_deployment(name=self.name, namespace=self.namespace, body=deployment)
        except Exception as e:
            logging.error(f"Failed to patch deployment '{self.name}': {e}. Retrying in {self.sleep}s...")
            time.sleep(self.sleep)
            self.update_deployment_replicas(new_replicas)

    def deploy_pod_replicas(self, n, env):
        new_replicas = self.num_pods + n
        if new_replicas <= self.max_pods:
            if self.k8s:
                self.update_deployment_replicas(new_replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = new_replicas
        else:
            env.constraint_max_pod_replicas = True

    def terminate_pod_replicas(self, n, env):
        new_replicas = self.num_pods - n
        if new_replicas >= self.min_pods:
            if self.k8s:
                self.update_deployment_replicas(new_replicas)
            else:
                self.num_previous_pods = self.num_pods
                self.num_pods = new_replicas
        else:
            env.constraint_min_pod_replicas = True