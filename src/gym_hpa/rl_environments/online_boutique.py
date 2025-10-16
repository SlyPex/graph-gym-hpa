import logging
import time
from statistics import mean
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.spaces import Box

# Local imports
from gym_hpa.rl_environments.deployment import (
    get_online_boutique_deployment_list,
)
from gym_hpa.rl_environments.util import (
    save_to_csv,
    get_num_pods,
    calculate_system_distance,
)
from gym_hpa.gnn.graphCreation import (
    build_graph,
    get_traffic_between_services,
    graph_to_data,
)
from gym_hpa.gnn.gnn import flatten_graph_data

# --- Constants ---
# Replication Limits
MIN_REPLICATION = 1
MAX_REPLICATION = 4  # Adjusted to align with a system-wide max of 44 pods
MAX_STEPS = 25

# Action Definitions
ACTION_DO_NOTHING = 0
SCALING_ACTIONS = [-3, -2, -1, 1, 2, 3]  # Scaling actions from -3 to +3, excluding 0

# Deployments
DEPLOYMENTS = [
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
NUM_DEPLOYMENTS = len(DEPLOYMENTS)


class OnlineBoutique(gym.Env):
    """Horizontal Scaling for Online Boutique in Kubernetes - an OpenAI gym environment"""

    metadata = {"render.modes": ["human", "ansi", "array"]}

    def __init__(self, waiting_period=0.3):
        super(OnlineBoutique, self).__init__()

        self.name = "online_boutique_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.waiting_period = waiting_period

        logging.info(f"[Init] Env: {self.name} | Version {self.__version__}")

        self.current_step = 0

        # Action Space: 1 (Do Nothing) + 11 services * 6 scaling actions = 67 total actions
        num_scaling_actions = len(SCALING_ACTIONS)
        self.action_space = spaces.Discrete(1 + NUM_DEPLOYMENTS * num_scaling_actions)

        # Observation Space: Flattened graph data
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32
        )

        self.deploymentList = get_online_boutique_deployment_list(
            MIN_REPLICATION, MAX_REPLICATION
        )
        for d in self.deploymentList:
            d.print_deployment()

        # Episode state
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []
        self.episode_over = False
        self.info = {}

        # Reward calculation state
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.previous_system_distance = 3

        # Timing and logging
        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"

    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Decode the discrete action into (deployment_id, scaling_value)
        deployment_id = None
        scaling_value = 0
        action_description = "Do Nothing"

        if action > ACTION_DO_NOTHING:
            action_index = action - 1  # Make it 0-indexed for calculations
            num_scaling_actions = len(SCALING_ACTIONS)

            deployment_id = action_index // num_scaling_actions
            scaling_id = action_index % num_scaling_actions
            scaling_value = SCALING_ACTIONS[scaling_id]

            action_description = (
                f"Service: {DEPLOYMENTS[deployment_id]}, Scale: {scaling_value:+}"
            )

        self.take_action(deployment_id, scaling_value)

        # Wait for changes to apply in the cluster
        if (
            action != ACTION_DO_NOTHING
            and not self.constraint_min_pod_replicas
            and not self.constraint_max_pod_replicas
        ):
            time.sleep(self.waiting_period)

        # Update observations from Kubernetes
        for d in self.deploymentList:
            d.update_obs_k8s()

        # Get reward and update state
        reward = self.get_reward
        self.total_reward += reward
        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        logging.info(
            f"[Step {self.current_step}] | Action: {action_description} | "
            f"Reward: {reward:.4f} | Total Reward: {self.total_reward:.4f}"
        )

        ob = self.get_state()
        self.info = {"total_reward": self.total_reward}

        # Reset penalty flags for the next step
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step >= MAX_STEPS:
            self.episode_over = True
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(
                self.file_results,
                self.episode_count,
                mean(self.avg_pods) if self.avg_pods else 0,
                mean(self.avg_latency) if self.avg_latency else 0,
                self.total_reward,
                self.execution_time,
            )

        terminated = self.episode_over
        truncated = self.episode_over
        return np.array(ob, dtype=np.float32), reward, terminated, truncated, self.info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        self.deploymentList = get_online_boutique_deployment_list(
            MIN_REPLICATION, MAX_REPLICATION
        )
        self.previous_system_distance = calculate_system_distance(
            self.deploymentList, 0
        )

        initial_state = self.get_state()
        self.info = {}
        return np.array(initial_state, dtype=np.float32), self.info

    def take_action(self, deployment_id, scaling_value):
        self.current_step += 1
        if self.current_step >= MAX_STEPS:
            self.episode_over = True

        if scaling_value == 0:  # Corresponds to ACTION_DO_NOTHING
            pass
        elif scaling_value > 0:  # Add replicas
            self.deploymentList[deployment_id].deploy_pod_replicas(scaling_value, self)
        elif scaling_value < 0:  # Terminate replicas
            self.deploymentList[deployment_id].terminate_pod_replicas(
                abs(scaling_value), self
            )

    @property
    def get_reward(self):
        if self.constraint_max_pod_replicas or self.constraint_min_pod_replicas:
            return -1.0

        current_system_distance = calculate_system_distance(
            self.deploymentList, self.previous_system_distance
        )
        improvement = self.previous_system_distance - current_system_distance

        max_distance = max(self.previous_system_distance, current_system_distance)
        normalized_improvement = improvement / max_distance if max_distance > 0 else 0

        self.previous_system_distance = current_system_distance
        return max(-1.0, min(1.0, normalized_improvement))

    def get_state(self):
        metrics_dict = {}
        for deployment in self.deploymentList:
            if deployment.obs is None:
                deployment.update_obs_k8s()
            metrics_dict[deployment.name] = {
                "cpu_ratio": deployment.obs[0],
                "mem_ratio": deployment.obs[1],
                "pod_count": deployment.obs[2],
                "desired_pod_count": deployment.obs[3],
            }

        traffic_relations = get_traffic_between_services()
        graph = build_graph(metrics_dict, traffic_relations)
        data = graph_to_data(graph)
        flattened_data = flatten_graph_data(data)
        return flattened_data

    def render(self, mode="human", close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
