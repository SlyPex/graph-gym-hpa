# Standard Library
import argparse
import logging
import os
import random
import time

# Libraries
import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from typing import Dict, Any

# Local
from gym_hpa.gnn.gnn import AdvancedGNNExtractor
from gym_hpa.paths import RESULTS_DIR
from gym_hpa.rl_environments.online_boutique import OnlineBoutique

# Logging Configuration
logging.basicConfig(
    handlers=[
        logging.FileHandler("run.log", mode="w"),
        logging.StreamHandler(),  # Also print to console
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


# --- Action Decoding Helper ---
# This function makes the logs human-readable by translating the integer action
# from the model into a descriptive string.
def decode_action(action: int) -> str:
    """Decodes a discrete action integer into a human-readable string."""
    if action == 0:
        return "Do Nothing"

    scaling_actions = [-3, -2, -1, 1, 2, 3]
    deployments = [
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
    num_scaling_actions = len(scaling_actions)

    action_index = action - 1
    deployment_id = action_index // num_scaling_actions
    scaling_id = action_index % num_scaling_actions

    service_name = deployments[deployment_id]
    scaling_value = scaling_actions[scaling_id]

    return f"Scale {service_name} by {scaling_value:+}"


# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Training/Test an autoscaling agent for the Online Boutique application"
)
parser.add_argument(
    "--alg",
    choices=["ppo", "recurrent_ppo", "a2c"],
    default="ppo",
    help="The algorithm to use.",
)
parser.add_argument(
    "--k8s", default=False, action="store_true", help="Enable Kubernetes mode."
)
parser.add_argument(
    "--training", default=False, action="store_true", help="Run in training mode."
)
parser.add_argument(
    "--testing", default=False, action="store_true", help="Run in testing/scaling mode."
)
parser.add_argument(
    "--loading", default=False, action="store_true", help="Load a pre-trained model."
)
parser.add_argument(
    "--load_path", default="logs/model/test.zip", help="Path to the model to load."
)
parser.add_argument(
    "--test_path", default="logs/model/test.zip", help="Path to the model to test."
)
parser.add_argument(
    "--steps", type=int, default=500, help="Frequency of saving model checkpoints."
)
parser.add_argument(
    "--total_steps", type=int, default=5000, help="Total number of training steps."
)
parser.add_argument(
    "--scaler_duration",
    type=int,
    default=10,
    help="Duration in minutes to run the continuous scaler.",
)
parser.add_argument(
    "--scaler_interval",
    type=int,
    default=30,
    help="Interval in seconds between scaling decisions.",
)


def get_policy_kwargs() -> Dict[str, Any]:
    """Defines the policy configuration, including the GNN feature extractor and network architecture."""
    return dict(
        features_extractor_class=AdvancedGNNExtractor,
        features_extractor_kwargs={
            "num_nodes": 11,
            "node_feature_dim": 4,
            "num_edges": 14,
            "edge_feature_dim": 1,
            "edge_index": torch.tensor(
                [
                    [9, 9, 9, 9, 9, 9, 9, 0, 8, 8, 8, 8, 8, 8],
                    [0, 1, 2, 8, 6, 5, 3, 1, 2, 4, 5, 6, 1, 10],
                ]
            ),
            "features_dim": 256,
        },
        net_arch=dict(
            pi=[256, 128, 64],  # Policy network (actor)
            vf=[256, 128, 64],  # Value network (critic)
        ),
        activation_fn=torch.nn.Tanh,
    )


def get_model(alg, env, tensorboard_log, policy_kwargs):
    """Creates a new SB3 model based on the specified algorithm."""
    if alg == "ppo":
        return PPO(
            "MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            n_steps=500,
            batch_size=64,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
    elif alg == "recurrent_ppo":
        return RecurrentPPO(
            "MlpLstmPolicy", env=env, verbose=1, tensorboard_log=tensorboard_log
        )
    elif alg == "a2c":
        return A2C("MlpPolicy", env=env, verbose=1, tensorboard_log=tensorboard_log)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def get_load_model(alg, load_path, tensorboard_log):
    """Loads a pre-trained SB3 model."""
    common_args = dict(verbose=1, tensorboard_log=tensorboard_log)
    if alg == "ppo":
        return PPO.load(load_path, **common_args)
    elif alg == "recurrent_ppo":
        return RecurrentPPO.load(load_path, **common_args)
    elif alg == "a2c":
        return A2C.load(load_path, **common_args)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def get_env(k8s):
    """Initializes the Online Boutique gym environment."""
    return OnlineBoutique(k8s=k8s)


def get_tensorboard_log_path(k8s):
    """Generates the path for TensorBoard logs."""
    scenario = "real" if k8s else "simulated"
    use_case = "online_boutique"
    return os.path.join(RESULTS_DIR, use_case, scenario)


def get_run_name(alg, env_name, k8s, total_steps):
    return f"{alg}_env_{env_name}_k8s_{k8s}_totalSteps_{total_steps}"


def train_model(model, total_steps, name, checkpoint_callback):
    model.learn(
        total_timesteps=total_steps,
        tb_log_name=name + "_run",
        callback=checkpoint_callback,
        progress_bar=True,
    )
    model.save(name)


def get_model_or_load(alg, env, tensorboard_log, loading, load_path, policy_kwargs):
    if loading:
        model = get_load_model(alg, load_path, tensorboard_log)
        model.set_env(env)
        return model
    else:
        return get_model(alg, env, tensorboard_log, policy_kwargs)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_continuous_scaler(model, env, duration_minutes, interval_seconds):
    """Runs the model as a continuous autoscaler for a specified duration."""
    logging.info(f"Starting continuous scaler for {duration_minutes} minutes...")
    logging.info(f"Scaling decisions will be made every {interval_seconds} seconds.")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    step_count = 0

    obs, _ = env.reset()

    while time.time() < end_time:
        action, _ = model.predict(obs, deterministic=True)
        action_str = decode_action(action.item())

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        logging.info(f"Step {step_count}: Action='{action_str}', Reward={reward:.3f}")

        for deployment in env.deploymentList:
            logging.info(
                f"  -> {deployment.name}: {deployment.num_pods} pods (desired: {deployment.desired_replicas})"
            )

        if terminated or truncated:
            logging.info("Environment episode finished. Resetting...")
            obs, _ = env.reset()

        time.sleep(interval_seconds)

    logging.info(f"Continuous scaler completed after {step_count} steps.")


def main():
    set_seed(42)
    args = parser.parse_args()
    logging.info(f"Starting with config: {vars(args)}")

    env = get_env(args.k8s)
    logging.info(f"Using environment: {env.name}")

    tensorboard_log = get_tensorboard_log_path(args.k8s)
    os.makedirs(tensorboard_log, exist_ok=True)
    logging.info(f"TensorBoard logs at: {tensorboard_log}")

    run_name = get_run_name(args.alg, env.name, args.k8s, args.total_steps)
    logging.info(f"Run name: {run_name}")

    policy_kwargs = get_policy_kwargs()

    if args.loading:
        logging.info(f"Loading model from: {args.load_path}")
    else:
        logging.info(f"Creating new model: {args.alg}")

    model = get_model_or_load(
        args.alg, env, tensorboard_log, args.loading, args.load_path, policy_kwargs
    )

    if args.training:
        logging.info(f"Training started for {args.total_steps} steps")
        checkpoint_callback = CheckpointCallback(
            save_freq=args.steps,
            save_path=os.path.join("logs", run_name),
            name_prefix=run_name,
        )
        train_model(model, args.total_steps, run_name, checkpoint_callback)
        logging.info("Training completed.")

    if args.testing:
        logging.info(f"Testing model from: {args.test_path}")
        # Create a fresh environment for testing
        test_env = get_env(args.k8s)
        model = get_load_model(args.alg, args.test_path, tensorboard_log)
        model.set_env(test_env)

        run_continuous_scaler(
            model, test_env, args.scaler_duration, args.scaler_interval
        )


if __name__ == "__main__":
    main()
