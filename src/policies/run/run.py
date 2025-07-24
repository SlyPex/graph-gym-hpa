# Standard Library
import argparse
import logging
import os

# Libraries
import torch
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import numpy as np


# Local
from gym_hpa.rl_environments.redis import Redis
from gym_hpa.rl_environments.online_boutique import OnlineBoutique
from gym_hpa.gnn.gnn import CustomGNNExtractor
from policies.util.util import test_model
from gym_hpa.paths import RESULTS_DIR



# Logging
# logger = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler("runn.log", mode="w"),
        logging.StreamHandler()  # This will also print to console
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)



# Parsing arguments
parser = argparse.ArgumentParser(description="Run ILP!")

parser.add_argument("--alg", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo" , help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]')

parser.add_argument("--k8s", default=False, action="store_true", help="K8s mode")
parser.add_argument(
    "--use_case", default="redis", help='Apps: ["redis", "online_boutique"]'
)
parser.add_argument("--goal", default="cost", help='Reward Goal: ["cost", "latency"]')

parser.add_argument(
    "--training", default=False, action="store_true", help="Training mode"
)
parser.add_argument(
    "--testing", default=False, action="store_true", help="Testing mode"
)
parser.add_argument(
    "--loading", default=False, action="store_true", help="Loading mode"
)
parser.add_argument(
    "--load_path",
    default="logs/model/test.zip",
    help="Loading path, ex: logs/model/test.zip",
)
parser.add_argument(
    "--test_path",
    default="logs/model/test.zip",
    help="Testing path, ex: logs/model/test.zip",
)

parser.add_argument("--steps", type= int ,  default=500, help="The steps for saving.")
parser.add_argument("--total_steps",type = int , default=5000, help="The total number of steps.")




def get_policy_kwargs():
    return dict(
        features_extractor_class=CustomGNNExtractor,
        features_extractor_kwargs={
            "num_nodes": 11,
            "node_feature_dim": 4,
            "num_edges": 15,
            "edge_feature_dim": 1,
            "edge_index": torch.tensor(
            [
                [9, 9, 9, 9, 9, 9, 9, 0, 2, 8, 8, 8, 8, 8, 8],
                [0, 1, 2, 8, 6, 5, 3, 1, 7, 2, 4, 5, 6, 1, 10],
            ]),
            "features_dim": 24,
        }
    )




def get_model(alg, env, tensorboard_log , policy_kwargs):
    common_args = dict(env=env, verbose=1, tensorboard_log=tensorboard_log)
   
    ## the batch size was fixed at 125 to clean the output , must update later
    ## n_steps ????
    if alg == "ppo":
        return PPO("MlpPolicy", n_steps=500, batch_size=125, policy_kwargs=policy_kwargs, **common_args)
    elif alg == "recurrent_ppo":
        return RecurrentPPO("MlpLstmPolicy", **common_args)
    elif alg == "a2c":
        return A2C("MlpPolicy", **common_args)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def get_load_model(alg, tensorboard_log, load_path ):
    common_args = dict( verbose=1, tensorboard_log=tensorboard_log)
   
    ## the batch size was fixed at 125 to clean the output , must update later
    ## n_steps ????
    ## change this
    if alg == "ppo":
        return PPO.load(load_path,  **common_args)
    elif alg == "recurrent_ppo":
        return RecurrentPPO.load(load_path, **common_args)
    elif alg == "a2c":
        return A2C.load(load_path, **common_args)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

def get_env(use_case, k8s, goal):
    env = 0
    if use_case == "redis":
        env = Redis(k8s=k8s, goal_reward=goal)
    elif use_case == "online_boutique":
        env = OnlineBoutique(k8s=k8s, goal_reward=goal)
    else:
        logging.error("Invalid use_case!")
        raise ValueError("Invalid use_case!")

    return env

def get_tensorboard_log_path(use_case, k8s, goal):
    scenario = "real" if k8s else "simulated"
    return os.path.join(RESULTS_DIR, use_case, scenario, goal)

def get_run_name(alg, env_name, goal, k8s, total_steps):
    return f"{alg}_env_{env_name}_goal_{goal}_k8s_{k8s}_totalSteps_{total_steps}"

def train_model(model, total_steps, name, checkpoint_callback):
    model.learn(
        total_timesteps=total_steps,
        tb_log_name=name + "_run",
        callback=checkpoint_callback,
        progress_bar=True
    )
    model.save(name)

def test_model_wrapper(model, env, name):
    test_model(
        model,
        env,
        n_episodes=100,
        n_steps=110,
        smoothing_window=5,
        fig_name=name + "_test_reward.png",
    )
def get_model_or_load(alg, env, tensorboard_log, loading, load_path, policy_kwargs):
    if loading:
        model = get_load_model(alg, tensorboard_log, load_path)
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
        
def main():
    set_seed(42)
    args = parser.parse_args()
    logging.info(f"Starting with config: {vars(args)}")

    env = get_env(args.use_case, args.k8s, args.goal)
    
    logging.info(f"Using environment: {env.name}")

    tensorboard_log = get_tensorboard_log_path(args.use_case, args.k8s, args.goal)
    logging.info(f"TensorBoard logs at: {tensorboard_log}")

    run_name = get_run_name(args.alg, env.name, args.goal, args.k8s, args.total_steps)
    logging.info(f"Run name: {run_name}")

    policy_kwargs = get_policy_kwargs()
    if args.loading:
        logging.info(f"Loading model from: {args.load_path}")
    else:
        logging.info(f"Creating new model: {args.alg}")

    model = get_model_or_load(
        args.alg,
        env,
        tensorboard_log,
        args.loading,
        args.load_path,
        policy_kwargs
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.steps,
        save_path=os.path.join("logs", run_name),
        name_prefix=run_name
    )
    print(checkpoint_callback)
    if args.training:
        logging.info(f"Training started for {args.total_steps} steps")


    if args.training:
        train_model(model, args.total_steps, run_name, checkpoint_callback)
        logging.info("Training completed.")

    if args.testing:
        logging.info(f"Testing model loaded from: {args.test_path}")


    if args.testing:
        model = get_load_model(args.alg, tensorboard_log, args.test_path)

        logging.info("Testing completed.")


if __name__ == "__main__":
    main()
