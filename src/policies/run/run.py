import logging
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
import torch
from gym_hpa.rl_environments.redis import Redis
from gym_hpa.rl_environments.online_boutique import OnlineBoutique
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_hpa.gnn.gnn import CustomGNNExtractor

# Logging
from policies.util.util import test_model


from gym_hpa.paths import RESULTS_DIR


logging.basicConfig(filename="run.log", filemode="w", level=logging.INFO)
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

parser = argparse.ArgumentParser(description="Run ILP!")
parser.add_argument(
    "--alg", default="ppo", help='The algorithm: ["ppo", "recurrent_ppo", "a2c"]'
)
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

parser.add_argument("--steps", default=500, help="The steps for saving.")
parser.add_argument("--total_steps", default=5000, help="The total number of steps.")

args = parser.parse_args()


policy_kwargs = dict(
    features_extractor_class=CustomGNNExtractor,
    features_extractor_kwargs=dict(
        num_nodes=11,
        node_feature_dim=4,
        num_edges=15,
        edge_feature_dim=1,
        edge_index=torch.tensor(
            [
                [9, 9, 9, 9, 9, 9, 9, 0, 2, 8, 8, 8, 8, 8, 8],
                [0, 1, 2, 8, 6, 5, 3, 1, 7, 2, 4, 5, 6, 1, 10],
            ]
        ),  # Must be torch.Tensor of shape (2, num_edges)
        features_dim=24,  # Output feature dimension for SB3 policy
    ),
)


def get_model(alg, env, tensorboard_log):
    model = 0
    ## the batch size was fixed at 125 to clean the output , must update later
    if alg == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            n_steps=500,
            batch_size=125,
            policy_kwargs=policy_kwargs,
        )
    elif alg == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, tensorboard_log=tensorboard_log
        )
    elif alg == "a2c":
        model = A2C(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log
        )  # , n_steps=steps
    else:
        logging.info("Invalid algorithm!")

    return model


def get_load_model(alg, tensorboard_log, load_path):
    if alg == "ppo":
        return PPO.load(
            load_path,
            reset_num_timesteps=False,
            verbose=1,
            tensorboard_log=tensorboard_log,
            n_steps=500,
        )
    elif alg == "recurrent_ppo":
        return RecurrentPPO.load(
            load_path,
            reset_num_timesteps=False,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )  # n_steps=steps
    elif alg == "a2c":
        return A2C.load(
            load_path,
            reset_num_timesteps=False,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
    else:
        logging.info("Invalid algorithm!")


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


def main():
    # Import and initialize Environment
    logging.info(args)

    alg = args.alg
    k8s = args.k8s
    use_case = args.use_case
    goal = args.goal

    loading = args.loading
    load_path = args.load_path
    training = args.training
    testing = args.testing
    test_path = args.test_path

    steps = int(args.steps)
    total_steps = int(args.total_steps)

    env = get_env(use_case, k8s, goal)

    scenario = ""
    if k8s:
        scenario = "real"
    else:
        scenario = "simulated"

    tensorboard_log = os.path.join(RESULTS_DIR, use_case, scenario, goal)

    name = (
        alg
        + "_env_"
        + env.name
        + "_goal_"
        + goal
        + "_k8s_"
        + str(k8s)
        + "_totalSteps_"
        + str(total_steps)
    )

    # callback
    checkpoint_callback = CheckpointCallback(
        save_freq=steps, save_path="logs/" + name, name_prefix=name
    )

    if training:
        if loading:  # resume training
            model = get_load_model(alg, tensorboard_log, load_path)
            model.set_env(env)
            model.learn(
                total_timesteps=total_steps,
                tb_log_name=name + "_run",
                callback=checkpoint_callback,
            )
        else:
            model = get_model(alg, env, tensorboard_log)
            init_params = {
                name: param.clone()
                for name, param in model.policy.features_extractor.named_parameters()
            }
            model.learn(
                total_timesteps=50,
                tb_log_name=name + "_run",
                callback=checkpoint_callback,
            )
        for name, param in model.policy.features_extractor.named_parameters():
            if not torch.equal(param, init_params[name]):
                print(f"{name}: Parameter updated ✅")
    else:
        print(f"{name}: No change ❌")

        model.save(name)

    if testing:
        model = get_load_model(alg, tensorboard_log, test_path)
        test_model(
            model,
            env,
            n_episodes=100,
            n_steps=110,
            smoothing_window=5,
            fig_name=name + "_test_reward.png",
        )


if __name__ == "__main__":
    main()
