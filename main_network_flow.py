import torch
from src.envs.network_flow_env import NetworkFlowEnv
import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum

from torch.utils.tensorboard import SummaryWriter
from gymnasium.envs.registration import register
from src.algos.a2c_stable_baselines import CustomMultiInputActorCriticPolicy
from src.algos.stable_baselines_gcnn_feature_extractor import CustomMultiInputExtractor
from src.algos.sac_stable_baselines import CustomMultiInputSACPolicy
from src.envs.stable_baselines_env_wrapper import MyDummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

class RLAlgorithm(Enum):
    A2C = 0
    PPO = 1
    SAC = 2

RL_ALGORITHM = RLAlgorithm.A2C
CHECKPOINT_PATH = ""


random.seed(104)

writer = SummaryWriter()


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, tensorboard_writer, eval_freq=1000, save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tensorboard_writer = tensorboard_writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = f"./checkpoints/{RL_ALGORITHM.name}"
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 or self.num_timesteps == 1:
            validation_reward = self.evaluate_model()
            self.tensorboard_writer.add_scalar("Validation reward", validation_reward, self.num_timesteps)
            print("Validation reward is ", validation_reward)
        if self.num_timesteps % self.save_freq == 0:
            self.save_checkpoint()
        return True
    
    def save_checkpoint(self):
        model_path = os.path.join(self.save_path, f"{self.num_timesteps}_steps.zip")
        self.model.save(model_path)
        print(f"Saving model checkpoint to {model_path}")


    def evaluate_model(self):
        env.env_method("set_start_to_end_test", True)
        obs = self.eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            episode_reward += reward
        env.env_method("set_start_to_end_test", False)
        env.env_method("visualize_prediction", info[0]["true_shortest_path"], info[0]["path_followed"], info[0]["episode_reward"])
        return episode_reward

# Register the environment
register(id='CustomEnv-v0', entry_point=NetworkFlowEnv)

# Create the environment
env = MyDummyVecEnv([lambda: gym.make('CustomEnv-v0')])

policy_kwargs = dict(
    features_extractor_class=CustomMultiInputExtractor,
    features_extractor_kwargs=dict(hidden_features_dim=8, node_features_dim=2, edge_features_dim=1),
    share_features_extractor=False,
)

if RL_ALGORITHM == RLAlgorithm.A2C:
    model = A2C(CustomMultiInputActorCriticPolicy,
                env, policy_kwargs=policy_kwargs, verbose=1,
                use_rms_prop=False, learning_rate=1e-4, ent_coef=0.01)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = A2C.load(CHECKPOINT_PATH, env=env)
elif RL_ALGORITHM == RLAlgorithm.PPO:
    model = PPO(CustomMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=1e-4, ent_coef=0.01)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = PPO.load(CHECKPOINT_PATH, env=env)
else:
    model = SAC(CustomMultiInputSACPolicy, env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=1e-4, ent_coef=0.01)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = SAC.load(CHECKPOINT_PATH, env=env)

eval_callback = EvaluationCallback(env, writer, eval_freq=1000)

model.learn(total_timesteps=1000000000, callback=eval_callback)
