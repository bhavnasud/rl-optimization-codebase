from __future__ import print_function
from tqdm import trange
import numpy as np
import torch
from src.envs.network_flow_env import NetworkFlowEnv
# from src.algos.a2c_gnn import A2C
# from src.algos.a2c_mpnn import Actor, Critic, A2C
from src.algos.reb_flow_solver import solveRebFlow
from torch_geometric.data import Data
import torch.optim as optim
from copy import deepcopy
import random
import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Sequence, Type
from collections import OrderedDict
from matplotlib.lines import Line2D
import gymnasium as gym
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
from src.algos.a2c_stable_baselines import CustomMultiInputExtractor
from torch.optim import Adam


from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info




NUM_EPOCHS = 1000000
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
# MAX_STEPS_TRAINING = 10
# MAX_STEPS_VALIDATION = 10
# CHECKPOINT_PATH = "network_flow_checkpoints_saved/episode_443000.pth"
CHECKPOINT_PATH = ""
SAVE_CHECKPOINTS = False
TRAIN = False

random.seed(104)
# env = NetworkFlow()

writer = SummaryWriter()


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# actor = Actor(2, 1, 8, 1)
# critic = Critic(2, 1, 8, 1)
# model = A2C(env=env, actor=actor, critic=critic)

# if len(CHECKPOINT_PATH) > 0:
#     model.load_checkpoint(CHECKPOINT_PATH)


class MyDummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            # if self.buf_dones[env_idx]:
            #     # save final observation where user can get it, then reset
            #     self.buf_infos[env_idx]["terminal_observation"] = obs
            #     obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, tensorboard_writer, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tensorboard_writer = tensorboard_writer
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward = self.evaluate_model()
            self.tensorboard_writer.add_scalar("Validation reward", mean_reward, self.num_timesteps)
            print("Average validation reward is ", mean_reward)
        return True

    def evaluate_model(self):
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            env.env_method("set_start_to_end_test", True)
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                # print("done is ", done)
                episode_reward += reward
                # print("got reward ", reward)
            # print("done with loop")
            episode_rewards.append(episode_reward)
        env.env_method("set_start_to_end_test", False)
        return sum(episode_rewards) / self.n_eval_episodes

# Register the environment
register(id='CustomEnv-v0', entry_point=NetworkFlowEnv)

# Create the environment
# env = DummyVecEnv([lambda: gym.make('CustomEnv-v0')])
env = MyDummyVecEnv([lambda: gym.make('CustomEnv-v0')])

# policy = CustomPolicy(sess=tf.Session(), ob_space=env.observation_space, ac_space=env.action_space,
#                       n_env=1, n_steps=1, n_batch=1)

policy_kwargs = dict(
    features_extractor_class=CustomMultiInputExtractor,
    features_extractor_kwargs=dict(features_dim=8),
)


model = A2C("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, use_rms_prop=False, learning_rate=1e-4)

eval_callback = EvaluationCallback(env, writer)

# model = A2C(CustomPolicy, env, verbose=1)
model.learn(total_timesteps=1000000000, callback=eval_callback)

# epochs = trange(NUM_EPOCHS)
# for i_episode in epochs:
#     model.train() #set model in train mode
#     obs = env.reset()  # initialize environment
#     episode_reward = 0

#     done = False
#     step = 0
#     prev_obs = None

#     log_probs = []
#     rewards = []
#     while not done:
#         obs = env.get_current_state()
#         cur_region = np.argmax(obs.x[:, 0]).item()
#         action_rl = model.select_action(obs) # desired commodity distribution
#         # select action based on action_rl
#         action = {}
#         highest_node_prob = 0
#         selected_edge_index = -1
#         for n, edge in enumerate(env.edges):
#             (i,j) = edge
#             # only consider adjacent nodes
#             if i == cur_region:
#                 if action_rl[j] > highest_node_prob:
#                     highest_node_prob = action_rl[j]
#                     selected_edge_index = n
#         action[env.edges[selected_edge_index]] = 1

#         # Take action in environment
#         next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_TRAINING)
#         episode_reward += reward
#         rewards.append(reward)
#         model.rewards.append(reward)
#         step += 1
    
#     if TRAIN:
#         # perform on-policy backprop
#         model.training_step(tensorboard_writer=writer, i_episode=i_episode)

#         if i_episode % 100 == 0:
#             writer.add_scalar("Training reward", episode_reward, i_episode)
#             if i_episode % 1000 == 0 and SAVE_CHECKPOINTS:
#                 model.save_checkpoint(f"network_flow_checkpoints/episode_{i_episode}.pth")

#     # validation test with deterministic concentration and always from 0 to 7
#     if i_episode % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             true_shortest_path = env.reset(start_to_end_test=True)  # initialize environment
#             episode_reward = 0

#             done = False
#             step = 0
#             prev_obs = None
#             predicted_shortest_path = []
#             cur_region = -1
#             while not done:
#                 obs = env.get_current_state()
#                 cur_region = np.argmax(obs.x[:, 0])
#                 predicted_shortest_path.append(cur_region.item())
#                 action_rl = model.select_action(obs, deterministic=True)
#                 action = {}
#                 highest_node_prob = 0
#                 selected_edge_index = -1
#                 for n, edge in enumerate(env.edges):
#                     (i,j) = edge
#                     # only consider adjacent nodes
#                     if i == cur_region:
#                         if action_rl[j] > highest_node_prob:
#                             highest_node_prob = action_rl[j]
#                             selected_edge_index = n
#                 action[env.edges[selected_edge_index]] = 1
#                 # Take action in environment
#                 next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_VALIDATION)
#                 episode_reward += reward
#                 step += 1 
#             cur_region = np.argmax(next_state.x[:, 0])
#             predicted_shortest_path.append(cur_region.item())    
#             custom_pos = {
#                 0: (0, 1),
#                 1: (1, 0),
#                 2: (1, 1),
#                 3: (1, 2),
#                 4: (2, 0),
#                 5: (2, 1),
#                 6: (2, 2),
#                 7: (3, 1)
#             }
#             if i_episode % 100 == 0:
#                 writer.add_scalar("Validation Reward", episode_reward, i_episode)
#             # Draw the graph
#             if i_episode % 100 == 0:
#                 plt.clf()
#                 nx.draw(env.G, custom_pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, font_weight='bold')

#                 # Highlight the true shortest path
#                 nx.draw_networkx_edges(env.G, custom_pos, edgelist=list(zip(true_shortest_path[:-1], true_shortest_path[1:])), edge_color='green', width=3)

#                 # Highlight the calculated shortest path
#                 nx.draw_networkx_edges(env.G, custom_pos, edgelist=list(zip(predicted_shortest_path[:-1], predicted_shortest_path[1:])), edge_color='red', width=1)


#                 legend_handles = [
#                     Line2D([0], [0], color='red', lw=2),
#                     Line2D([0], [0], color='green', lw=2)
#                 ]
#                 # Add a legend
#                 plt.legend(legend_handles, ['Predicted Shortest Path', 'True Shortest Path'])
#                 plt.text(2.2, 0, f'Difference in reward: {round(episode_reward, 2)}', ha='left', va='top')

#                 # Show the plot
#                 # plt.show(block=False)
#                 plt.pause(1)

# writer.flush()

