from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.network_flow_env import NetworkFlow
from src.algos.sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import json
from torch_geometric.data import Data
import copy

NUM_EPOCHS = 10
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
BATCH_SIZE = 32

env = NetworkFlow()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model = SAC(
    env=env,
    input_size=1, # is this supposed to be the number of node features?
    hidden_size=256,
    p_lr=1e-3,
    q_lr=1e-3,
    alpha=0.3,
    batch_size=100,
    use_automatic_entropy_tuning=False,
    clip=500,
    critic_version=4,
).to(device)

epochs = trange(NUM_EPOCHS)
for i_episode in epochs:
    obs = env.reset()  # initialize environment
    episode_reward = 0
    actions = []

    current_eps = []
    done = False
    step = 0
    prev_obs = None
    while not done:
        obs = env.get_current_state()
        action_rl = model.select_action(obs)
        # convert from Dirichlet distribution to integers
        desired_commodity_distribution = {
            env.region[i]: int(
                action_rl[i] * env.total_commodity
            )
            for i in range(len(env.region))
        }
        action = solveRebFlow(
            env,
            "network_flow_reb",
            desired_commodity_distribution,
            CPLEX_PATH,
            "saved_files",
            use_current_time=True
        )

        # Take action in environment
        reward, done = env.step(action)
        episode_reward += reward
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f}"
        )
        if step > 0:
            # Q: WHY DO WE STORE BOTH OBS AND PREV OBS?
            model.replay_buffer.store(
                prev_obs, action_rl, reward, obs
            )
        prev_obs = copy.deepcopy(obs)

        step += 1
        if i_episode > 10:
            # sample from memory and update model
            # Q: HOW DOES THIS BATCHING WORK?
            batch = model.replay_buffer.sample_batch(
                BATCH_SIZE, norm=False)
            model.update(data=batch)
