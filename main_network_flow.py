from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.network_flow_env import NetworkFlow
from src.algos.sac import SAC
from src.algos.policy_network import PolicyNetwork
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import json
from torch_geometric.data import Data
import copy
import torch.optim as optim
import random

NUM_EPOCHS = 10000
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
BATCH_SIZE = 100

random.seed(100)
env = NetworkFlow()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# model = SAC(
#     env=env,
#     input_size=1, # is this supposed to be the number of node features?
#     hidden_size=256,
#     p_lr=1e-3,
#     q_lr=1e-3,
#     alpha=0.3,
#     batch_size=BATCH_SIZE,
#     use_automatic_entropy_tuning=False,
#     clip=500,
#     critic_version=4,
# ).to(device)

policy_net = PolicyNetwork(input_size=env.nregion, hidden_size=256, output_size=env.nregion)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

epochs = trange(NUM_EPOCHS)
for i_episode in epochs:
    obs = env.reset()  # initialize environment
    episode_reward = 0
    actions = []

    current_eps = []
    done = False
    step = 0
    prev_obs = None

    log_probs = []
    rewards = []
    while not done:
        obs = env.get_current_state()
        state = torch.FloatTensor(obs.x)[:, 0]
        # action_rl = model.select_action(obs)
        print("state ", state)
        action_rl, log_prob = policy_net(state)
        log_probs.append(log_prob)
        # convert from Dirichlet distribution to integer distribution
        desired_commodity_distribution = {
            env.region[i]: round(
                action_rl[i].detach().numpy() * env.total_commodity
            )
            for i in range(len(env.region))
        }
        # print("desired commodity distribution ", desired_commodity_distribution)
        # reb flow solver assumes that it is always possible to achieve that desired
        # commodity distribution (or better) which is not true in our case
        action = solveRebFlow(
            env,
            "network_flow_reb",
            desired_commodity_distribution,
            CPLEX_PATH,
            "saved_files",
            use_current_time=True
        )

        # Take action in environment
        next_state, reward, done = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        # if step > 0:
        #     model.replay_buffer.store(
        #         obs, action_rl, reward, next_state
        #     )

        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f}"
        )

        # prev_obs = copy.deepcopy(obs)
        
        # step += 1
        # if i_episode > 10:
        #     # sample from memory and update model
        #     if (len(model.replay_buffer.data_list) >= BATCH_SIZE):
        #         batch = model.replay_buffer.sample_batch(
        #             BATCH_SIZE, norm=False)
        #         model.update(data=batch)
        #     else:
        #         print("not enough data!")
    # Compute returns
    returns = []
    R = 0
    gamma = 0.99
    # print("rewards ", rewards)
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    # Normalize returns
    returns = torch.tensor(returns)
    # print("returns before ", returns)
    # print("mean ", returns.mean())
    # print("std ", returns.std())
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = [0]
    # print("returns ", returns)
    # print("log probs ", log_probs)

    # Policy gradient update
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    print("episode ", i_episode + 1, "reward ", episode_reward)
