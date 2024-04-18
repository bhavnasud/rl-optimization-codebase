from __future__ import print_function
from tqdm import trange
import numpy as np
import torch
from src.envs.network_flow_env import NetworkFlow
from src.algos.a2c_gnn import A2C
from src.algos.reb_flow_solver import solveRebFlow
from torch_geometric.data import Data
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 10000000
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
MAX_STEPS_TRAINING = 100
MAX_STEPS_VALIDATION = 10


random.seed(100)
env = NetworkFlow()

writer = SummaryWriter()


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model = A2C(env=env, input_size=2).to(device)

epochs = trange(NUM_EPOCHS)
for i_episode in epochs:
    model.train() #set model in train mode
    obs = env.reset()  # initialize environment
    print("start node ", env.start_node, " goal node ", env.goal_node)
    episode_reward = 0

    done = False
    step = 0
    prev_obs = None

    log_probs = []
    rewards = []
    while not done and step < MAX_STEPS_TRAINING:
        obs = env.get_current_state()
        cur_region = np.argmax(obs.x[:, 0]).item()
        action_rl = model.select_action(obs)
        
        # convert from Dirichlet distribution to integer distribution
        # TODO: switch from just choosing max probability region to rounding distribution
        max_region = np.argmax(action_rl)
        desired_commodity_distribution = {
            env.region[i]: 1 if i == max_region else 0
            for i in range(len(env.region))
        }
        # desired_commodity_distribution = {
        #     env.region[i]: round(
        #         action_rl[i] * env.total_commodity
        #     )
        #     for i in range(len(env.region))
        # }
        # select action based on action_rl
        # TODO: switch to using optimizer rather than hardcoding action selection
        action = []
        self_edge_index = -1
        for edge_index, edge in enumerate(env.edges):
            (i,j) = edge
            if i == cur_region and j == cur_region:
                self_edge_index = edge_index

            if j == max_region and i == cur_region:
                action.append(1)
            else:
                action.append(0)
        # if it is not possible to get from i to j, default to taking a self edge
        if (cur_region, max_region) not in env.G.edges:
            action[self_edge_index] += 1

        # action = solveRebFlow(
        #     env,
        #     "network_flow_reb",
        #     desired_commodity_distribution,
        #     CPLEX_PATH,
        #     "saved_files",
        #     use_current_time=True
        # )

        # Take action in environment
        next_state, reward, done = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        model.rewards.append(reward)
        step += 1
    
    # perform on-policy backprop
    model.training_step()

    print("episode ", i_episode + 1, "reward ", episode_reward)
    if i_episode % 100 == 0:
        writer.add_scalar("Training reward", episode_reward, i_episode)


    if i_episode % 10 == 0:
        model.eval()
        print("RUNNING VALIDATION TEST")
        with torch.no_grad():
            env.reset(start_to_end_test=True)  # initialize environment
            episode_reward = 0

            done = False
            step = 0
            prev_obs = None
            while not done and step < MAX_STEPS_VALIDATION:
                obs = env.get_current_state()
                cur_region = np.argmax(obs.x[:, 0])
                action_rl = model.select_action(obs, deterministic=True)
                max_region = np.argmax(action_rl)
                desired_commodity_distribution = {
                    env.region[i]: 1 if i == max_region else 0
                    for i in range(len(env.region))
                }
                action = []
                for edge in env.edges:
                    (i,j) = edge
                    if j == max_region and i == cur_region:
                        action.append(1)
                    else:
                        action.append(0)
                # if it is not possible to get from i to j, default to taking a self edge
                if (cur_region, max_region) not in env.G.edges:
                    action[self_edge_index] += 1
                # Take action in environment
                next_state, reward, done = env.step(action)
                episode_reward += reward
                step += 1      
            print("validation reward ", episode_reward)
            if i_episode % 100 == 0:
                writer.add_scalar("Validation Reward", episode_reward, i_episode)

writer.flush()

