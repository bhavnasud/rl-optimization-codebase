from __future__ import print_function
from tqdm import trange
import numpy as np
import torch
from src.envs.network_flow_env import NetworkFlow
from src.algos.a2c_gnn import A2C
from src.algos.a2c_mpnn import Actor, Critic, A2C
from src.algos.reb_flow_solver import solveRebFlow
from torch_geometric.data import Data
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 10000000
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
MAX_STEPS_TRAINING = 10
MAX_STEPS_VALIDATION = 10


random.seed(87)
env = NetworkFlow()

writer = SummaryWriter()


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


actor = Actor(2, 1, 8, 1)
critic = Critic(2, 1, 8, 1)
model = A2C(env=env, actor=actor, critic=critic)
# model = A2C(env=env, input_size=2).to(device)

epochs = trange(NUM_EPOCHS)
for i_episode in epochs:
    model.train() #set model in train mode
    obs = env.reset()  # initialize environment
    episode_reward = 0

    done = False
    step = 0
    prev_obs = None

    log_probs = []
    rewards = []
    while not done:
        obs = env.get_current_state()
        cur_region = np.argmax(obs.x[:, 0]).item()
        action_rl = model.select_action(obs) # desired commodity distribution
        # print("cur region ", cur_region, " action_rl ", action_rl)
        # select action based on action_rl
        # TODO: switch to using optimizer rather than hardcoding action selection
        action = {}
        highest_node_prob = 0
        selected_edge_index = -1
        for n, edge in enumerate(env.edges):
            (i,j) = edge
            # only consider adjacent nodes
            if i == cur_region:
                if action_rl[j] > highest_node_prob:
                    highest_node_prob = action_rl[j]
                    selected_edge_index = n
        action[env.edges[selected_edge_index]] = 1
        # print("action ", action)

        # action = solveRebFlow(
        #     env,
        #     "network_flow_reb",
        #     desired_commodity_distribution,
        #     CPLEX_PATH,
        #     "saved_files",
        #     use_current_time=True
        # )

        # Take action in environment
        next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_TRAINING)
        # print("reward ", reward)
        episode_reward += reward
        rewards.append(reward)
        model.rewards.append(reward)
        step += 1
    
    # perform on-policy backprop
    model.training_step(tensorboard_writer=writer, i_episode=i_episode)

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
            while not done:
                obs = env.get_current_state()
                cur_region = np.argmax(obs.x[:, 0])
                action_rl = model.select_action(obs, deterministic=True)
                # print("action_rl ", action_rl)
                action = {}
                highest_node_prob = 0
                selected_edge_index = -1
                for n, edge in enumerate(env.edges):
                    (i,j) = edge
                    # only consider adjacent nodes
                    if i == cur_region:
                        if action_rl[j] > highest_node_prob:
                            highest_node_prob = action_rl[j]
                            selected_edge_index = n
                action[env.edges[selected_edge_index]] = 1
                # print("action ", action)
                # Take action in environment
                next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_VALIDATION)
                episode_reward += reward
                step += 1      
            print("validation reward ", episode_reward)
            if i_episode % 100 == 0:
                writer.add_scalar("Validation Reward", episode_reward, i_episode)

writer.flush()

