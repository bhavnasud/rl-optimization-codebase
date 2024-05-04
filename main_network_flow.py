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
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 1000000
CPLEX_PATH = "/Applications/CPLEX_Studio2211/opl/bin/arm64_osx/"
MAX_STEPS_TRAINING = 10
MAX_STEPS_VALIDATION = 10
# CHECKPOINT_PATH = "network_flow_checkpoints_saved/episode_443000.pth"
CHECKPOINT_PATH = ""
SAVE_CHECKPOINTS = False
TRAIN = False

random.seed(104)
env = NetworkFlow()

writer = SummaryWriter()


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


actor = Actor(2, 1, 8, 1)
critic = Critic(2, 1, 8, 1)
model = A2C(env=env, actor=actor, critic=critic)

if len(CHECKPOINT_PATH) > 0:
    model.load_checkpoint(CHECKPOINT_PATH)

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
        # select action based on action_rl
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

        # Take action in environment
        next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_TRAINING)
        episode_reward += reward
        rewards.append(reward)
        model.rewards.append(reward)
        step += 1
    
    if TRAIN:
        # perform on-policy backprop
        model.training_step(tensorboard_writer=writer, i_episode=i_episode)

        if i_episode % 100 == 0:
            writer.add_scalar("Training reward", episode_reward, i_episode)
            if i_episode % 1000 == 0 and SAVE_CHECKPOINTS:
                model.save_checkpoint(f"network_flow_checkpoints/episode_{i_episode}.pth")

    # validation test with deterministic concentration and always from 0 to 7
    if i_episode % 10 == 0:
        model.eval()
        with torch.no_grad():
            true_shortest_path = env.reset(start_to_end_test=True)  # initialize environment
            episode_reward = 0

            done = False
            step = 0
            prev_obs = None
            predicted_shortest_path = []
            cur_region = -1
            while not done:
                obs = env.get_current_state()
                cur_region = np.argmax(obs.x[:, 0])
                predicted_shortest_path.append(cur_region.item())
                action_rl = model.select_action(obs, deterministic=True)
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
                # Take action in environment
                next_state, reward, done = env.step(action, step, max_steps=MAX_STEPS_VALIDATION)
                episode_reward += reward
                step += 1 
            cur_region = np.argmax(next_state.x[:, 0])
            predicted_shortest_path.append(cur_region.item())    
            custom_pos = {
                0: (0, 1),
                1: (1, 0),
                2: (1, 1),
                3: (1, 2),
                4: (2, 0),
                5: (2, 1),
                6: (2, 2),
                7: (3, 1)
            }
            if i_episode % 100 == 0:
                writer.add_scalar("Validation Reward", episode_reward, i_episode)
            # Draw the graph
            if i_episode % 100 == 0:
                plt.clf()
                nx.draw(env.G, custom_pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, font_weight='bold')

                # Highlight the true shortest path
                nx.draw_networkx_edges(env.G, custom_pos, edgelist=list(zip(true_shortest_path[:-1], true_shortest_path[1:])), edge_color='green', width=3)

                # Highlight the calculated shortest path
                nx.draw_networkx_edges(env.G, custom_pos, edgelist=list(zip(predicted_shortest_path[:-1], predicted_shortest_path[1:])), edge_color='red', width=1)


                legend_handles = [
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2)
                ]
                # Add a legend
                plt.legend(legend_handles, ['Predicted Shortest Path', 'True Shortest Path'])
                plt.text(2.2, 0, f'Difference in reward: {round(episode_reward, 2)}', ha='left', va='top')

                # Show the plot
                # plt.show(block=False)
                plt.pause(1)

writer.flush()

