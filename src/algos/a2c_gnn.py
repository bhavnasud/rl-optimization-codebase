"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DenseSAGEConv, DenseGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import grid
from torch_geometric.utils import to_dense_adj
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 8)
        self.lin2 = nn.Linear(8, 8)
        self.lin3 = nn.Linear(8, 1)
        # self.lin1 = nn.Linear(5, 256)
        # self.lin2 = nn.Linear(256, 256)
        # self.lin3 = nn.Linear(256, 5)
    
    def forward(self, data):
        out = F.leaky_relu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr))
        # print("Out ", out)
        x = out + data.x
        # print("after adding x ", x)
        # print("after adding node data ", x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        # print("actor output ", x)
        return x
        # x = F.leaky_relu(self.lin1(data.x[:, 0]))
        # # print("layer 1 ", x)
        # x = F.leaky_relu(self.lin2(x))
        # # print("layer 2 ", x)
        # x = self.lin3(x)
        # return x
        

#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 8)
        self.lin2 = nn.Linear(8, 8)
        self.lin3 = nn.Linear(8, 1)
        # self.lin1 = nn.Linear(5, 256)
        # self.lin2 = nn.Linear(256, 256)
        # self.lin3 = nn.Linear(256, 1)
    
    def forward(self, data):
        out = F.leaky_relu(self.conv1(data.x, data.edge_index, edge_weight=data.edge_attr))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        return x
        # x = F.leaky_relu(self.lin1(data.x[:, 0]))
        # # print("layer 1 ", x)
        # x = F.leaky_relu(self.lin2(x))
        # # print("layer 2 ", x)
        # x = self.lin3(x)
        # return x

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        
        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)
        # print("x ", x.x)
        
        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)
        # print("a out ", a_out)
        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        # print("x before calling critic ", x)
        value = self.critic(x)
        return concentration, value
    
    def parse_obs(self, obs):
        return obs
    
    def select_action(self, obs, deterministic=False):
        # print("state ", obs.x)
        concentration, value = self.forward(obs, jitter = 0 if deterministic else 1e-20)
        # print("concentration ", concentration)
        if deterministic:
            action = (concentration) / (concentration.sum())
            return list(action.cpu().numpy())
        else:
            # print("concentration ", concentration)
            # print("value ", value)

            m = Dirichlet(concentration)
            action = m.sample()
            # print("dirichlet sampled action ", action)
            self.saved_actions.append(SavedAction(m.log_prob(action), value))
            return list(action.cpu().numpy())

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # print("returns ", returns)
        if (len(returns) == 1):
            std_dev = 0
        else:
            std_dev = returns.std()
        returns = (returns - returns.mean()) / (std_dev + self.eps)

        # print("normalized returns ", returns)
        # print("saved actions ", saved_actions)

        for (log_prob, value), R in zip(saved_actions, returns):
            # print("R ", R)
            # print("value ", value)
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)