import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, LogNormal, Poisson
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import grid, degree
from collections import namedtuple
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.99
args.log_interval = 10

class EdgeConv(MessagePassing):
    def __init__(self, node_size=4, edge_size=0, out_channels=4):
        super().__init__(aggr='min', flow="source_to_target")
        self.mlp = Seq(Linear(2 * node_size + edge_size, out_channels),
                       # maybe change relu
                       LeakyReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


#########################################
############## A2C ACTOR ################
#########################################

class Actor(torch.nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, node_size=4, edge_size=0, hidden_dim=32, out_channels=1):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.conv2 = EdgeConv(hidden_dim, edge_size, hidden_dim)
        self.conv3 = EdgeConv(hidden_dim, edge_size, hidden_dim)
        # TODO: add more message passing layers, equivalent to depth of network
        # output of this edge conv will serve as node feature input to next edge conv
        self.h_to_concentration = nn.Linear(node_size + hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # print("x shape ", x.shape)
        # print("edge attr ", edge_attr)
        x_pp = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x_pp = F.leaky_relu(self.conv2(x_pp, edge_index, edge_attr))
        x_pp = F.leaky_relu(self.conv3(x_pp, edge_index, edge_attr))
        x_pp = torch.cat([x, x_pp], dim=1)
        alpha = self.h_to_concentration(x_pp)
        return alpha

#########################################
############## A2C CRITIC ###############
#########################################

class Critic(torch.nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, node_size=4, edge_size=2, hidden_dim=32, out_channels=1):
        super(Critic, self).__init__()
        # self.hidden_dim = hidden_dim
        
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.conv2 = EdgeConv(hidden_dim, edge_size, hidden_dim)
        self.conv3 = EdgeConv(hidden_dim, edge_size, hidden_dim)
        self.g_to_v = nn.Linear(node_size + hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = F.leaky_relu(self.conv2(x_pp, edge_index, edge_attr))
        x_pp = F.leaky_relu(self.conv3(x_pp, edge_index, edge_attr))
        x_pp = torch.cat([x, x_pp], dim=1)
        x_pp = torch.sum(x_pp, dim=0)

        v = self.g_to_v(x_pp)
        return v

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, actor, critic, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.actor = actor
        self.critic = critic        
        self.optimizers = self.configure_optimizers()
        
        self.eps = eps
        self.device = device
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        #self.env_baseline = env_baseline
        self.to(self.device)
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        data = obs
        graph = data.to(self.device)
        
        # actor: computes concentration parameters of a X distribution
        # print("edge attr is ", graph.edge_attr)
        a_probs = self.actor(graph.x, graph.edge_index, graph.edge_attr)
        concentration = F.softplus(a_probs).reshape(-1)
        concentration += torch.rand(concentration.shape) * jitter

        # critic: estimates V(s_t)
        value = self.critic(graph.x, graph.edge_index, graph.edge_attr)
        return concentration, value
    
    def select_action(self, obs, deterministic=False):
        concentration , value = self.forward(obs, jitter=0 if deterministic else 1e-20)
        print("concentration ", concentration)
        if deterministic:
            action = (concentration) / (concentration.sum())
            return list(action.cpu().numpy()) 
        else:   
            dirichlet = Dirichlet(concentration)
            action = dirichlet.sample()
            # action += torch.rand(action.shape) * 1e-20
            print("sampled action ", action, " from concnetration ", concentration)
            dir_log_prob = dirichlet.log_prob(action)
            print("Log prob is ", dir_log_prob)
            # if dir_log_prob < 0:
            #     quit()
            self.saved_actions.append(SavedAction(dir_log_prob, value))
            return list(action.cpu().numpy())

    def training_step(self, tensorboard_writer=None, i_episode=0):
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


        # TODO: try normalizing reward again to see if that helps with stability
        returns = torch.tensor(returns)
        if (len(returns) == 1):
            std_dev = 0
        else:
            std_dev = returns.std()
        returns = (returns - returns.mean()) / (std_dev + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            print("log prob ", log_prob, "R ", R, " value ", value)
            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)
            # policy_losses.append(-log_prob * R)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        print("policy losses ", policy_losses)
        a_loss = torch.stack(policy_losses).sum()
        print("a loss ", a_loss)
        entropy_loss = torch.mean(-0.2 * torch.abs(torch.tensor(saved_actions)))
        a_loss = a_loss + entropy_loss
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        # a_loss = v_loss + entropy_loss
        v_loss.backward()
        self.optimizers['c_optimizer'].step()

        if i_episode % 100 == 0:
            tensorboard_writer.add_scalar("Policy loss", a_loss, i_episode)
            # tensorboard_writer.add_scalar("Critic loss", v_loss, i_episode)
        
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