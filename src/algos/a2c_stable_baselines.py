import gymnasium as gym
import tensorflow as tf
from typing import List, Tuple, Callable, Dict, Optional, Union, Type
from torch.distributions import Dirichlet
import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, LeakyReLU
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 1), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        a_probs = self.policy_net(features).squeeze(2)
        concentration = F.softplus(a_probs)
        return concentration
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.sum(features, dim=1)
        return_val = self.value_net(features)
        return return_val
    


class CustomMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomMultiInputActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        # TODO: don't hardcode latent dim policy here
        self.mlp_extractor = CustomNetwork(self.features_dim, 8, 1)
    
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features
        concentrations = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        dirichlet = Dirichlet(concentrations)
        # Evaluate the values for the given observations
        if deterministic:
            actions = (concentrations) / (concentrations.sum(dim=1))
        else:   
            actions = dirichlet.sample()
        log_prob = dirichlet.log_prob(actions)
        return actions, latent_vf, log_prob
    
    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features
        concentrations = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        dirichlet = Dirichlet(concentrations)
        log_prob = dirichlet.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = torch.mean(torch.abs(concentrations), dim=1)
        return values, log_prob, entropy
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, latent_vf, log_probs = self.forward(observation, deterministic=deterministic)
        return actions

class EdgeConv(MessagePassing):
    def __init__(self, node_size=4, edge_size=0, out_channels=4):
        super().__init__(aggr='min', flow="source_to_target")
        self.mlp = Seq(Linear(2 * node_size + edge_size, out_channels),
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

class CustomMultiInputExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 8):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # TODO: don't hardcode 2 (node features dim) and 1 (edge features dim)
        # TODO: use different features_dim here as features_dim for MLP
        # node features dim will be added to new_features_dim to get features_dim features
        new_features_dim = features_dim - 2
        self.conv1 = EdgeConv(2, 1, new_features_dim)
        self.conv2 = EdgeConv(new_features_dim, 1, new_features_dim)
        self.conv3 = EdgeConv(new_features_dim, 1, new_features_dim)
        # n_input_channels = observation_space.shape[0]
        # self.cnn = torch.nn.Sequential(
        #     torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        # )


        # self.linear = torch.nn.Sequential(torch.nn.Linear(2 + features_dim, 1))

    def forward(self, observations) -> torch.Tensor:
        # TODO: figure out why they are batching for on policy algorithm
        x = observations['node_features']
        edge_index = observations['edge_index'].type(torch.LongTensor)
        edge_attr = observations['edge_features']
        batch_size = x.shape[0]
        return_val = None
        # TODO: figure out if this can be done as batch operation instead of looping
        for idx in range(batch_size):
            x_idx = x[idx]
            edge_attr_idx = edge_attr[idx]
            edge_index_idx = edge_index[idx]

            x_pp_idx = F.leaky_relu(self.conv1(x_idx, edge_index_idx, edge_attr_idx))
            x_pp_idx = F.leaky_relu(self.conv2(x_pp_idx, edge_index_idx, edge_attr_idx))
            x_pp_idx = F.leaky_relu(self.conv3(x_pp_idx, edge_index_idx, edge_attr_idx))
            x_pp_idx = torch.cat([x_idx, x_pp_idx], dim=1)
            # TODO: figure out if I can directly use this as network output instead of applying MLP after
            # output = self.linear(x_pp_idx).flatten()
            if return_val is None:
                return_val = x_pp_idx[None, :, :]
            else:
                return_val = torch.cat([return_val, x_pp_idx[None, :, :]])
        return return_val
