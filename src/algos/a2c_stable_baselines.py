import gymnasium as gym
import tensorflow as tf
from typing import List, Tuple, Callable, Dict, Optional, Union, Type, TypeVar
from torch.distributions import Dirichlet
import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn import Sequential as Seq, Linear, LeakyReLU
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.algos.dirichlet_distribution import DirichletDistribution

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

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
    ):
        super(CustomNetwork, self).__init__()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 1)
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        a_probs = self.policy_net(features).squeeze(2)
        return a_probs
    
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
        self.mlp_extractor = CustomNetwork(self.features_dim)
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        action_dim = self.action_space.shape[-1]
        self.action_dist = DirichletDistribution(action_dim)
        self.action_net = self.action_dist.proba_distribution_net(0)
        self.value_net = nn.Identity()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions)


