from gymnasium import spaces
from typing import List, Tuple, Any, Dict, Optional, Union, Type, TypeVar
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, LeakyReLU
import torch.nn as nn

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from src.algos.dirichlet_distribution import DirichletDistribution


class CustomSACActor(Actor):
    """
    Actor network (policy) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(CustomSACActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images
        )
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, 1)
        )
        action_dim = self.action_space.shape[-1]
        self.action_dist = DirichletDistribution(action_dim)

    def get_action_dist_params(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            concentration
        """
        features = self.extract_features(obs, self.features_extractor)
        a_logits = self.policy_net(features).squeeze(2)
        concentration = F.softplus(a_logits)
        concentration += torch.rand(concentration.shape) * 1e-20
        return concentration

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        concentration = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(concentration, deterministic=deterministic)

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        concentration = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(concentration)

class CustomSACContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(CustomSACContinuousCritic, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
            features = torch.sum(features, dim=1)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
            features = torch.sum(features, dim=1)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class CustomMultiInputSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super(CustomMultiInputSACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomSACActor(**actor_kwargs).to(self.device)
    
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomSACContinuousCritic(**critic_kwargs).to(self.device)
