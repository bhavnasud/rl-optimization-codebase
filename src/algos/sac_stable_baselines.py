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

from stable_baselines3.common.distributions import (
    Distribution,
)

SelfDirichletDistribution = TypeVar("SelfDirichletDistribution", bound="DirichletDistribution")

class DirichletDistribution(Distribution):
    """
    Dirichlet distribution

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
       """
        Create the layer that represents the distribution:
        You can then get probabilities using a softplus.

        :return:
        """
       action_logits = nn.Identity()
       return action_logits
    
    def proba_distribution(self: SelfDirichletDistribution, action_logits: torch.Tensor) -> SelfDirichletDistribution:
        concentration = F.softplus(action_logits)
        concentration += torch.rand(concentration.shape) * 1e-20
        self.concentration = concentration
        self.distribution = Dirichlet(concentration)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        # potentially try out rsample for stability
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return self.concentration / (self.concentration.sum(dim=1)[:, None])

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the feature extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#     ):
#         super(CustomNetwork, self).__init__()

#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(feature_dim, 1)
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Linear(feature_dim, 1)
#         )

#     def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
#         # TODO: move these networks to feature extractor as well, policy_net and value_net can just
#         # be torch identity
#         a_probs = self.policy_net(features).squeeze(2)
#         return a_probs
    
#     def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
#         features = torch.sum(features, dim=1)
#         return_val = self.value_net(features)
#         return return_val
    

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

    def __init__(self, observation_space, features_dim: int = 10):
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
        num_nodes = x.shape[1]
        edge_index = observations['edge_index'].type(torch.LongTensor)
        edge_attr = observations['edge_features']
        data_list = [Data(x=x[i], edge_index=edge_index[i], edge_attr=edge_attr[i]) for i in range(x.shape[0])]
        loader = DataLoader(data_list, batch_size=x.shape[0])
        for data in loader:
            x_pp = F.leaky_relu(self.conv1(data.x, data.edge_index, data.edge_attr))
            x_pp = F.leaky_relu(self.conv2(x_pp, data.edge_index, data.edge_attr))
            x_pp = F.leaky_relu(self.conv3(x_pp, data.edge_index, data.edge_attr))
            x_pp = torch.cat([data.x, x_pp], dim=1)
            return_val = x_pp.reshape(-1, num_nodes, self.features_dim)
            return return_val
        # TODO: figure out if this can be done as batch operation instead of looping
        # for idx in range(batch_size):
        #     x_idx = x[idx]
        #     edge_attr_idx = edge_attr[idx]
        #     edge_index_idx = edge_index[idx]

        #     x_pp_idx = F.leaky_relu(self.conv1(x_idx, edge_index_idx, edge_attr_idx))
        #     x_pp_idx = F.leaky_relu(self.conv2(x_pp_idx, edge_index_idx, edge_attr_idx))
        #     x_pp_idx = F.leaky_relu(self.conv3(x_pp_idx, edge_index_idx, edge_attr_idx))
        #     x_pp_idx = torch.cat([x_idx, x_pp_idx], dim=1)
        #     # TODO: figure out if I can directly use this as network output instead of applying MLP after
        #     # output = self.linear(x_pp_idx).flatten()
        #     if return_val is None:
        #         return_val = x_pp_idx[None, :, :]
        #     else:
        #         return_val = torch.cat([return_val, x_pp_idx[None, :, :]])
        return return_val
