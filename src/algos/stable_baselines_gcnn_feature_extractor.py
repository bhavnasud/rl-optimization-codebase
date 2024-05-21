from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch
from torch.nn import Sequential as Seq, Linear, LeakyReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
    :param features_dim: (int) Number of features extracted per node.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, hidden_features_dim: int = 8, node_features_dim=2, edge_features_dim=1):
        super(CustomMultiInputExtractor, self).__init__(observation_space, node_features_dim + hidden_features_dim)
        self.conv1 = EdgeConv(node_features_dim, edge_features_dim, hidden_features_dim)
        self.conv2 = EdgeConv(hidden_features_dim, 1, hidden_features_dim)
        self.conv3 = EdgeConv(hidden_features_dim, 1, hidden_features_dim)

    def forward(self, observations) -> torch.Tensor:
        x = observations['node_features'].type(torch.IntTensor)
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
