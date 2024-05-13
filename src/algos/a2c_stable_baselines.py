import gymnasium as gym
import tensorflow as tf
from typing import List
import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, LeakyReLU

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


# def gnn_extractor(flat_observations: tf.Tensor, act_fun: tf.function,
#                   network_graphs: List[nx.DiGraph], dm_memory_length: int,
#                   iterations: int = 10, layer_size: int = 128,
#                   layer_count: int = 3,
#                   vf_arch: str = "mlp"):
#     """
#     Constructs a graph network from the graph passed in. Then inputs are
#     traffic demands, placed on nodes as feature vectors. The output policy
#     tensor is built from the edge outputs (in line with the softmin routing
#     approach). The value function can be switched between mlp and graph net
#     using the net_arch argument.

#     :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the
#     specified network. If all layers are shared, then ``latent_policy ==
#     latent_value``
#     """
#     # get graph info
#     sorted_edges_list = [sorted(network_graph.edges()) for network_graph in
#                          network_graphs]
#     num_edges_list = [len(l) for l in sorted_edges_list]
#     num_nodes_list = [network_graph.number_of_nodes() for network_graph in
#                       network_graphs]
#     max_edges_len = max(num_edges_list)
#     sorted_edges_list = [edges + [(0, 0)] * (max_edges_len - len(edges)) for
#                          edges in sorted_edges_list]
#     sorted_edges = tf.constant(sorted_edges_list)
#     num_edges = tf.constant(num_edges_list)
#     num_nodes = tf.constant(num_nodes_list)

#     # start manipulating input data
#     latent = flat_observations
#     num_batches = tf.shape(latent)[0]

#     # prepare helper data for graphs per entry in batch
#     graph_idxs = tf.cast(latent[:, 0], np.int32)
#     num_nodes_per_batch = tf.map_fn(lambda i: num_nodes[i], graph_idxs)
#     num_edges_per_batch = tf.map_fn(lambda i: num_edges[i], graph_idxs)
#     observation_sizes = tf.multiply(num_nodes_per_batch, dm_memory_length * 2)
#     full_observations = latent[:, 1:]
#     trimmed_observations = tf.RaggedTensor.from_tensor(full_observations,
#                                                        lengths=observation_sizes)

#     # reshape data into correct sizes for gnn input
#     node_features = tf.reshape(trimmed_observations.flat_values,
#                                [-1, 2 * dm_memory_length],
#                                name="node_feat_input")
#     node_features = tf.pad(node_features,
#                            [[0, 0], [0, layer_size - (2 * dm_memory_length)]])

#     # initialise unused input features to all zeros
#     edge_features = tf.zeros((tf.reduce_sum(num_edges_per_batch), layer_size),
#                              np.float32)
#     global_features = tf.zeros((num_batches, layer_size), np.float32)

#     # repeat edge information across batches and flattened for graph_nets
#     sender_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 0], graph_idxs)
#     sender_nodes = tf.RaggedTensor.from_tensor(sender_nodes,
#                                                lengths=num_edges_per_batch)
#     sender_nodes = sender_nodes.flat_values
#     receiver_nodes = tf.map_fn(lambda i: sorted_edges[i][:, 1], graph_idxs)
#     receiver_nodes = tf.RaggedTensor.from_tensor(receiver_nodes,
#                                                  lengths=num_edges_per_batch)
#     receiver_nodes = receiver_nodes.flat_values

#     # repeat graph information across batches and flattened for graph_nets
#     n_node_list = num_nodes_per_batch
#     n_edge_list = num_edges_per_batch

#     input_graph = GraphsTuple(nodes=node_features,
#                               edges=edge_features,
#                               globals=global_features,
#                               senders=sender_nodes,
#                               receivers=receiver_nodes,
#                               n_node=n_node_list,
#                               n_edge=n_edge_list)

#     model = DDRGraphNetwork(layer_size=layer_size, layer_count=layer_count)
#     output_graph = model(input_graph, iterations)

#     # NB: reshape needs num_edges as otherwise output tensor has too many
#     #     unknown dims
#     # first split per graph
#     output_edges = tf.RaggedTensor.from_row_lengths(output_graph.edges,
#                                                     num_edges_per_batch)
#     # make a normal tensor so we can slice out edge values
#     output_edges = output_edges.to_tensor()
#     # then extract from each split the values we want and squeeze away last axis
#     output_edges = tf.squeeze(output_edges[:, :, 0::layer_size], axis=2)
#     # finally pad to correct size for output
#     output_edges = tf.pad(output_edges, [[0, 0], [0, max_edges_len -
#                                                   tf.shape(output_edges)[1]]])

#     # global output is softmin gamma
#     output_globals = tf.reshape(output_graph.globals, (-1, layer_size))
#     output_globals = output_globals[:, 0]
#     output_globals = tf.reshape(output_globals, (-1, 1))

#     latent_policy_gnn = tf.concat([output_edges, output_globals], axis=1)
#     # build value function network
#     latent_vf = vf_builder(vf_arch, flat_observations, act_fun,
#                            output_graph, input_graph, layer_size, layer_count,
#                            iterations)

#     return latent_policy_gnn, latent_vf

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
        self.conv1 = EdgeConv(2, 1, features_dim)
        self.conv2 = EdgeConv(features_dim, 1, features_dim)
        self.conv3 = EdgeConv(features_dim, 1, features_dim)
        # n_input_channels = observation_space.shape[0]
        # self.cnn = torch.nn.Sequential(
        #     torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        # )


        self.linear = torch.nn.Sequential(torch.nn.Linear(2 + features_dim, 1))

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
            output = self.linear(x_pp_idx).flatten()
            if return_val is None:
                return_val = output[None, :]
            else:
                return_val = torch.cat([return_val, output[None, :]])
        return return_val
