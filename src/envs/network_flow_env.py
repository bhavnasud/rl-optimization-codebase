"""
Simple Network Flow Environment
-----------------------------------------
This file contains the specifications for the network flow simulator.
"""
from collections import defaultdict
import torch
import numpy as np
import subprocess
import os
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import json
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt


# # Function to check if graph is connected
# def is_connected(graph):
#     return nx.is_connected(graph)

# Function to generate a random connected graph that is not complete
def generate_connected_graph(num_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    
    # Start with a spanning tree (connected graph with no cycles)
    for i in range(1, num_nodes):
        node_to_connect = random.randint(0, i-1)
        G.add_edge(i, node_to_connect)
        G.add_edge(node_to_connect, i)
    
    # # Add additional edges randomly to ensure it's not complete
    # edges_to_add = num_nodes * 2
    # while edges_to_add > 0:
    #     node1 = random.randint(0, num_nodes-1)
    #     node2 = random.randint(0, num_nodes-1)
    #     if node1 != node2 and not G.has_edge(node1, node2):
    #         G.add_edge(node1, node2)
    #         G.add_edge(node2, node1)
    #         edges_to_add -= 1
    
    return G

class NetworkFlow:
    # initialization
    def __init__(
        self, num_nodes=10
    ):
        # graph, complete graph for now
        # self.G = nx.erdos_renyi_graph(num_nodes, p=0.15) # probability of edge?
        # self.G = self.G.to_directed()

        # Generate a connected graph that is not complete
        self.G = generate_connected_graph(num_nodes)
        # Check if the graph is connected (optional)
        # print("Is the graph connected?", is_connected(self.G))

        # Draw the graph
        nx.draw(self.G, with_labels=True)
        plt.show()
        # print("edges ", self.G.edges)
        self.region = list(self.G)  # set of nodes
        self.edges = list(self.G.edges)
        # allows for variable total amount of commodity
        self.total_commodity = 1
        # edges of graph in format expected by RL network
        self.edge_index = self.get_edge_index()
        self.nregion = len(self.G)
        self.time = 0  # current time
        self.acc = defaultdict(dict)
        self.start_node, self.goal_node = random.choices(self.region, k=2)
        print("start ", self.start_node, "end ", self.goal_node)
        # shortest_path = nx.shortest_path(G, source=1, target=5)

        for n in self.region:
            self.acc[n][0] = self.total_commodity if n == self.start_node else 0
        for i in self.G.edges:
            self.G.edges[i]['time'] = 1
        self.reset()

    def get_edge_index(self):
        # print("edges ", self.edges)
        source_nodes, target_nodes = zip(*list(self.edges))
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)
        edges = torch.from_numpy(np.vstack((source_nodes, target_nodes))).to(torch.long)
        return edges

    def get_current_state(self):
        # current state (input to RL policy) includes current
        # commodity distribution (integer distribution that adds up to total commodity), 
        # travel times, 
        # and graph layout (self.edge_index) 
        node_data = torch.FloatTensor([self.acc[i][self.time] for i in range(len(self.G.nodes))]).unsqueeze(1)
        # print("node data ", node_data)
        edge_data = torch.FloatTensor([self.G.edges[i,j]['time'] for i,j in self.edges]).unsqueeze(1)
        return Data(node_data, self.edge_index, edge_data)

    def step(self, flows):
        """
        Params:
            flows is a list of edge flows corresponding to self.edges
        Returns:
        next state (Data object)
        integer reward for that step
            reward is 0 if all commodity reaches goal, -1 otherwise
        boolean indicating whether trajectory is over or not
            currently trajectory only ends when all commodity reaches goal,
            should I set a number of steps limit?
        """
        self.reward = 0
        # copy commodity distribution from current time to next
        for n in self.region:
            self.acc[n][self.time + 1] = self.acc[n][self.time]
        # add flows to commodity distribution for next timestamp
        for n, flow in enumerate(flows):
            [i, j] = self.edges[n]
            if (i, j) not in self.G.edges:
                continue
            # update the position of the commodities
            if flow > 0:
                self.acc[i][self.time + 1] -= flow
                self.acc[j][self.time + 1] += flow
        # check that commodities were conserved (no node has negative commodity)
        for n in self.region:
            if self.acc[n][self.time + 1] < 0:
                print("NEGATIVE COMMODITY AT NODE ", n)
        self.time += 1
        # 0 reward if all commodity at goal, -1 otherwise
        if self.acc[self.goal_node][self.time] == self.total_commodity:
            # print("REACHED GOAL ", self.goal_node)
            # print(self.acc)
            return self.get_current_state(), 0, True
        else:
            # TODO: normalize reward by travel time of shortest path
            return self.get_current_state(), -1, False
            
    def reset(self):
        # resets environment for next trajectory, randomly chooses
        # start and goal node and travel times
        # all commodity starts at start node
        self.time = 0  # current time
        self.acc = defaultdict(dict) # maps nodes to time to amount of commodity at that node at that time
        # self.start_node, self.goal_node = random.choices(self.region, k=2)
        for n in self.region:
            self.acc[n][0] = self.total_commodity if n == self.start_node else 0
        # for i in self.G.edges:
        #     self.G.edges[i]['time'] = random.randint(1,5)
