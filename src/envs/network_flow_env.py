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
    #### Random graph
    # G = nx.DiGraph()
    # G.add_nodes_from(range(num_nodes))
    
    # # Start with a spanning tree (connected graph with no cycles)
    # for i in range(1, num_nodes):
    #     node_to_connect = random.randint(0, i-1)
    #     G.add_edge(i, node_to_connect)
    #     # G.add_edge(node_to_connect, i)
    
    # # Add additional edges randomly to ensure it's not complete
    # edges_to_add = num_nodes
    # while edges_to_add > 0:
    #     node1 = random.randint(0, num_nodes-1)
    #     node2 = random.randint(0, num_nodes-1)
    #     if node1 != node2 and not G.has_edge(node1, node2):
    #         G.add_edge(node1, node2)
    #         # G.add_edge(node2, node1)
    #         edges_to_add -= 1

    # G = nx.complete_graph(num_nodes)
    # G = G.to_directed()

    ##### Graph 1
    # G = nx.DiGraph()

    # # Add nodes
    # source_node = 0
    # goal_node = 4
    # other_nodes = [1, 2, 3]

    # # Add nodes to the graph
    # G.add_nodes_from([source_node, goal_node] + other_nodes)

    # # Add edges between nodes
    # edges = [(0,1), (1, 0), (1, 2), (2, 1), (2, 4), (4, 2), (0,3), (3, 0), (3,4), (4,3)]
    # # edges = [(0,1), (1, 2), (2, 4), (0,3), (3,4)]

    # G.add_edges_from(edges)

    ###### Graph 2
    G = nx.DiGraph()

    # Add nodes
    start_node = 0
    goal_node = 7
    other_nodes = [1, 2, 3, 4, 5, 6]

    # Add nodes to the graph
    G.add_nodes_from([start_node, goal_node] + other_nodes)

    # Add edges between nodes
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 4), (4, 1), (1, 5), (5, 1), (2, 4), (4, 2), (2, 5), (5, 2), (2, 6), (6, 2), (3, 5), (5, 3), (3, 6), (6, 3), (4, 7), (7, 4), (5, 7), (7, 5), (6, 7), (7, 6),
             (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7,7)]

    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()
    return G, start_node, goal_node

class NetworkFlow:
    def __init__(
        self, num_nodes=10
    ):
        # Generate a connected graph that is not complete
        self.G, self.start_node, self.goal_node = generate_connected_graph(num_nodes)
        self.region = list(self.G)  # set of nodes
        self.edges = list(self.G.edges)
        self.total_commodity = 1
        self.edge_index = self.get_edge_index()
        self.nregion = len(self.G)
        
        self.time = 0  # current time
        self.acc = defaultdict(dict)
        # for i in self.G.edges:
        #     (a, b) = i
        #     # self edges will always have travel time 1
        #     if a == b:
        #         self.G.edges[i]['originalTime'] = 1
        #     # all other edges have random travel time between 1 and 5
        #     else:
        #         self.G.edges[i]['originalTime'] = random.randint(1,5)
        self.reset()

    def get_edge_index(self):
        edge_index = np.array(self.edges).T
        edge_index_tensor = torch.LongTensor(edge_index)
        return edge_index_tensor


    def get_current_state(self):
        # current state (input to RL policy) includes current
        # commodity distribution (integer distribution that adds up to total commodity), 
        # where the goal node is,
        # travel times, 
        # and graph layout (self.edge_index) 
        node_data = torch.FloatTensor([self.acc[i][self.time] for i in range(len(self.G.nodes))]).unsqueeze(1)
        node_data = torch.hstack([node_data, self.goal_node_feature[:, None]])
        return Data(node_data, self.edge_index, self.edge_data)

    def step(self, flows, step, max_steps):
        """
        Params:
            flows is a map from edge to edge flows
            only contains nonzero flows
        Returns:
        next state (Data object)
        integer reward for that step
            reward is -travel_time, plus 1 if all commodity reaches goal
        boolean indicating whether trajectory is over or not
            currently trajectory only ends when all commodity reaches goal
        """
        self.reward = 0
        # copy commodity distribution from current time to next
        for n in self.region:
            self.acc[n][self.time + 1] = self.acc[n][self.time]
        # add flows to commodity distribution for next timestamp
        total_travel_time = 0
        for edge, flow in flows.items():
            (i, j) = edge
            if edge not in self.G.edges:
                continue
            # update the position of the commodities
            if flow > 0:
                self.acc[i][self.time + 1] -= flow
                self.acc[j][self.time + 1] += flow
                total_travel_time += self.G.edges[(i,j)]['time'] * flow
        # check that commodities were conserved (no node has negative commodity)
        for n in self.region:
            assert self.acc[n][self.time + 1] >= 0
        self.time += 1
        # return next state, reward, trajectory complete boolean
        if self.acc[self.goal_node][self.time] == self.total_commodity:
            # return self.get_current_state(), -total_travel_time + 10, True
            return self.get_current_state(), -total_travel_time + 1, True
        elif step == max_steps:
            # return self.get_current_state(), -total_travel_time - 10, True
            return self.get_current_state(), -total_travel_time - 1, True
        else:
            return self.get_current_state(), -total_travel_time, False
            
    def reset(self, start_to_end_test=False):
        # resets environment for next trajectory, randomly chooses
        # start and goal node and travel times
        # all commodity starts at start node
        self.time = 0  # current time
        self.acc = defaultdict(dict) # maps nodes to time to amount of commodity at that node at that time

        for i in self.G.edges:
            (a, b) = i
            # self edges will always have travel time 1
            if a == b:
                self.G.edges[i]['originalTime'] = 1
            # all other edges have random travel time between 1 and 5
            else:
                self.G.edges[i]['originalTime'] = random.randint(1,5)
        if start_to_end_test:
            self.start_node, self.goal_node = 0, self.nregion - 1
        else:
            shortest_path_length = -1
            # make sure we never choose start and goal node next to each other
            while shortest_path_length < 3:
                self.start_node, self.goal_node = np.random.choice(self.region, 2, replace=False)
                shortest_path = nx.shortest_path(self.G, source=self.start_node, target=self.goal_node, weight='originalTime')
                shortest_path_length = len(shortest_path)
        self.goal_node_feature = torch.IntTensor([1 if i == self.goal_node else 0 for i in range(self.nregion)])
        for n in self.region:
            self.acc[n][0] = self.total_commodity if n == self.start_node else 0
        shortest_path = nx.shortest_path(self.G, source=self.start_node, target=self.goal_node, weight='originalTime')
        print("shortest path ", shortest_path)
        # normalize travel times by travel time of shortest path
        shortest_path_travel_time = 0
        for n in range(len(shortest_path) - 1):
            (a, b) = shortest_path[n], shortest_path[n + 1]
            shortest_path_travel_time += self.G.edges[(a,b)]['originalTime']

        for i in self.G.edges:
            self.G.edges[i]['time'] = self.G.edges[i]['originalTime'] / float(shortest_path_travel_time)
            (a, b) = i
        for n in range(len(shortest_path) - 1):
            (a, b) = shortest_path[n], shortest_path[n + 1]
            print(f"shortest path leg {n} travel time is {self.G.edges[(a,b)]['time']}")
        self.edge_data = torch.FloatTensor([self.G.edges[i,j]['time'] for i,j in self.edges]).unsqueeze(1)

