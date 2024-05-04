"""
Simple Node Production Environment
-----------------------------------------
This file contains the specifications for the node production simulator.
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


def generate_graph():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3])
    edges = [(0, 1), (0,2), (0,3)]
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True)
    plt.show()
    return G

class NodeProduction:
    def __init__(
        self
    ):
        self.G = generate_graph()
        self.region = list(self.G)  # set of nodes
        self.edges = list(self.G.edges)
        self.production_capacity = 10
        self.edge_index = self.get_edge_index()
        self.nregion = len(self.G)
        self.production_cost = 1
        self.consumption_reward = 1
        self.producer_node = 0
        
        self.time = 0  # current time
        self.acc = defaultdict(dict)
        self.T = 5
        self.max_steps = 100
        self.reset()

    def get_edge_index(self):
        edge_index = np.array(self.edges).T
        edge_index_tensor = torch.LongTensor(edge_index)
        return edge_index_tensor


    def get_current_state(self):
        # current state (input to RL policy) includes current
        # commodity distribution (integer distribution that adds up to total commodity), 
        # demand for next T timesteps,
        # and production capacity constraint 
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [self.acc[n][self.time] for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [self.demand[n][t] for n in self.env.region]
                            for t in range(
                                self.env.time, self.env.time + self.T
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T, self.env.nregion)
            .T
        )
        return Data(x, self.edge_index)

    def step(self, flows, production_amount, step):
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
        production_amount = min(production_amount, self.production_capacity)
        reward = 0
        # copy commodity distribution from current time to next
        for n in self.region:
            self.acc[n][self.time + 1] = self.acc[n][self.time]
        # produce commodity
        self.acc[self.producer_node][self.time + 1] += production_amount
        reward -= self.production_cost * production_amount
        # add flows to commodity distribution for next timestamp
        for edge, flow in flows.items():
            (i, j) = edge
            if edge not in self.G.edges:
                continue
            # update the position of the commodities
            if flow > 0:
                self.acc[i][self.time + 1] -= flow
                self.acc[j][self.time + 1] += flow
        # check that commodities were conserved (no node has negative commodity)
        for n in self.region:
            assert self.acc[n][self.time + 1] >= 0
        # consume commodity
        for n in self.region:
            if n != self.producer_node:
                amount_consumed = min(self.demand[i][self.time], self.acc[i][self.time + 1])
                self.acc[i][self.time + 1] -= amount_consumed
                reward += self.consumption_reward * amount_consumed
        self.time += 1
        # return next state, reward, trajectory complete boolean
        if step == self.max_steps:
            return self.get_current_state(), reward, True
        else:
            return self.get_current_state(), reward, False
            
    def reset(self):
        # resets environment for next trajectory, randomly chooses
        # demand for all timesteps
        self.time = 0  # current time
        self.acc = defaultdict(dict) # maps nodes to time to amount of commodity at that node at that time

        self.demand = defaultdict(dict)
        for n in self.region:
            for t in range(self.max_steps):
                self.demand[n][t] = random.randint(0, 5)