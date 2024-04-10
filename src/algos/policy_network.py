import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # network just takes in node state, not edge index for now
        # print("input ", x)
        # print('weights ', self.lin1.state_dict())
        x = F.leaky_relu(self.lin1(x))
        # print("layer 1 ", x)
        x = F.leaky_relu(self.lin2(x))
        # print("layer 2 ", x)
        x = F.softplus(self.lin3(x))
        # print("layer 3 ", x)
        concentration = x.squeeze(-1)
        print("concentration ", concentration)
        # print("concentration ", concentration)
        m = Dirichlet(concentration + 1e-20)
        action = m.rsample()
        # print('action ', action)
        log_prob = m.log_prob(action)
        # print("log prob ", log_prob)
        return action, log_prob