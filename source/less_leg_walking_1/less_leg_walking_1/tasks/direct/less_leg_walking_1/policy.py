import torch
from torch import nn as nn
import torch.nn.functional as F

class ResPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim): 
        super(ResPolicy, self).__init__()

        print('Tanh ver')
        self.seq = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ELU(),
            nn.Linear(hidden_dims[2], action_dim),
        )
    
    def forward(self, x):
        return self.seq(x)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super(Critic, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ELU(),
            nn.Linear(hidden_dims[2], 1),
        )
    
    def forward(self, x):
        return self.seq(x)