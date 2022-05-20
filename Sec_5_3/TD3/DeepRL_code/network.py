import torch
from torch import nn
import torch.nn.functional as F

class TD3Actor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_shape[0]),
        )
    def forward(self, states):
        return torch.tanh(self.net(states))
    def sample(self, states):
        return self(states)
        
class TD3Critic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)









