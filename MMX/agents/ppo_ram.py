import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Neural Network function class
class PPONetworkRAM(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(68, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_size), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer