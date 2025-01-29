import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, policy_layers):
        super().__init__()
        self.state_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_dim = np.array(envs.single_action_space.shape).prod()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # If 3 layers
        if policy_layers == 3:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(self.state_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.action_dim), std=0.01)
            )

        # If 7 layers
        if policy_layers == 7:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(self.state_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.action_dim), std=0.01)
            )
        self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def get_value(self, s):
        return self.critic(s)

    def get_action_and_value(self, s, a=None):
        action_mean = self.actor_mean(s)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if a is None:
            a = probs.sample()
        return a, probs.log_prob(a).sum(-1), probs.entropy().sum(-1), self.critic(s), probs
