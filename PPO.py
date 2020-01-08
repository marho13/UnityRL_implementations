import numpy as np
import torch
import torch.hub
import torch.nn as nn
from torch.distributions import Categorical
import re
import gym
from collections import deque


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

#Make a class for the memory
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.images = []
        self.onRoads = []

    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.images = []
        self.onRoads = []

    def getMemory(self):
        dicty = {True:1, False:0}
        output = [dicty[booly] for booly in self.onRoads]
        return np.asarray(self.images), np.asarray(output)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.affine = nn.Linear(state_dim, n_latent_var)

        self.action_layer = nn.Sequential(
            self.conv_bn(  3,  32, 2),
            self.conv_dw( 32,  64, 1),
            self.conv_dw( 64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.actionOutput = nn.Linear(1024, action_dim)

        # critic
        self.value_layer = nn.Sequential(
            self.conv_bn(  3,  32, 2),
            self.conv_dw( 32,  64, 1),
            self.conv_dw( 64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.valueOutput = nn.Linear(1024, 1)

    # actor
    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    # in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'
    def conv_dw(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=stride, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        value = self.value_layer(x)
        value = value.view(-1, 1024)
        value = self.valueOutput(value)
        action = self.action_layer(x)
        action = action.view(-1, 1024)
        action = self.actionOutput(action)
        return value, action

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        state_value, action_probs = self.forward(state)
        # action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        state_value, action_probs = self.forward(state)
        # action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)

        # print(self.policy)
        # print(self.policy_old.action_layer.weight)

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0.0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprogs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for e in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs.detach() - old_logprogs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())