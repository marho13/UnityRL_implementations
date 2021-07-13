import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import re
import gym
from collections import deque


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(12345678)

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

#Make a class for the actorCritic
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var): # n_latent_var = number of variables in hidden layer
        super(ActorCritic, self).__init__()

        #actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        #critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
        )

    def forward(self):
        raise NotImplementedError

    #creates the action to take
    def act(self, state, memory):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action_probs.detach().numpy()

    #Does the evaluation
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0.0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprogs = torch.stack(memory.logprobs).detach()

        # Optimize policy for K epochs:
        for e in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs.detach() - old_logprogs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach() #logprobs calculated from piOld and pi
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())