import numpy as np
import random
import torch
import torch.hub
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
from collections import Counter


import math
device = 'cpu'
torch.manual_seed(42)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# class QValues():
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = 'cpu'
#
#     @staticmethod
#     def get_current(policy_net, states, actions):
#         return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
#
#     @staticmethod
#     def get_next(target_net, next_states):
#         final_state_locations = next_states.flatten(start_dim=1) \
#             .max(dim=1)[0].eq(0).type(torch.bool)
#         non_final_state_locations = (final_state_locations == False)
#         non_final_states = next_states[non_final_state_locations]
#         batch_size = next_states.shape[0]
#         values = torch.zeros(batch_size).to(QValues.device)
#         values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
#         return values

# def extract_tensors(experiences):
#     # Convert batch of Experiences to Experience of batches
#     batch = Experience(*zip(*experiences))
#
#     t1 = torch.stack(batch.state, dim=0)
#     t2 = torch.stack(batch.action, dim=0)
#     t3 = torch.stack(batch.reward, dim=0)
#     t4 = torch.stack(batch.next_state, dim=0)
#
#     return (t1,t2,t3,t4

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step//self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate

# class Memory:
#     def __init__(self, capacity=7000):
#         self.capacity = capacity
#         self.memory = []
#         self.push_count = 0
#
#     def push(self, experience):
#         if len(self.memory) < self.capacity:
#             self.memory.append(experience)
#         else:
#             self.memory[self.push_count % self.capacity] = experience
#         self.push_count += 1
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def can_provide_sample(self, batch_size):
#         return len(self.memory) >= batch_size
#
#     def __len__(self):
#         return len(self.memory)


class policyNetworks(nn.Module):
    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        # self.device = device
        self.randPolicy = {"Rand":0, "Policy":0}
        self.current_step = 0
        self.num_actions = outputSize
        self.fc1 = nn.Linear(in_features=stateDim, out_features=n_latent_var).float()
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var).float()
        self.out = nn.Linear(in_features=n_latent_var, out_features=outputSize).float()


    def forward(self, t):
        t = t.flatten().float()
        t = self.fc1(t).float()
        t = F.relu(t).float()
        t = self.fc2(t).float()
        t = F.relu(t).float()
        t = self.out(t).float()
        return t

    def act(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            self.randPolicy["Rand"] += 1
            action = random.randrange(self.num_actions)
            output = []
            for a in range(self.num_actions):
                if a != (action-1):
                    output.append(0.0)
                else:
                    output.append(1.0)
            return torch.tensor(output).to('cpu').detach().numpy() # explore
        else:
            self.randPolicy["Policy"] += 1
            state = torch.tensor(state, dtype=torch.float32)
            # state = torch.from_numpy(state)
            with torch.no_grad():
                return self.forward(state).to('cpu').detach().numpy() # exploit



class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        # torch.cuda.set_device(0)
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas)
        self.target_net = policyNetworks(state_dim, action_dim, n_latent_var).to(device)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()

        self.MseLoss = nn.MSELoss()


    def update(self, memory, BATCH_SIZE, update_timestep):
        for _ in range((update_timestep//2)//BATCH_SIZE):
            batch, idx, weight = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # # to Transition of batch-arrays.
            batch = Transition(*zip(*batch))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.float32)
            non_final_next_states = torch.stack([s for s in batch.next_state
                                               if s is not None], dim=0)
            state_batch = torch.stack(batch.state, dim=0)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # tempStates = state_batch.cpu().detach().numpy()
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            ones = (Counter(non_final_mask.detach().numpy()))
            reward_calc = torch.zeros(int(ones[1.0]))
            state_action_values = torch.zeros(int(ones[1.0]))
            lastInd = 0
            for x in range(len(state_batch)-1, -1, -1):
                # print(x)
                # state = torch.from_numpy(state.cpu().detach().numpy())
                # state = torch.as_tensor(state)
                # if reward_batch[x] ==
                if non_final_mask[x].detach().numpy() != 0.0:
                    stateAction = self.policy_net(state_batch[x]).max(0)[0]
                    state_action_values[lastInd] = (stateAction)
                    reward_calc[lastInd] = (reward_batch[x])
                    lastInd += 1
            # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            # reward_calc = torch.from_numpy(np.array(reward_calc))
            # state_action_values = torch.from_numpy(np.array(state_action_values))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            # next_state_values = torch.zeros(200)
            next_state_values = torch.zeros(int(ones[1.0]))
            for y in range(len(non_final_next_states)-1, -1, -1):
                stateAction = self.target_net(non_final_next_states[y]).max(0)[0].detach()
                next_state_values[y] = stateAction

            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(0)[0].detach()
            # next_state_values = torch.from_numpy(np.array(next_state_values))
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_calc

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        print(self.policy_net.randPolicy["Rand"]/(self.policy_net.randPolicy["Rand"]+self.policy_net.randPolicy["Policy"]))
        self.policy_net.randPolicy = {"Rand":0, "Policy":0}