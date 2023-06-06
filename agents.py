import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory
from memory import Transition
from collections import defaultdict
import math

device = "cpu"

class Agent(object):
    def __init__(self, states_dim, actions_dim, discount=1, update_rate=0.1,
                 eps_init=0.8, eps_lower_bound=0.1, anneal=0.001):
        self.actions_dim = actions_dim
        self.states_dim = states_dim
        self.discount = discount  # \gamma
        self.update_rate = update_rate  # \alpha
        self.eps = eps_init
        self.steps_done = 0
        self.eps_lower_bound = eps_lower_bound
        self.EPS_DECAY = 200
        self.anneal = anneal

    def greedy(self, state, network):        
        return NotImplementedError

    def epsilon_greedy(self, state):
        is_eps = random.random()
        eps_threshold = self.eps_lower_bound + (self.eps - self.eps_lower_bound)*math.exp(-1*self.steps_done/self.EPS_DECAY)
        self.steps_done += 1
        if is_eps > self.eps:
            action = self.greedy(state)
        else:
            action = random.randint(0, self.actions_dim - 1)
        return action

    def anneal_eps(self):
        self.eps -= self.anneal
        self.eps = max(self.eps, self.eps_lower_bound)

    def update_q(self):
        return NotImplementedError


class MonteCarlo(Agent):
    def __init__(self, states_dim, actions_dim, discount=1, update_rate=0.1, eps_init=0.8, eps_lower_bound=0.1, anneal=0.001):
        super().__init__(states_dim, actions_dim, discount,
                         update_rate, eps_init, eps_lower_bound, anneal)
        self.q = np.zeros((states_dim,actions_dim))
        self.returns = defaultdict(lambda: defaultdict(list))
        

    def greedy(self, state):
        
        return np.argmax(self.q[state])
        

    def update_q(self, episode_history):
        
        G = 0
        for state, action, reward in reversed(episode_history):
            G = self.discount*G + reward
            
            self.returns[state][action].append(G)
            
            self.q[state,action] = np.mean(self.returns[state][action])

    


class QLearning(Agent):
    def __init__(self, states_dim, actions_dim, discount=1, update_rate=0.1, eps_init=0.8, eps_lower_bound=0.1, anneal=0.001, buffer_size=1000):
        super().__init__(states_dim, actions_dim, discount, update_rate, eps_init, eps_lower_bound, anneal)
        self.q_table = np.zeros((states_dim, actions_dim))
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def greedy(self, state):
        return np.argmax(self.q_table[state])

    def update_q(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            self.q_table[state,action] = (1 - self.update_rate) * self.q_table[state, action] \
                                        + self.update_rate * (reward + self.discount * np.max(self.q_table[next_state]))

    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)



class DeepQLearning(Agent):
    def __init__(self, states_dim, actions_dim, discount=1, update_rate=0.001, eps_init=0.8, eps_lower_bound=0.1, anneal=0.001, buffer_size=1000, device='cpu'):
        super().__init__(states_dim, actions_dim, discount,
                         update_rate, eps_init, eps_lower_bound, anneal)
        
        self.network = DQN(states_dim, actions_dim).to(device)
        self.target_network = DQN(states_dim,actions_dim).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.RMSprop(self.network.parameters())
        self.device =device
        self.memory = ReplayMemory(10000)
        
        
    def greedy(self,state):
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            return self.network(state).argmax(dim=1).item()
        
    def update_q(self,batch_size):
        if len(self.memory) < batch_size:
            return
        self.network.train()
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state).float().unsqueeze(1).to(self.device)
        action_batch = torch.tensor(batch.action).float().to(self.device)
        action_batch_long = torch.tensor(action_batch, dtype = torch.int64)
        reward_batch = torch.tensor(batch.reward).float().to(self.device)
        next_state_batch = torch.tensor(batch.next_state).float().unsqueeze(1).to(self.device)
        
        state_action_values = self.network(state_batch).gather(1, action_batch_long.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class DQN(nn.Module):
    
    def __init__(self):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU())
          
        self.fc3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU())
        
        self.fc4 = nn.Sequential(
            nn.Linear(16, 1))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
class deeperDQN(nn.Module):
    
    def __init__(self):
        super(deeperDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(16, 10),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.network(x).to(device)
        return x
    