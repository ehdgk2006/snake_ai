from copy import deepcopy
import torch
from torch import nn
import numpy as np
import torch.functional as F
from collections import namedtuple
import random
import math
from collections import deque
import itertools
import matplotlib.pyplot as plt

from snake import SnakeGame, GameState, Cell, Direction
     


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self, n_inputs, n_actions, n_supports):                                                  
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions*n_supports)
        )
    
        self.n_supports = n_supports
        self.n_actions = n_actions
        self.n_inputs = n_inputs

    def forward(self,state):
        x = self.net(state)
        return nn.Softmax(dim=2)(x.view(-1, self.n_actions, self.n_supports)), nn.LogSoftmax(dim=2)(x.view(-1, self.n_actions, self.n_supports))
     

transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity 
        self.memory = [] 
        self.position = 0 
        self.batch_size = batch_size 
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
     

class Agent:
    def __init__(self, n_inputs, n_actions, n_supports):
        self.q = Network(n_inputs, n_actions, n_supports).to(device)
        self.target = Network(n_inputs, n_actions, n_supports).to(device)
        self.update_target()
        self.V_min = -10
        self.V_max = 10 
        self.gamma = 0.99
        self.delta_z = (self.V_max - self.V_min) / (n_supports - 1)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=0.01)
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.n_supports = n_supports
        self.target_update_frequency = 10

    def action(self, state, epsilon):
        if np.random.randn() < epsilon:
            return np.random.randint(0, self.n_actions-1)
        
        else:
            z_distribution = torch.from_numpy(
                np.array([[self.V_min + i * self.delta_z for i in range(self.n_supports)]])
                )
            z_distribution = torch.unsqueeze(z_distribution, 2).float().to(device)

            Q_dist, _ = self.q.forward(state)
            Q_dist = Q_dist.detach()
            Q_target = torch.matmul(Q_dist, z_distribution)

            return Q_target.argmax(dim=1)[0].detach().cpu().numpy()[0]

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def update(self, buffer):
        if len(buffer) < buffer.batch_size:
            return 
        
        batch = buffer.sample()
        batch = transition(*zip(*batch))

        batch_size = buffer.batch_size

        states = batch.state
        next_states = batch.next_state
        rewards = torch.FloatTensor(batch.reward).to(device)
        dones = torch.FloatTensor(batch.done).to(device)
        actions = torch.tensor(batch.action).long().to(device)

        z_dist = torch.from_numpy( 
            np.array(
                [[self.V_min + i * self.delta_z for i in range(self.n_supports)]] * batch_size
                )
            )
        z_dist = torch.unsqueeze(z_dist, 2).float().to(device)
        _, Q_log_dist = self.q.forward(torch.FloatTensor(states).to(device))
        Q_log_dist = Q_log_dist[torch.arange(batch_size), actions, :]
        
        Q_next_target_dist, _ = self.target.forward(torch.FloatTensor(next_states).to(device))

        Q_target = torch.matmul(Q_next_target_dist, z_dist).squeeze(1)

        max_Q_next_target= Q_next_target_dist[torch.arange(batch_size), torch.argmax(Q_target, dim=1).squeeze(1), :]

        m = torch.zeros(batch_size, self.n_supports).to(device)
        for j in range(self.n_supports):

            T_zj = torch.clamp(rewards + self.gamma * (1 - dones) * (self.V_min + j * self.delta_z), min = self.V_min, max = self.V_max)
            bj = (T_zj - self.V_min) / self.delta_z
            l = bj.floor().long()
            u = bj.ceil().long()

            Q_l = torch.zeros(m.size()).to(device)
            Q_l.scatter_( 1, l.reshape( (batch_size,1) ), max_Q_next_target[:,j].unsqueeze(1) * (u.float() - bj.float()).unsqueeze(1) )
            Q_u = torch.zeros(m.size()).to(device)
            Q_u.scatter_( 1, u.reshape( (batch_size,1) ), max_Q_next_target[:,j].unsqueeze(1) * (bj.float() - l.float()).unsqueeze(1) )
            m = m + Q_l
            m = m + Q_u

        self.optimizer.zero_grad()
        loss = - torch.sum( torch.sum( torch.mul(Q_log_dist, m), -1 ), -1 ) / batch_size

        loss.backward()
        self.optimizer.step()
        self.target_update_frequency += 1

        if (self.target_update_frequency == 100):
            self.update_target()
            self.target_update_frequency = 0


def analysis_tool(number_of_atoms):
    EPISODES = 1001
    BATCH_SIZE = 128
    ENV = SnakeGame()
    REWARD_BUFFER = deque([0.0], maxlen=100)
    MEMORY_BUFFER = ReplayMemory(10000, BATCH_SIZE)  
    REWARDS = list()
    AGENT = Agent(29, 3, number_of_atoms)    
    
    EPSILON_START = 0.9
    EPSILON_END = 0.005
    EPSILON_DECAY = 500
    score = 0

    for EPISODE in range(EPISODES):
        
        epsilon = np.interp(EPISODE, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        print(f"Episode {EPISODE} started. epsilon value is {epsilon}. score is {score}")
        state = ENV.get_state()
        episode_reward = 0 

        t = 0
        for STEP in itertools.count():
            action = AGENT.action(torch.FloatTensor(state).to(device), epsilon)
            next_state, reward, done, _, _ = ENV.step(action)
            if reward == 1.:
                t = 0
                print('.')
            if t == 100:
                reward = -1.
                done = True
            MEMORY_BUFFER.push(state, action, next_state, reward, done)
            AGENT.update(MEMORY_BUFFER)
            state = deepcopy(next_state)
            episode_reward = episode_reward + reward

            if done:
                score = ENV.length - 4
                ENV.init()
                REWARD_BUFFER.append(episode_reward)
                REWARDS.append(np.mean(REWARD_BUFFER))
                break
            t += 1
        
        if EPISODE % 100 == 0:
            torch.save(AGENT.q.state_dict(), f'./saves/snake{EPISODE}.pt')
    return REWARDS

if __name__ == '__main__':
    C51_REWARDS = analysis_tool(51)
    
    length = len(C51_REWARDS)

    fig, ax = plt.subplots(figsize=(15, 10))

    x = np.linspace(0, length, length)

    ax.plot(x, C51_REWARDS, label='C51', color='red')

    ax.set_xlabel("Step from 0 to 1000 / Information earned when step % 10 == 0")
    ax.set_ylabel("Last 100 steps average reward.")
    ax.set_title("Average reward comparison. C5, C11, C21, C51")

    ax.legend()
    ax.grid()
    ax.plot()
    plt.show()
