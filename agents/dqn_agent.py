# agents/dqn_agent.py

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        n_inputs = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        obs_shape,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10_000,
        batch_size=32,
        target_update=1000,
        device=None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        # state: numpy array, shape=obs_shape
        if random.random() < epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals = self.policy_net(s)
                return int(q_vals.argmax(1).item())

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.push(
            np.array(s, copy=False),
            a,
            r,
            np.array(s_next, copy=False),
            done
        )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.steps_done += 1
        trans = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(np.array(trans.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(trans.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(trans.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodically update the target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
