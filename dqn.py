"""
DQN: Deep Q-Network (PyTorch)

Classic DQN with:
- Q-network predicting E[return] per action
- MSE / Huber loss between Q(s,a) and r + gamma * max Q'(s')
- Hard target network updates
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: int = 5000,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        target_update_freq: int = 500,
        hidden: int = 128,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.online = QNetwork(state_dim, n_actions, hidden).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim, device)

        self.total_steps = 0
        self.losses = []

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def select_actions(self, states: torch.Tensor) -> np.ndarray:
        """Epsilon-greedy action selection for n_envs states."""
        n = states.shape[0]
        actions = np.zeros(n, dtype=np.int64)
        eps = self.epsilon

        random_mask = np.random.random(n) < eps
        actions[random_mask] = np.random.randint(0, self.n_actions, size=random_mask.sum())

        if (~random_mask).any():
            with torch.no_grad():
                q = self.online(states[~random_mask])
                actions[~random_mask] = q.argmax(dim=1).cpu().numpy()

        return actions

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q values
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style: online selects action, target evaluates)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.huber_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def store(self, states, actions, rewards, next_states, dones):
        self.buffer.add_batch(states, actions, rewards, next_states, dones)
        self.total_steps += len(actions)
