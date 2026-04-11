"""
Replay buffer supporting vectorized environment transitions.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add a batch of transitions (from n_envs parallel envs)."""
        n = len(actions)
        idxs = np.arange(self.ptr, self.ptr + n) % self.capacity

        self.states[idxs] = states.cpu().numpy() if torch.is_tensor(states) else states
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards.cpu().numpy() if torch.is_tensor(rewards) else rewards
        self.next_states[idxs] = next_states.cpu().numpy() if torch.is_tensor(next_states) else next_states
        self.dones[idxs] = dones.cpu().numpy().astype(np.float32) if torch.is_tensor(dones) else dones.astype(np.float32)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]).to(self.device),
            torch.LongTensor(self.actions[idxs]).to(self.device),
            torch.FloatTensor(self.rewards[idxs]).to(self.device),
            torch.FloatTensor(self.next_states[idxs]).to(self.device),
            torch.FloatTensor(self.dones[idxs]).to(self.device),
        )

    def __len__(self):
        return self.size
