# cvar_rl.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer


# ─────────────────────────────────────────────────────────────
# Quantile Network
# ─────────────────────────────────────────────────────────────

class QuantileNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_quantiles, hidden):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_actions   = n_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions * n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.n_actions, self.n_quantiles)


# ─────────────────────────────────────────────────────────────
# CVaR Agent (QR-DQN + CVaR policy)
# ─────────────────────────────────────────────────────────────

class CVaRAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        batch_size,
        buffer_size,
        target_update_freq,
        hidden,
        device,
        n_quantiles=32,
        cvar_alpha=0.25,
        seed=0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.n_quantiles = n_quantiles
        self.cvar_alpha  = cvar_alpha

        # ε schedule params (property-based)
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps   = 0

        # Quantile fractions
        self.taus = torch.linspace(0.0, 1.0, n_quantiles + 1)[1:].to(device)
        # self.taus = ((torch.arange(n_quantiles, device=device) + 0.5) / n_quantiles)

        # Networks
        self.online = QuantileNetwork(state_dim, n_actions, n_quantiles, hidden).to(device)
        self.target = QuantileNetwork(state_dim, n_actions, n_quantiles, hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, device)

    # ─────────────────────────────────────────────────────────
    # Epsilon (same as DQN/IQN)
    # ─────────────────────────────────────────────────────────

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    # ─────────────────────────────────────────────────────────
    # CVaR computation
    # ─────────────────────────────────────────────────────────

    def _cvar_values(self, quantiles):
        """
        quantiles: [B, A, N]
        returns:   [B, A]
        """
        k = max(1, int(self.n_quantiles * self.cvar_alpha))
        return quantiles[:, :, :k].mean(dim=2)

    # ─────────────────────────────────────────────────────────
    # Action selection (vectorized ε-greedy)
    # ─────────────────────────────────────────────────────────

    def select_actions(self, states: torch.Tensor) -> np.ndarray:
        n = states.shape[0]
        actions = np.zeros(n, dtype=np.int64)

        eps = self.epsilon
        random_mask = np.random.random(n) < eps

        # Random actions
        actions[random_mask] = np.random.randint(
            0, self.n_actions, size=random_mask.sum()
        )

        # Greedy (CVaR-based)
        if (~random_mask).any():
            with torch.no_grad():
                q = self.online(states[~random_mask])     # [B, A, N]
                cvar_q = self._cvar_values(q)             # [B, A]
                actions[~random_mask] = cvar_q.argmax(dim=1).cpu().numpy()

        return actions

    # ─────────────────────────────────────────────────────────

    def store(self, s, a, r, ns, d):
        self.buffer.add_batch(s, a, r, ns, d)

    # ─────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current quantiles
        q = self.online(states)  # [B, A, N]
        actions_expanded = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.n_quantiles)
        q = q.gather(1, actions_expanded).squeeze(1)  # [B, N]

        with torch.no_grad():
            # Double Q-style action selection (important for stability)
            next_q_online = self.online(next_states)
            next_cvar     = self._cvar_values(next_q_online)
            next_actions  = next_cvar.argmax(dim=1)

            next_actions_expanded = next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.n_quantiles)
            next_q_target = self.target(next_states)
            next_q = next_q_target.gather(1, next_actions_expanded).squeeze(1)

            target = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_q

        # ── Quantile regression loss ─────────────────────────
        td_error = (target.unsqueeze(1) - q.unsqueeze(2)).clamp(-10, 10)
        # td_error = (target.unsqueeze(1) - q.unsqueeze(2)).clamp(-200, 200)

        huber = torch.where(
            td_error.abs() <= 1.0,
            0.5 * td_error.pow(2),
            td_error.abs() - 0.5,
        )

        tau = self.taus.view(1, self.n_quantiles, 1)
        loss = (torch.abs(tau - (td_error.detach() < 0).float()) * huber).mean()
        # loss = (torch.abs(tau - (td_error.detach() < 0).float()) * huber).sum(dim=2).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (critical for stability)
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)

        self.optimizer.step()

        # Target update
        if self.total_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()
