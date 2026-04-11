"""
IQN: Implicit Quantile Network (PyTorch)

Why IQN should outperform DQN on the bimodal environment:
- The risky action has a BIMODAL return distribution (+20% / -10%).
- DQN collapses this to a single mean (~+5.5%), barely above safe (+5%),
  so it struggles to confidently prefer the risky action.
- IQN models the FULL return distribution via sampled quantile levels tau~U(0,1).
  It explicitly represents both the upside (+20%) and downside (-10%) as separate
  quantile values, making risk-adjusted decisions possible.
- As capital compounds, the variance of the risky action matters: IQN can
  learn to always take risky (positive EV) without being confused by the noise.

Loss: normalised by N * N_target so scale is independent of quantile count.
"""

import copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer


class IQNNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, state_emb_dim=256,
                 quantile_emb_dim=64, n_cos=64, hidden=128):
        super().__init__()
        self.n_cos             = n_cos
        self.quantile_emb_dim  = quantile_emb_dim

        # Deeper state encoder — gives IQN more signal to work with
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_emb_dim), nn.ReLU(),
            nn.Linear(state_emb_dim, state_emb_dim), nn.ReLU(),
            nn.Linear(state_emb_dim, state_emb_dim), nn.ReLU(),
        )
        self.cos_embedding = nn.Sequential(
            nn.Linear(n_cos, quantile_emb_dim), nn.ReLU(),
        )
        self.state_proj = nn.Linear(state_emb_dim, quantile_emb_dim)
        self.output = nn.Sequential(
            nn.Linear(quantile_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.register_buffer(
            "cos_idx",
            torch.arange(1, n_cos + 1, dtype=torch.float32).unsqueeze(0) * math.pi,
        )

    def forward(self, states, n_quantiles):
        B = states.shape[0]
        s_proj = self.state_proj(self.state_encoder(states))  # (B, Q_emb)

        taus     = torch.rand(B, n_quantiles, device=states.device)
        cos_feat = torch.cos(taus.unsqueeze(-1) * self.cos_idx)          # (B, N, n_cos)
        q_emb    = self.cos_embedding(
            cos_feat.view(B * n_quantiles, self.n_cos)
        ).view(B, n_quantiles, self.quantile_emb_dim)                    # (B, N, Q_emb)

        combined = s_proj.unsqueeze(1) * q_emb                          # (B, N, Q_emb)
        return self.output(combined), taus                               # (B, N, A), (B, N)


class IQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000,
                 batch_size=256, buffer_size=100_000, target_update_freq=500,
                 n_quantiles=16, n_quantiles_target=16, n_quantiles_policy=64,
                 kappa=1.0, hidden=128, state_emb_dim=256, device="cuda", seed=0):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.n_quantiles        = n_quantiles
        self.n_quantiles_target = n_quantiles_target
        self.n_quantiles_policy = n_quantiles_policy
        self.kappa              = kappa
        self.device             = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.online = IQNNetwork(state_dim, n_actions,
                                 state_emb_dim=state_emb_dim,
                                 hidden=hidden).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        self.optimizer   = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer      = ReplayBuffer(buffer_size, state_dim, device)
        self.total_steps = 0

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.total_steps / self.epsilon_decay)

    def select_actions(self, states):
        n, eps = states.shape[0], self.epsilon
        actions   = np.zeros(n, dtype=np.int64)
        rand_mask = np.random.random(n) < eps
        actions[rand_mask] = np.random.randint(0, self.n_actions, rand_mask.sum())
        if (~rand_mask).any():
            with torch.no_grad():
                qv, _ = self.online(states[~rand_mask], self.n_quantiles_policy)
                actions[~rand_mask] = qv.mean(1).argmax(1).cpu().numpy()
        return actions

    def _quantile_huber_loss(self, pred_qv, target_z, actions, taus):
        B, N, _ = pred_qv.shape
        N_t     = target_z.shape[1]
        act_idx = actions.unsqueeze(1).expand(B, N)
        pred    = pred_qv.gather(2, act_idx.unsqueeze(2)).squeeze(2)  # (B, N)
        u       = target_z.unsqueeze(2) - pred.unsqueeze(1)           # (B, N_t, N)
        abs_u   = u.abs()
        huber   = torch.where(abs_u <= self.kappa,
                              0.5 * u.pow(2),
                              self.kappa * (abs_u - 0.5 * self.kappa))
        weights = (taus.unsqueeze(1) - (u < 0).float()).abs()         # (B, N_t, N)
        # Normalise by N*N_t: loss scale independent of quantile count
        return (weights * huber).sum(dim=(1, 2)).mean() / (N * N_t)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            nqv_p, _   = self.online(next_states, self.n_quantiles_policy)
            next_acts   = nqv_p.mean(1).argmax(1)
            nqv_t, _   = self.target(next_states, self.n_quantiles_target)
            na_idx      = next_acts.unsqueeze(1).expand(-1, self.n_quantiles_target)
            next_z      = nqv_t.gather(2, na_idx.unsqueeze(2)).squeeze(2)
            target_z    = rewards.unsqueeze(1) + self.gamma * next_z * (1 - dones.unsqueeze(1))
        pred_qv, taus = self.online(states, self.n_quantiles)
        loss = self._quantile_huber_loss(pred_qv, target_z, actions, taus)
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
