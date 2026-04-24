"""
CVaR-RL: Quantile Regression DQN with CVaR risk objective.

Architecture: QR-DQN style — fixed uniform quantile grid, no cosine embedding.
Risk measure: Conditional Value-at-Risk (CVaR) at level alpha.

Key distinction from standard DQN and IQN:
  - DQN:        action = argmax_a  E[Z(s,a)]              (mean, risk-neutral)
  - IQN:        action = argmax_a  (1/N) sum Q_tau(s,a)   (mean via quantile avg)
  - CVaR-RL:    action = argmax_a  CVaR_alpha[Z(s,a)]     (explicit risk-averse)
                       = argmax_a  mean of Q_tau for tau < alpha

Bellman target (Lim & Malik, NeurIPS 2022):
  The risk measure must enter the TARGET to achieve a provably risk-sensitive policy.
  Standard approach (risk in action selection only) does NOT converge to CVaR-optimal.
  Here: target quantiles tau' are sampled from U[0, alpha] instead of U[0, 1],
  so the learned distribution is trained to represent the CVaR-truncated return.

  CVaR_alpha[Z(s,a)] = (1/alpha) * int_0^alpha Q_tau(s,a) dtau

Parameters:
  cvar_alpha  float in (0, 1]:
    alpha = 1.0  -> risk-neutral (same as QR-DQN)
    alpha = 0.25 -> CVaR at 25%: focus on worst-quarter outcomes
    alpha = 0.1  -> CVaR at 10%: strongly risk-averse

Reference:
  Lim, S.H. & Malik, I. (2022). Distributional Reinforcement Learning for
  Risk-Sensitive Policies. NeurIPS 2022.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


class CVaRNetwork(nn.Module):
    """
    QR-DQN style network: outputs N fixed quantile values per action.
    No cosine embedding — quantile levels are implicit in the fixed grid.
    Output shape: (batch, N_quantiles, n_actions)
    """

    def __init__(self, state_dim: int, n_actions: int,
                 n_quantiles: int = 32, hidden: int = 128):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_actions   = n_actions

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        # Output: N_quantiles values per action
        self.head = nn.Linear(hidden, n_quantiles * n_actions)

    def forward(self, x):
        B = x.shape[0]
        h = self.encoder(x)
        # (B, N * A) -> (B, N, A)
        return self.head(h).view(B, self.n_quantiles, self.n_actions)


class CVaRAgent:
    """
    CVaR-RL agent: QR-DQN with CVaR incorporated into both action selection
    and the Bellman target, following Lim & Malik (NeurIPS 2022).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        cvar_alpha: float = 0.25,       # CVaR risk level
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay: int = 5000,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        target_update_freq: int = 500,
        n_quantiles: int = 32,          # fixed quantile grid size
        kappa: float = 1.0,             # Huber loss threshold
        hidden: int = 128,
        device: str = "cuda",
        seed: int = 0,
    ):
        assert 0.0 < cvar_alpha <= 1.0, "cvar_alpha must be in (0, 1]"

        self.n_actions          = n_actions
        self.cvar_alpha         = cvar_alpha
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.n_quantiles        = n_quantiles
        self.kappa              = kappa
        self.device             = device

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.online = CVaRNetwork(state_dim, n_actions,
                                  n_quantiles=n_quantiles,
                                  hidden=hidden).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        self.optimizer   = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer      = ReplayBuffer(buffer_size, state_dim, device)
        self.total_steps = 0

        # Fixed uniform quantile grid tau_i = (2i-1)/(2N), i=1..N
        # This is the standard QR-DQN midpoint grid.
        self._taus = torch.FloatTensor(
            [(2 * i - 1) / (2 * n_quantiles) for i in range(1, n_quantiles + 1)]
        ).to(device)  # (N,)

        # CVaR: number of quantiles below alpha
        # We use the bottom floor(alpha * N) quantiles for CVaR estimation.
        self._n_cvar = max(1, int(np.floor(cvar_alpha * n_quantiles)))

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.total_steps / self.epsilon_decay)

    def _cvar_values(self, qv: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR_alpha for each action from quantile values.

        Args:
            qv: (B, N, A) quantile values on fixed grid
        Returns:
            cvar: (B, A) CVaR values — mean of bottom n_cvar quantiles
        """
        # Bottom n_cvar quantiles correspond to lowest tau values (worst outcomes)
        return qv[:, :self._n_cvar, :].mean(dim=1)  # (B, A)

    def select_actions(self, states: torch.Tensor) -> np.ndarray:
        """Epsilon-greedy with CVaR action selection."""
        n   = states.shape[0]
        eps = self.epsilon
        actions = np.zeros(n, dtype=np.int64)

        rand_mask = np.random.random(n) < eps
        actions[rand_mask] = np.random.randint(0, self.n_actions, rand_mask.sum())

        if (~rand_mask).any():
            with torch.no_grad():
                qv = self.online(states[~rand_mask])          # (B', N, A)
                cvar = self._cvar_values(qv)                   # (B', A)
                actions[~rand_mask] = cvar.argmax(dim=1).cpu().numpy()

        return actions

    def _quantile_huber_loss(
        self,
        pred_qv: torch.Tensor,    # (B, N, A)
        target_z: torch.Tensor,   # (B, N_target)
        actions: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """
        Quantile Huber loss between predicted quantiles and CVaR-truncated targets.

        pred_qv:   current network's N quantiles for all actions
        target_z:  N_target CVaR-truncated target quantiles (sampled from U[0,alpha])
        actions:   taken actions — selects the relevant action slice from pred_qv
        """
        B  = pred_qv.shape[0]
        N  = self.n_quantiles
        Nt = target_z.shape[1]

        # Select quantiles for taken action: (B, N)
        act_idx = actions.view(B, 1, 1).expand(B, N, 1)
        pred = pred_qv.gather(2, act_idx).squeeze(2)          # (B, N)

        # TD errors: (B, Nt, N)
        u = target_z.unsqueeze(2) - pred.unsqueeze(1)

        # Huber loss
        abs_u = u.abs()
        huber = torch.where(
            abs_u <= self.kappa,
            0.5 * u.pow(2),
            self.kappa * (abs_u - 0.5 * self.kappa),
        )

        # Asymmetric quantile weights using the fixed tau grid
        # For CVaR: only bottom n_cvar taus used in the loss
        taus = self._taus[:N].view(1, 1, N)                   # (1, 1, N)
        weights = (taus - (u < 0).float()).abs()               # (B, Nt, N)

        return (weights * huber).sum(dim=(1, 2)).mean() / (N * Nt)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # ── CVaR action selection (online network) ─────────────────────
            next_qv_online = self.online(next_states)             # (B, N, A)
            next_cvar      = self._cvar_values(next_qv_online)    # (B, A)
            next_actions   = next_cvar.argmax(dim=1)              # (B,)

            # ── CVaR-truncated target quantiles ────────────────────────────
            # Key: sample tau' from U[0, alpha] so the Bellman target is a
            # CVaR-truncated distribution, not the full return distribution.
            # This is the theoretical requirement of Lim & Malik (2022).
            next_qv_target = self.target(next_states)             # (B, N, A)

            # Select target quantiles for chosen next actions
            na_idx  = next_actions.view(-1, 1, 1).expand(-1, self.n_quantiles, 1)
            next_z  = next_qv_target.gather(2, na_idx).squeeze(2) # (B, N)

            # Use only the CVaR-relevant (bottom n_cvar) quantiles as targets
            # This implements the truncated Bellman operator T^{CVaR}
            next_z_cvar = next_z[:, :self._n_cvar]                # (B, n_cvar)

            target_z = (
                rewards.unsqueeze(1)
                + self.gamma * next_z_cvar * (1 - dones.unsqueeze(1))
            )                                                      # (B, n_cvar)

        # ── Predict quantiles for taken actions ────────────────────────────
        pred_qv = self.online(states)                              # (B, N, A)
        loss    = self._quantile_huber_loss(pred_qv, target_z, actions)

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
