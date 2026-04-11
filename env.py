"""
Bimodal Investment Environment
Two-action investment environment with multiplicative returns.
  Action 0: Safe investment, always +5%
  Action 1: Risky investment, +20% or -10%
  Reward = log return (ln(capital_t / capital_{t-1})), naturally normalized
  Capital compounds over time.
"""

import numpy as np
import torch


class BimodalInvestmentEnv:
    """
    Vectorized bimodal investment environment for parallel training.

    State:  [log(capital/100), step_fraction]  shape: (n_envs, 2)
    Action: 0 (safe) or 1 (risky)
    Reward: log return — ln(capital_t / capital_{t-1})
            safe  → ln(1.05)  ≈  0.0488
            risky → ln(1.20)  ≈  0.1823  or  ln(0.90) ≈ -0.1054
            Naturally O(0.05-0.18), no explosion.
    """

    def __init__(self, n_envs: int = 8, episode_length: int = 50, device: str = "cuda"):
        self.n_envs = n_envs
        self.episode_length = episode_length
        self.device = device
        self.state_dim = 2
        self.n_actions = 2

        self.safe_return   = 0.05
        self.risky_win     = 0.20
        self.risky_loss    = -0.10
        self.risky_prob_win = 0.55

        self.reset()

    def reset(self):
        self.capital = np.ones(self.n_envs, dtype=np.float32) * 100.0
        self.steps   = np.zeros(self.n_envs, dtype=np.int32)
        return self._get_states()

    def _get_states(self):
        log_cap   = np.log(self.capital / 100.0)        # 0 at start, grows with capital
        step_frac = self.steps / self.episode_length
        states    = np.stack([log_cap, step_frac], axis=1)
        return torch.FloatTensor(states).to(self.device)

    def step(self, actions: np.ndarray):
        prev_capital = self.capital.copy()

        safe_mask  = (actions == 0)
        risky_mask = (actions == 1)

        self.capital[safe_mask] *= (1 + self.safe_return)

        risky_outcomes = np.random.random(self.n_envs) < self.risky_prob_win
        risky_returns  = np.where(risky_outcomes, self.risky_win, self.risky_loss)
        self.capital[risky_mask] *= (1 + risky_returns[risky_mask])

        # Log return: naturally ~O(0.05), no explosion
        rewards = np.log(self.capital / prev_capital)

        self.steps += 1
        dones = (self.steps >= self.episode_length)

        if dones.any():
            self.capital[dones] = 100.0
            self.steps[dones]   = 0

        return (
            self._get_states(),
            torch.FloatTensor(rewards).to(self.device),
            torch.BoolTensor(dones).to(self.device),
        )
