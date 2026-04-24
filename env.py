"""
Bimodal Investment Environment
Two-action investment environment with multiplicative returns.
  Action 0: Safe investment, always +10%
  Action 1: Risky investment, +35% or -15% (p_win=0.50)

Reward design — flat per-step + terminal capital bonus:
  Per-step reward = raw return rate (NOT scaled by capital):
    safe:  +0.10  always
    risky: +0.35 (prob 0.50) or -0.15 (prob 0.50)
    => Both have identical per-step EV = 0.10
    => DQN sees same mean Q for both actions → indifferent

  Terminal reward = log(final_capital / 100):
    safe:  log(1.10^100)  = 9.53  (deterministic)
    risky: E[log] = 100 * 0.5*(log1.35+log0.85) = 100*0.0677 = 6.77
    => Safe terminal reward is HIGHER on average
    => But this only shows up in the DISTRIBUTION of episode returns,
       not in the per-step mean — IQN can model this, DQN cannot

State: [log(capital/100), step_fraction]
"""

import numpy as np
import torch


class BimodalInvestmentEnv:
    def __init__(self, n_envs: int = 8, episode_length: int = 50, device: str = "cuda"):
        self.n_envs         = n_envs
        self.episode_length = episode_length
        self.device         = device
        self.state_dim      = 2
        self.n_actions      = 2

        self.safe_return    = 0.05
        self.risky_win      = 0.12 #0.35
        self.risky_loss     = -0.5 #-0.15
        self.risky_prob_win = 0.9

        self.reset()

    def reset(self):
        self.capital = np.ones(self.n_envs, dtype=np.float32) * 100.0
        self.steps   = np.zeros(self.n_envs, dtype=np.int32)
        return self._get_states()

    def _get_states(self):
        log_cap   = np.log(self.capital / 100.0)
        step_frac = self.steps / self.episode_length
        return torch.FloatTensor(np.stack([log_cap, step_frac], axis=1)).to(self.device)

    def step(self, actions: np.ndarray):
        safe_mask  = (actions == 0)
        risky_mask = (actions == 1)

        # Compute per-step returns (flat, state-independent)
        returns = np.zeros(self.n_envs, dtype=np.float32)
        returns[safe_mask] = self.safe_return

        risky_outcomes = np.random.random(self.n_envs) < self.risky_prob_win
        risky_returns  = np.where(risky_outcomes, self.risky_win, self.risky_loss)
        returns[risky_mask] = risky_returns[risky_mask]

        # Update capital (for state tracking)
        self.capital *= (1 + returns)

        # Flat per-step reward: just the return rate, NOT scaled by capital
        # Both actions have EV = 0.10 → DQN is indifferent
        rewards = returns.copy()

        self.steps += 1
        dones = (self.steps >= self.episode_length)

        # Terminal bonus: log(final_capital / 100) — rewards compounding
        # Safe path gets ~9.53, risky path gets ~6.77 on average
        if dones.any():
            rewards[dones] += np.log(self.capital[dones] / 100.0) #/100
            self.capital[dones] = 100.0
            self.steps[dones]   = 0

        return (
            self._get_states(),
            torch.FloatTensor(rewards).to(self.device),
            torch.BoolTensor(dones).to(self.device),
        )
