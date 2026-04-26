# Distributional Reinforcement Learning

A research codebase comparing distributional and standard value-based RL algorithms
on two environments designed to analyze advantages of
distributional RL: a **Bimodal Investment Environment** and a **Stochastic Cliff
Walking** environment.

---

## Overview

Standard DQN learns a scalar expected return $Q(s,a) = \mathbb{E}[Z(s,a)]$, discarding
all information about the shape of the return distribution. Distributional RL methods
instead model the full return distribution $Z(s,a)$, enabling richer gradient signals
and implicit or explicit sensitivity to tail risk.

This repository implements and compares three agents:

| Agent | Algorithm | Risk Objective |
|---|---|---|
| **DQN** | Deep Q-Network (Mnih et al., 2015) | Mean-greedy (risk-neutral) |
| **IQN** | Implicit Quantile Network (Dabney et al., 2018) | Mean-greedy with implicit risk aversion via distributional loss dynamics |
| **CVaR-RL** | QR-DQN + CVaR action selection (Lim & Malik, 2022) | Explicit CVaR$_\alpha$ risk criterion at deployment |

### DQN
Classic deep Q-network with Double DQN-style target computation and a Huber loss.
Converges to the mean-optimal policy, making it rational under arithmetic expected
return but blind to variance, skewness, and tail risk.

### IQN
Learns the full return distribution by mapping sampled quantile levels
$\tau \sim \mathcal{U}[0,1]$ through a cosine embedding fused with the state
representation. Trained with the asymmetric quantile Huber loss:
$$\rho_\tau(u) = |\tau - \mathbf{1}[u < 0]| \cdot \ell_\kappa(u)$$
Action selection uses the mean of sampled quantiles — identical in principle to DQN —
but the distributional loss creates an implicit bias toward safer policies in
environments with rare catastrophic losses, through simultaneous multi-quantile
gradient updates on tail events.

### CVaR-RL (QR-DQN + CVaR)
Learns the full quantile function using a fixed uniform quantile grid (QR-DQN style).
At action selection, uses Conditional Value-at-Risk:
$$\text{CVaR}_\alpha[Z(s,a)] = \frac{1}{\lfloor \alpha N \rfloor} \sum_{i=1}^{\lfloor \alpha N \rfloor} Q_{\tau_{(i)}}(s,a)$$
where $Q_{\tau_{(i)}}$ are the sorted quantile values. The Bellman bootstrap uses
mean-greedy action selection for stable representation learning, with CVaR applied
only at the policy level. The parameter $\alpha \in (0, 1]$ is a continuous dial:
$\alpha = 1$ recovers mean-greedy (DQN-equivalent), $\alpha \to 0$ gives strongly
risk-averse behaviour focusing on worst-case outcomes.

---

## Environments

### Bimodal Investment (`env.py`)
A two-action episodic environment modelling multiplicative capital growth over 50
steps.

| Parameter | Value |
|---|---|
| Safe return | $+5\%$ always |
| Risky return | $+12\%$ (prob 0.9) or $-50\%$ (prob 0.1) |
| Risky arithmetic EV | $0.058 > 0.05$ (risky is mean-optimal) |
| Risky geometric mean | $1.12^{0.9} \times 0.5^{0.1} \approx 1.028 < 1.05$ (safe is growth-optimal) |
| Terminal bonus | $\log(C_T / 100) / 100$ |

DQN converges to risky (correct by arithmetic EV). IQN and CVaR-RL converge to
safe (correct by geometric/risk-adjusted criterion), demonstrating that
distributional methods implicitly capture the variance drag absent from scalar Q.

### Stochastic Cliff Walking (`train_cliff.py`)
A $4 \times 12$ grid with a cliff along the bottom row. An agent traversing the row
directly above the cliff faces a slip probability of falling onto a random cliff cell
($r = -100$, episode reset). Two natural strategies exist:
- **Short path** (11 steps along cliff edge): mean-optimal but high slip exposure
- **Safe path** (13 steps via upper rows): lower mean return, zero slip exposure

This directly replicates the classic SARSA vs Q-learning divergence
(Sutton & Barto, 2018): DQN gravitates toward the short path; CVaR-RL replicates
SARSA's conservative behaviour through distributional pessimism rather than an
on-policy Bellman correction.

---

## Repository Structure

```
.
├── env.py              # Bimodal Investment Environment (vectorised)
├── dqn.py              # DQN agent (Double DQN, Huber loss)
├── iqn.py              # IQN agent (cosine embedding, quantile Huber loss)
├── cvar_rl.py          # CVaR-RL agent (QR-DQN + CVaR action selection)
├── replay_buffer.py    # Shared replay buffer (numpy-backed, vectorised add)
├── train.py            # Training script: DQN vs IQN vs CVaR on investment env
├── train_cliff.py      # Training script: DQN vs CVaR on cliff walking
├── results/            # Output directory for investment env (auto-created)
└── cliff_results/      # Output directory for cliff walking (auto-created)
```

---

## Dependencies

Requires **Python 3.9.18**. Create and activate a virtual environment:

```bash
python3.9 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib
```

> **Note:** Replace `cu118` with `cu121` or `cpu` depending on your CUDA version.
> To check: `nvidia-smi`. For CPU-only: `pip install torch torchvision`.

Tested versions:
```
torch==2.1.0
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4
```

---

## Implementation Challenges and Their Resolution

Three design problems were identified and corrected:

### (i) Itô leakage in the terminal bonus

A terminal bonus of $\log(C_T / C_0)$ implicitly penalises variance via the Itô correction $E[\log(1 + r_r)] \approx E[r_r] - \tfrac{1}{2}\mathrm{Var}[r_r] = 0.040 < 0.05$, making safe *genuinely* optimal in expected log-return even for DQN and   confounding the comparison; the bonus was scaled by $1/100$ to reduce its influence to the order of a single per-step reward.

### (ii) Exploration contamination

With a slow $\varepsilon$-decay, the replay buffer accumulated far more $-0.50$ outcomes than the converged policy would generate, artificially biasing both agents toward safe during early training; the decay constant was tightened so that near-greedy transitions dominate the buffer before the bulk of gradient updates occur.

### (iii) Capital-scaled reward leakage

Rewarding the agent with $r\_t \times C\_t$ encodes the current wealth level into the reward signal: after risky losses $C\_t$ is small and future rewards are depressed regardless of action, so DQN associates the risky action with low value through a spurious wealth-level correlation rather than through return-distribution reasoning. Using the raw ratio $r\_t$ as the per-step reward removes this leakage entirely.

---

## Running

### Investment Environment — DQN vs IQN vs CVaR-RL

```bash
python train.py --num_envs 32
```

Full options:

```
--n_seeds         INT    Seeds per algorithm             (default: 2)
--total_steps     INT    Environment steps per run       (default: 300000)
--num_envs        INT    Parallel environments           (default: 8)
--episode_length  INT    Steps per episode               (default: 50)
--eval_interval   INT    Steps between evaluations       (default: 5000)
--eval_episodes   INT    Greedy evaluation episodes      (default: 50)
--warmup_steps    INT    Steps before training starts    (default: 2000)
--lr              FLOAT  Learning rate                   (default: 3e-4)
--batch_size      INT    Batch size                      (default: 256)
--gamma           FLOAT  Discount factor                 (default: 0.99)
--hidden          INT    Hidden layer width              (default: 128)
--epsilon_frac    FLOAT  Fraction of steps for eps decay (default: 0.35)
--cvar_alpha      FLOAT  CVaR risk level in (0, 1]       (default: 0.25)
--out_dir         STR    Output directory                (default: ./results)
```

Outputs saved to `./results/`:
- `comparison.png` — 5-panel comparison figure
- `dqn_top2_progress.png`, `iqn_top2_progress.png`, `cvar_top2_progress.png`
- `{label}_seed{n}_log.json` — per-seed training logs

### Cliff Walking — DQN vs CVaR-RL

```bash
python train_cliff.py --num_envs 32
```

Full options:

```
--total_steps     INT    Environment steps per run       (default: 1000000)
--num_envs        INT    Parallel environments           (default: 16)
--eval_interval   INT    Steps between evaluations       (default: 20000)
--eval_episodes   INT    Greedy evaluation episodes      (default: 50)
--slip_prob       FLOAT  Cliff slip probability          (default: 0.2)
--cvar_alpha      FLOAT  CVaR risk level in (0, 1]       (default: 0.5)
--lr              FLOAT  Learning rate                   (default: 3e-4)
--out_dir         STR    Output directory                (default: ./cliff_results)
```

Outputs saved to `./cliff_results/`:
- `cliff_comparison.png` — 4-panel comparison figure (broken y-axis box plot)
- `cliff_logs.json` — full training history
- `cliff.log` — timestamped training log

---

## Key Results

### Bimodal Return Environment
| Agent | Converged Policy | Median Capital | Fall Rate (cliff) |
|---|---|---|---|
| DQN | ~80% risky | \$663 | — |
| IQN | <15% risky | \$1,147 | — |
| CVaR-RL (α=0.25) | <5% risky | \$1,147 | — |

### Cliff Walking (slip\_prob=0.2, α=0.4)
| Agent | Mean Return | Cliff Fall Rate | Cliff Hits / 500 eps |
|---|---|---|---|
| DQN | −32.0 | 0.216 | 108 |
| CVaR-RL | −35.7 | 0.120 | 60 |

CVaR-RL reduces catastrophic cliff falls by **44%** relative to DQN at α=0.4,
at the cost of a slightly lower mean return from taking longer safe paths.

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Bellemare, Dabney & Munos (2017). *A distributional perspective on reinforcement learning.* ICML.
- Dabney et al. (2018). *Distributional reinforcement learning with quantile regression.* AAAI.
- Dabney et al. (2018). *Implicit quantile networks for distributional reinforcement learning.* ICML.
- Lim & Malik (2022). *Distributional reinforcement learning for risk-sensitive policies.* NeurIPS.
- Vincent et al. (2024). *Iterated Q-Network: Beyond one-step Bellman updates.* TMLR.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.

