import numpy as np
import torch
import argparse
import logging
import time

from dqn import DQNAgent
from cvar_rl import CVaRAgent


# ─────────────────────────────────────────────────────────────
# CliffWalking Environment (with stochastic cliff risk)
# ─────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    """
    Grid: 4 x 12 (classic)
    Start: (3, 0)
    Goal:  (3, 11)
    Cliff: (3, 1..10)

    Reward:
      -1 per step
      -100 if fall into cliff

    Added stochasticity:
      If agent is adjacent to cliff, with prob p_slip → falls
    """

    def __init__(self, slip_prob=0.05):
        self.h = 4
        self.w = 12
        self.start = (3, 0)
        self.goal  = (3, 11)

        self.slip_prob = slip_prob

        self.reset()

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        # normalize coordinates
        return np.array([self.pos[0] / self.h, self.pos[1] / self.w], dtype=np.float32)

    def step(self, action):
        r, c = self.pos

        # actions: 0=up,1=down,2=left,3=right
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.h - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.w - 1, c + 1)

        next_pos = (r, c)

        # ── stochastic slip near cliff ──
        if self._near_cliff(self.pos) and np.random.rand() < self.slip_prob:
            next_pos = (3, np.random.randint(1, 11))  # fall somewhere in cliff

        self.pos = next_pos

        # ── reward logic ──
        if self.pos[0] == 3 and 1 <= self.pos[1] <= 10:
            # cliff
            reward = -100.0
            done = True
            self.reset()
        elif self.pos == self.goal:
            reward = 0.0
            done = True
            self.reset()
        else:
            reward = -1.0
            done = False

        return self._state(), reward, done

    def _near_cliff(self, pos):
        r, c = pos
        return (r == 2 and 1 <= c <= 10)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(agent, env, episodes=50, device="cpu"):
    returns = []
    cliff_falls = 0

    for _ in range(episodes):
        s = env.reset()
        total = 0

        for _ in range(200):
            st = torch.FloatTensor(s).unsqueeze(0).to(device)

            with torch.no_grad():
                if hasattr(agent, "cvar_alpha"):
                    q = agent.online(st)
                    a = agent._cvar_values(q).argmax(1).item()
                else:
                    a = agent.online(st).argmax(1).item()

            s, r, d = env.step(a)
            total += r

            if r == -100:
                cliff_falls += 1

            if d:
                break

        returns.append(total)

    return np.mean(returns), cliff_falls / episodes


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(agent, env, args, name):
    device = args.device
    s = env.reset()

    start = time.time()

    for step in range(1, args.total_steps + 1):
        agent.total_steps = step

        st = torch.FloatTensor(s).unsqueeze(0).to(device)
        a = agent.select_actions(st)[0]

        ns, r, d = env.step(a)

        agent.store(
            np.array([s]),
            np.array([a]),
            np.array([r], dtype=np.float32),
            np.array([ns]),
            np.array([d], dtype=np.float32),
        )

        loss = agent.update()

        s = ns if not d else env.reset()

        if step % args.eval_interval == 0:
            ret, fall_rate = evaluate(agent, env, device=device)

            logging.info(
                f"[{name}] step={step:6d} | return={ret:7.2f} | "
                f"cliff_fall={fall_rate:.2f} | eps={agent.epsilon:.3f} | "
                f"elapsed={int(time.time()-start)}s"
            )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=150_000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cvar_alpha", type=float, default=0.25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env = CliffWalkingEnv(slip_prob=0.05)

    state_dim = 2
    n_actions = 4

    common = dict(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=256,
        buffer_size=100_000,
        target_update_freq=500,
        hidden=128,
        device=args.device,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=30000,
    )

    # ── DQN ─────────────────────────
    dqn = DQNAgent(**common)
    logging.info("=== Training DQN ===")
    train(dqn, env, args, "DQN")

    # ── CVaR ───────────────────────
    cvar = CVaRAgent(**common, n_quantiles=64, cvar_alpha=args.cvar_alpha)
    logging.info(f"=== Training CVaR (alpha={args.cvar_alpha}) ===")
    train(cvar, env, args, "CVaR")


if __name__ == "__main__":
    main()
