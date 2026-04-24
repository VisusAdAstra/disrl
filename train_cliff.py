import numpy as np
import torch
import argparse
import logging
import time
import os
import matplotlib.pyplot as plt

from dqn import DQNAgent
from cvar_rl import CVaRAgent


# ─────────────────────────────────────────────────────────────
# Vectorized CliffWalking
# ─────────────────────────────────────────────────────────────

class CliffWalkingVec:
    def __init__(self, n_envs=8, slip_prob=0.05):
        self.n_envs = n_envs
        self.h, self.w = 4, 12
        self.slip_prob = slip_prob

        self.start = np.array([3, 0])
        self.goal  = np.array([3, 11])

        self.pos = np.tile(self.start, (n_envs, 1))

    def reset(self):
        self.pos[:] = self.start
        return self._state()

    def _state(self):
        return np.stack([
            self.pos[:, 0] / self.h,
            self.pos[:, 1] / self.w
        ], axis=1).astype(np.float32)

    def step(self, actions):
        r, c = self.pos[:, 0], self.pos[:, 1]

        # actions
        r = np.where(actions == 0, np.maximum(0, r - 1), r)
        r = np.where(actions == 1, np.minimum(self.h - 1, r + 1), r)
        c = np.where(actions == 2, np.maximum(0, c - 1), c)
        c = np.where(actions == 3, np.minimum(self.w - 1, c + 1), c)

        next_pos = np.stack([r, c], axis=1)

        # slip
        near = (self.pos[:, 0] == 2) & (self.pos[:, 1] >= 1) & (self.pos[:, 1] <= 10)
        slip = (np.random.rand(self.n_envs) < self.slip_prob) & near

        next_pos[slip] = np.stack([
            np.full(slip.sum(), 3),
            np.random.randint(1, 11, size=slip.sum())
        ], axis=1)

        self.pos = next_pos

        rewards = -np.ones(self.n_envs, dtype=np.float32)
        dones   = np.zeros(self.n_envs, dtype=np.float32)

        cliff = (self.pos[:, 0] == 3) & (self.pos[:, 1] >= 1) & (self.pos[:, 1] <= 10)
        goal  = (self.pos[:, 0] == 3) & (self.pos[:, 1] == 11)

        rewards[cliff] = -100.0
        dones[cliff]   = 1.0

        dones[goal] = 1.0

        # reset done envs
        reset_mask = dones == 1.0
        self.pos[reset_mask] = self.start

        return self._state(), rewards, dones, cliff


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate(agent, env, device):
    states = env.reset()
    total_rewards = np.zeros(env.n_envs)
    falls = np.zeros(env.n_envs)

    for _ in range(200):
        st = torch.FloatTensor(states).to(device)

        with torch.no_grad():
            if hasattr(agent, "cvar_alpha"):
                q = agent.online(st)
                actions = agent._cvar_values(q).argmax(1).cpu().numpy()
            else:
                actions = agent.online(st).argmax(1).cpu().numpy()

        states, rewards, dones, cliff = env.step(actions)

        total_rewards += rewards
        falls += cliff.astype(np.float32)

    return np.mean(total_rewards), np.mean(falls)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(agent, env, args, name):
    device = args.device
    states = env.reset()

    log = {
        "steps": [],
        "returns": [],
        "falls": [],
        "loss": [],
    }

    start = time.time()

    for step in range(1, args.total_steps + 1):
        agent.total_steps = step

        st = torch.FloatTensor(states).to(device)
        actions = agent.select_actions(st)

        next_states, rewards, dones, _ = env.step(actions)

        agent.store(states, actions, rewards, next_states, dones)
        loss = agent.update()

        states = next_states

        if step % args.eval_interval == 0:
            ret, fall = evaluate(agent, env, device)

            log["steps"].append(step)
            log["returns"].append(ret)
            log["falls"].append(fall)
            log["loss"].append(loss if loss else 0)

            logging.info(
                f"[{name}] step={step} | return={ret:.2f} | fall={fall:.2f} | "
                f"eps={agent.epsilon:.3f} | loss={loss:.2f} | time={int(time.time()-start)}s"
            )

    return log


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_results(dqn_log, cvar_log, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Return
    plt.figure()
    plt.plot(dqn_log["steps"], dqn_log["returns"], label="DQN")
    plt.plot(cvar_log["steps"], cvar_log["returns"], label="CVaR")
    plt.legend()
    plt.title("Return")
    plt.savefig(f"{out_dir}/return.png")

    # Cliff falls
    plt.figure()
    plt.plot(dqn_log["steps"], dqn_log["falls"], label="DQN")
    plt.plot(cvar_log["steps"], cvar_log["falls"], label="CVaR")
    plt.legend()
    plt.title("Cliff Fall Rate")
    plt.savefig(f"{out_dir}/falls.png")

    # Loss
    plt.figure()
    plt.plot(dqn_log["steps"], dqn_log["loss"], label="DQN")
    plt.plot(cvar_log["steps"], cvar_log["loss"], label="CVaR")
    plt.legend()
    plt.title("Loss")
    plt.savefig(f"{out_dir}/loss.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
#! python train_cliff.py --n_envs 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=300000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cvar_alpha", type=float, default=0.25)
    parser.add_argument("--out_dir", type=str, default="./cliff_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = CliffWalkingVec(n_envs=args.n_envs)

    common = dict(
        state_dim=2,
        n_actions=4,
        lr=args.lr,
        gamma=0.99,
        batch_size=256,
        buffer_size=100000,
        target_update_freq=500,
        hidden=128,
        device=args.device,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=30000,
    )

    dqn = DQNAgent(**common)
    cvar = CVaRAgent(**common, n_quantiles=64, cvar_alpha=args.cvar_alpha)

    logging.info("Training DQN...")
    dqn_log = train(dqn, env, args, "DQN")

    logging.info("Training CVaR...")
    cvar_log = train(cvar, env, args, "CVaR")

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(f"{args.out_dir}/dqn.npz", **dqn_log)
    np.savez(f"{args.out_dir}/cvar.npz", **cvar_log)

    plot_results(dqn_log, cvar_log, args.out_dir)


if __name__ == "__main__":
    main()
