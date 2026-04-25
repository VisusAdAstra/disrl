import numpy as np
import torch
import argparse
import logging
import time
import os
import json
import matplotlib.pyplot as plt

from dqn import DQNAgent
from cvar_rl import CVaRAgent

# ─────────────────────────────────────────────────────────────
# CliffWalking Environment
# ─────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    def __init__(self, slip_prob=0.05):
        self.h, self.w = 4, 12
        self.start, self.goal = (3, 0), (3, 11)
        self.slip_prob = slip_prob
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        return np.array([self.pos[0] / self.h, self.pos[1] / self.w], dtype=np.float32)

    def step(self, action):
        r, c = self.pos
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.h - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.w - 1, c + 1)

        if (r == 2 and 1 <= c <= 10) and np.random.rand() < self.slip_prob:
            r, c = 3, np.random.randint(1, 11)

        self.pos = (r, c)

        if self.pos[0] == 3 and 1 <= self.pos[1] <= 10:
            return self.reset(), -100.0, True
        elif self.pos == self.goal:
            return self.reset(), 0.0, True
        return self._state(), -1.0, False

class VectorCliffEnv:
    def __init__(self, num_envs, slip_prob=0.05):
        self.envs = [CliffWalkingEnv(slip_prob) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        return np.array([e.reset() for e in self.envs])

    def step(self, actions):
        results = [e.step(a) for e, a in zip(self.envs, actions)]
        states, rewards, dones = zip(*results)
        return np.array(states), np.array(rewards), np.array(dones)

# ─────────────────────────────────────────────────────────────
# Evaluation & Plotting
# ─────────────────────────────────────────────────────────────

def evaluate(agent, env, episodes=20, device="cpu"):
    returns, falls = [], 0
    for _ in range(episodes):
        s = env.reset()
        total_r = 0
        for _ in range(200):
            st = torch.FloatTensor(s).unsqueeze(0).to(device)
            # Use deterministic selection if possible, else standard greedy
            with torch.no_grad():
                if hasattr(agent, "cvar_alpha"):
                    a = agent._cvar_values(agent.online(st)).argmax(1).item()
                else:
                    a = agent.online(st).argmax(1).item()
            s, r, d = env.step(a)
            total_r += r
            if r == -100: falls += 1
            if d: break
        returns.append(total_r)
    return np.mean(returns), falls / episodes

def plot_results(dqn_log, cvar_log, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["returns", "falls", "loss"]
    titles = ["Return", "Cliff Fall Rate", "Loss"]
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 5))
        plt.plot(dqn_log["steps"], dqn_log[metric], label="DQN")
        plt.plot(cvar_log["steps"], cvar_log[metric], label="CVaR")
        plt.xlabel("Steps")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/{metric}.png")
        plt.close()

# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(agent, venv, eval_env, args, name):
    device = args.device
    states = venv.reset()
    start = time.time()
    
    # Storage for statistics
    history = {"steps": [], "returns": [], "falls": [], "loss": []}
    current_losses = []

    for step in range(0, args.total_steps, args.num_envs):
        agent.total_steps = step
        st = torch.FloatTensor(states).to(device)
        actions = agent.select_actions(st)
        next_states, rewards, dones = venv.step(actions)

        agent.store(states, actions, rewards, next_states, dones)
        loss = agent.update()
        if loss is not None:
            current_losses.append(loss)
        
        states = next_states

        if step % args.eval_interval == 0:
            ret, fall_rate = evaluate(agent, eval_env, device=device)
            avg_loss = np.mean(current_losses) if current_losses else 0
            
            history["steps"].append(step)
            history["returns"].append(ret)
            history["falls"].append(fall_rate)
            history["loss"].append(avg_loss)
            
            current_losses = [] # Reset loss tracking for the next window

            logging.info(
                f"[{name}] step={step:6d} | ret={ret:7.2f} | cliff={fall_rate:.2f} | "
                f"loss={avg_loss:.4f} | eps={agent.epsilon:.3f}"
            )
            
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="cliff_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    venv = VectorCliffEnv(args.num_envs) #, slip_prob=0.05
    eval_env = CliffWalkingEnv()

    common = dict(
        state_dim=2, n_actions=4, lr=3e-4, gamma=0.99,
        batch_size=256, buffer_size=100_000, target_update_freq=1000,
        hidden=128, device=args.device,
        epsilon_start=1.0, epsilon_end=0.02, 
        epsilon_decay=args.total_steps // 5 #6
    )



    # 2. Train CVaR
    logging.info(f"\n=== Training CVaR {args.num_envs} envs ===")
    cvar_log = train(CVaRAgent(**common, n_quantiles=64, cvar_alpha=0.75), venv, eval_env, args, "CVaR")

    # 1. Train DQN
    logging.info(f"=== Training DQN {args.num_envs} envs ===")
    dqn_log = train(DQNAgent(**common), venv, eval_env, args, "DQN")

    # 3. Save Statistics to file
    stats_path = os.path.join(args.out_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump({"dqn": dqn_log, "cvar": cvar_log}, f, indent=4)
    logging.info(f"\nStatistics saved to {stats_path}")

    # 4. Generate Plots
    plot_results(dqn_log, cvar_log, args.out_dir)
    logging.info(f"Plots saved to {args.out_dir}/")

if __name__ == "__main__":
    main()
