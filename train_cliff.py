"""
train_cliff.py — DQN vs CVaR-RL on Stochastic Cliff Walking

Environment fixes over original:
  1. Slip condition fires on PRE-move position (not post-move)
     Original: checked (r,c) AFTER updating position -> fires on wrong cell
     Fixed:    check (old_r, old_c) BEFORE move -> correct slip semantics

  2. Goal reachability: original goal (3,11) was only reachable from (2,11)
     because (3,10) is a cliff cell that resets before agent can step right.
     Fixed:    goal moved to (0,11) top-right corner — reachable from any
               direction without cliff adjacency, making exploration tractable.

  3. State encoding extended to 3 dims: [row/H, col/W, step/max_steps]
     Time signal helps agents distinguish early vs late episode states.

Expected result:
  DQN  -> learns the short cliff-edge path (optimal EV) but falls frequently
  CVaR -> learns the longer safe upper path (lower EV, zero cliff risk)
  This cleanly mirrors the Sutton & Barto cliff-walking SARSA vs Q-learning result.

Usage:
  python train_cliff.py
  python train_cliff.py --total_steps 300000 --num_envs 16 --cvar_alpha 0.25
"""

import numpy as np
import torch
import argparse
import logging
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from dqn import DQNAgent
from cvar_rl import CVaRAgent

log = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {"DQN": "#E05C5C", "CVaR": "#4CAF50"}

# ── Environment ───────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    """
    Stochastic Cliff Walking (4x12 grid).

    Layout (row 3 = bottom):
      Row 0: ....................................G   <- goal at (0,11)
      Row 1: ....................................
      Row 2: ....................................
      Row 3: S  C  C  C  C  C  C  C  C  C  C  .
             0  1  2  3  4  5  6  7  8  9 10 11

    S = start (3,0), G = goal (0,11)
    C = cliff cells (3, 1-10): immediate reset + reward -100

    Stochastic slip: when agent is about to leave a row-2 cell
    (the row immediately above the cliff), there is slip_prob chance
    of being pushed down into a random cliff cell instead.
    Slip is checked on the PRE-MOVE position to correctly model
    the hazard of traversing the dangerous zone above the cliff.

    Two natural strategies:
      Short path (DQN preferred): hug row 3, step right 11 times then up.
        Fast but risky — slip_prob exposure each step in row 2.
      Safe path (CVaR preferred): go up to row 0-1, traverse right, come down.
        Longer (-13 vs -11 steps baseline) but zero cliff exposure.
    """

    # Actions
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, slip_prob: float = 0.1, max_steps: int = 99):
        self.h, self.w     = 4, 12
        self.start         = (3, 0)
        self.goal          = (0, 11)   # top-right: reachable without cliff adjacency
        self.slip_prob     = slip_prob
        self.max_steps     = max_steps
        self.state_dim     = 3         # [row/H, col/W, step/max_steps]
        self.n_actions     = 4
        self.reset()

    def reset(self):
        self.pos   = self.start
        self.steps = 0
        return self._obs()

    def _obs(self):
        r, c = self.pos
        return np.array(
            [r / (self.h - 1), c / (self.w - 1), self.steps / self.max_steps],
            dtype=np.float32,
        )

    def _is_cliff(self, r, c):
        return r == 3 and 1 <= c <= 10

    def step(self, action: int):
        r, c = self.pos

        # ── Slip check on PRE-MOVE position ──────────────────────────────────
        # If currently in row 2 (directly above cliff) and attempting to move
        # down, there is slip_prob chance of landing on a random cliff cell.
        slipped = False
        if r == 2 and (c != 0 or c != 11) and np.random.rand() < self.slip_prob: #and action != self.UP and action != self.DOWN
            # Slip: land on random cliff cell in row 3
            new_r, new_c = 3, np.random.randint(1, 11)
            slipped = True
        else:
            # Normal move
            if   action == self.UP:    new_r, new_c = max(0, r - 1), c
            elif action == self.DOWN:  new_r, new_c = min(self.h - 1, r + 1), c
            elif action == self.LEFT:  new_r, new_c = r, max(0, c - 1)
            elif action == self.RIGHT: new_r, new_c = r, min(self.w - 1, c + 1)
            else:                      new_r, new_c = r, c

        self.pos    = (new_r, new_c)
        self.steps += 1

        # ── Terminal conditions ───────────────────────────────────────────────
        if self._is_cliff(new_r, new_c):
            obs = self.reset()
            return obs, -100.0, True

        if self.pos == self.goal:
            obs = self.reset()
            return obs, 0.0, True

        if self.steps >= self.max_steps:
            obs = self.reset()
            return obs, -1.0, True

        return self._obs(), -1.0, False


class VectorCliffEnv:
    """Vectorised wrapper: runs num_envs independent CliffWalkingEnv in parallel."""

    def __init__(self, num_envs: int, slip_prob: float = 0.1, max_steps: int = 99):
        self.envs     = [CliffWalkingEnv(slip_prob, max_steps) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        return np.array([e.reset() for e in self.envs], dtype=np.float32)

    def step(self, actions):
        results              = [e.step(int(a)) for e, a in zip(self.envs, actions)]
        states, rewards, dones = zip(*results)
        return (
            np.array(states,  dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones,   dtype=bool),
        )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, slip_prob: float, episodes: int, device: str):
    """
    Greedy evaluation. Returns:
      mean_return  — mean episode return
      fall_rate    — fraction of episodes with at least one cliff fall
      cliff_hits   — total cliff falls across all episodes
      ep_returns   — list of per-episode returns (for distribution)
    """
    env        = CliffWalkingEnv(slip_prob=slip_prob)
    ep_returns = []
    cliff_hits = 0
    fell_eps   = 0

    for _ in range(episodes):
        s        = env.reset()
        total_r  = 0.0
        fell     = False

        for _ in range(env.max_steps):
            st = torch.FloatTensor(s).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(agent, "cvar_alpha"):
                    a = agent._cvar_values(agent.online(st)).argmax(1).item()
                else:
                    a = agent.online(st).argmax(1).item()

            s, r, d = env.step(a)
            total_r += r
            if r == -100.0:
                cliff_hits += 1
                fell = True
            if d:
                break

        ep_returns.append(total_r)
        if fell:
            fell_eps += 1

    return (
        float(np.mean(ep_returns)),
        fell_eps / episodes,
        cliff_hits,
        ep_returns,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(agent, venv, args, label, log):
    log.info(f"=== Training {label} ===")
    states = venv.reset()

    history = {
        "steps":       [],
        "returns":     [],
        "fall_rate":   [],
        "cliff_hits":  [],
        "loss":        [],
        "ep_returns":  [],   # full distribution at each checkpoint
    }
    loss_buf = deque(maxlen=200)

    for step in range(0, args.total_steps+1, args.num_envs):
        # if isinstance(agent, CVaRAgent):
        #     agent.lr = args.lr/2
        agent.total_steps = step

        st      = torch.FloatTensor(states).to(args.device)
        actions = agent.select_actions(st)
        next_states, rewards, dones = venv.step(actions)

        agent.store(states, actions, rewards, next_states, dones.astype(np.float32))
        loss = agent.update()
        if loss is not None:
            loss_buf.append(loss)

        states = next_states

        if step % args.eval_interval == 0:
            mean_ret, fall_rate, cliff_hits, ep_rets = evaluate(
                agent, args.slip_prob, args.eval_episodes, args.device
            )
            avg_loss = float(np.mean(loss_buf)) if loss_buf else 0.0

            history["steps"].append(step)
            history["returns"].append(mean_ret)
            history["fall_rate"].append(fall_rate)
            history["cliff_hits"].append(cliff_hits)
            history["loss"].append(avg_loss)
            history["ep_returns"].append(ep_rets)

            log.info(
                f"[{label}] step={step:7d} | "
                f"mean_ret={mean_ret:7.2f} | median_ret={float(np.median(ep_rets)):7.2f} | "
                f"falls={fall_rate:.2f} | hits={cliff_hits:3d} | "
                f"loss={avg_loss:.4f} | eps={agent.epsilon:.3f}"
            )

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def smooth(arr, w=5):
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def plot_results(logs: dict, args):
    """
    4-panel comparison figure:
      Top-L:   Episode return over training
      Top-R:   Cliff fall rate over training
      Bot-L:   Return distribution (last 20% of training) — box plot
      Bot-R:   Cliff hits histogram (last 20%)
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    ax_ret   = fig.add_subplot(gs[0, 0])
    ax_fall  = fig.add_subplot(gs[0, 1])
    ax_box   = fig.add_subplot(gs[1, 0])
    ax_hist  = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"DQN vs CVaR-RL — Stochastic Cliff Walking "
        f"(slip_prob={args.slip_prob}, CVaR α={args.cvar_alpha})",
        fontsize=14, fontweight="bold",
    )

    for label, h in logs.items():
        color  = PALETTE[label]
        steps  = np.array(h["steps"])

        # ── Return curve ──────────────────────────────────────────────────────
        ax_ret.plot(steps, smooth(h["returns"]), color=color, lw=2, label=label)
        ax_ret.fill_between(
            steps,
            smooth(h["returns"]) - 5,
            smooth(h["returns"]) + 5,
            color=color, alpha=0.12,
        )

        # ── Fall rate curve ───────────────────────────────────────────────────
        ax_fall.plot(steps, smooth(h["fall_rate"]), color=color, lw=2, label=label)

    # Optimal return reference lines
    ax_ret.axhline(-13, color="gray", lw=1, linestyle="--", alpha=0.7,
                   label="Optimal safe path (−13)")
    ax_ret.axhline(-11, color="black", lw=1, linestyle=":", alpha=0.7,
                   label="Optimal risky path (−11)")

    ax_ret.set_xlabel("Environment Steps")
    ax_ret.set_ylabel("Mean Episode Return")
    ax_ret.set_title("Return over Training", fontweight="bold")
    ax_ret.legend(fontsize=9)
    ax_ret.grid(True, alpha=0.3)

    ax_fall.set_xlabel("Environment Steps")
    ax_fall.set_ylabel("Cliff Fall Rate (fraction of episodes)")
    ax_fall.set_title("Cliff Fall Rate over Training", fontweight="bold")
    ax_fall.set_ylim(-0.05, 1.05)
    ax_fall.legend(fontsize=9)
    ax_fall.grid(True, alpha=0.3)

    # ── Converged return distribution (last 20%) ──────────────────────────────
    # Use broken y-axis: top panel shows path quality (-10 to -55),
    # bottom panel shows cliff falls (-88 to -112).
    # This prevents -100 cliff penalty from compressing the path distribution,
    # which caused DQN to appear as a single point in the original plot.
    ax_box.remove()

    gs2 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 0], height_ratios=[3, 1], hspace=0.08
    )
    ax_top = fig.add_subplot(gs2[0])
    ax_bot = fig.add_subplot(gs2[1])

    box_data, box_labels, box_colors = [], [], []
    for label, h in logs.items():
        n_conv = max(1, len(h["ep_returns"]) // 5)
        flat   = [r for ep in h["ep_returns"][-n_conv:] for r in ep]
        box_data.append(flat)
        box_labels.append(label)
        box_colors.append(PALETTE[label])

    for ax in [ax_top, ax_bot]:
        bp = ax.boxplot(
            box_data, patch_artist=True,
            medianprops=dict(color="white", lw=2.5),
            whiskerprops=dict(lw=1.3), capprops=dict(lw=1.3),
            flierprops=dict(marker=".", markersize=4, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)

    # Top: path quality region
    ax_top.set_ylim(-55, -8)
    ax_top.axhline(-13, color="gray",  lw=1.2, linestyle="--",
                   alpha=0.8, label="Safe path optimal (−13)")
    ax_top.axhline(-11, color="black", lw=1.2, linestyle=":",
                   alpha=0.8, label="Risky path optimal (−11)")
    ax_top.set_xticklabels([])
    ax_top.legend(fontsize=8, loc="lower right")
    ax_top.grid(True, alpha=0.3, axis="y")
    ax_top.set_ylabel("Episode Return")
    ax_top.set_title("Converged Return Distribution\n(last 20% of training)",
                     fontweight="bold")
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(bottom=False)

    # Bottom: cliff fall region
    ax_bot.set_ylim(-112, -88)
    ax_bot.axhline(-100, color="red", lw=1.2, linestyle="-.",
                   alpha=0.6, label="Cliff fall (−100)")
    ax_bot.set_xticklabels(box_labels, fontsize=11)
    ax_bot.legend(fontsize=8, loc="upper right")
    ax_bot.grid(True, alpha=0.3, axis="y")
    ax_bot.set_ylabel("Return")
    ax_bot.spines["top"].set_visible(False)

    # Broken axis diagonal marks
    d = 0.015
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw2 = dict(transform=ax_bot.transAxes, color="k", clip_on=False, lw=1.2)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw2)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw2)

    # ── Cliff hits histogram ──────────────────────────────────────────────────
    for label, h in logs.items():
        n_conv     = max(1, len(h["cliff_hits"]) // 5)
        conv_hits  = h["cliff_hits"][-n_conv:]
        color      = PALETTE[label]
        ax_hist.plot(
            np.array(h["steps"][-n_conv:]),
            conv_hits,
            color=color, lw=2, label=label, alpha=0.8,
        )

    ax_hist.set_xlabel("Environment Steps")
    ax_hist.set_ylabel("Cliff Hits per Eval Window")
    ax_hist.set_title("Cliff Hits — Converged Phase\n(last 20% of training)",
                      fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # Stats summary
    stats_lines = []
    for label, h in logs.items():
        n_conv    = max(1, len(h["ep_returns"]) // 5)
        flat      = [r for ep in h["ep_returns"][-n_conv:] for r in ep]
        tot_falls = sum(h["cliff_hits"][-n_conv:])
        n_eps_conv  = n_conv * args.eval_episodes
        fall_rate_c = tot_falls / n_eps_conv if n_eps_conv > 0 else 0.0
        stats_lines.append(
            f"{label}: mean={np.mean(flat):.1f}  median={np.median(flat):.1f}  "
            f"p10={np.percentile(flat,10):.1f}  "
            f"fall_rate={fall_rate_c:.3f}  cliff_hits={tot_falls}/{n_eps_conv}"
        )
    fig.text(
        0.5, 0.01, "   |   ".join(stats_lines),
        ha="center", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
    )

    path = os.path.join(args.out_dir, "cliff_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to {path}")
    for line in stats_lines:
        log.info(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps",   type=int,   default=1_000_000)
    parser.add_argument("--num_envs",      type=int,   default=16)
    parser.add_argument("--eval_interval", type=int,   default=5_000)
    parser.add_argument("--eval_episodes", type=int,   default=50)
    parser.add_argument("--slip_prob",     type=float, default=0.2)
    parser.add_argument("--cvar_alpha",    type=float, default=0.5)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--out_dir",       type=str,   default="cliff_results")
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "cliff.log")),
            logging.StreamHandler(),
        ],
    )
    log.info(f"Device: {args.device} | slip_prob={args.slip_prob} | "
             f"cvar_alpha={args.cvar_alpha} | num_envs={args.num_envs} | total_steps={args.total_steps}")

    common = dict(
        state_dim=3,           # [row/H, col/W, step/max_steps]
        n_actions=4,
        lr=args.lr,
        gamma=0.99,
        batch_size=1024,        # 256
        buffer_size=2_000_000,  # 100_000
        target_update_freq=1_000,
        hidden=128,
        device=args.device,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay=args.total_steps // 8,
    )

    venv = VectorCliffEnv(args.num_envs, slip_prob=args.slip_prob)

    logs = {}

    # Train CVaR first (resets buffer state for fair comparison)
    cvar_agent = CVaRAgent(**common, n_quantiles=64, cvar_alpha=args.cvar_alpha)
    logs["CVaR"] = train(cvar_agent, venv, args, "CVaR", log)

    # Reset vectorised env for DQN
    venv = VectorCliffEnv(args.num_envs, slip_prob=args.slip_prob)

    dqn_agent = DQNAgent(**common)
    logs["DQN"] = train(dqn_agent, venv, args, "DQN", log)

    # Save raw logs
    json_path = os.path.join(args.out_dir, "cliff_logs.json")
    # ep_returns nested lists are large — save summaries only
    save_logs = {}
    for label, h in logs.items():
        save_logs[label] = {k: v for k, v in h.items() if k != "ep_returns"}
    with open(json_path, "w") as f:
        json.dump(save_logs, f, indent=2)

    plot_results(logs, args)
    log.info("Done.")


if __name__ == "__main__":
    main()
