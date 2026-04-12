"""
Training: DQN vs IQN on Bimodal Investment Environment.

Usage: python train.py [options]

  --n_seeds          INT    Seeds per algorithm          (default: 4)
  --total_steps      INT    Env steps per run            (default: 200000)
  --n_envs           INT    Parallel environments        (default: 8)
  --episode_length   INT    Steps per episode            (default: 50)
  --eval_interval    INT    Steps between evaluations    (default: 5000)
  --eval_episodes    INT    Greedy eval episodes         (default: 30)
  --warmup_steps     INT    Steps before training starts (default: 2000)
  --lr               FLOAT  Learning rate                (default: 3e-4)
  --batch_size       INT    Batch size                   (default: 256)
  --gamma            FLOAT  Discount factor              (default: 0.99)
  --hidden           INT    Hidden layer width           (default: 128)
  --epsilon_frac     FLOAT  Fraction of total_steps for eps decay (default: 0.3)
  --capital_ylim     FLOAT  Y-axis upper limit for capital plots  (default: 3000)
  --out_dir          STR    Output directory             (default: ./results)

Reward design:
  Flat per-step reward = raw return rate (0.10 safe, +0.35/-0.15 risky)
  + terminal bonus = log(final_capital / 100)
  Per-step EV is identical (0.10) → DQN indifferent.
  Terminal bonus favors safe compounding → IQN can learn this.

Theoretical optima (episode_length=100):
  Always safe:  $100 * 1.10^100  = $1,378,061  (terminal bonus ~9.53)
  Always risky (geometric): $100 * (1.35*0.85)^50 ≈ $87,025  (terminal bonus ~6.77)
"""

import argparse, os, json, time, logging
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from env import BimodalInvestmentEnv
from dqn import DQNAgent
from iqn import IQNAgent

PALETTE = {"DQN": "#E05C5C", "IQN": "#4A9EE0"}
SAFE_CAPITAL   = 100 * 1.10**50             # $11,739 — always-safe baseline
RISKY_CAPITAL  = 100 * (1.35 * 0.85)**25     # ~$3,178   — always-risky geometric mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds",        type=int,   default=4)
    p.add_argument("--total_steps",    type=int,   default=300_000)
    p.add_argument("--n_envs",         type=int,   default=8)
    p.add_argument("--episode_length", type=int,   default=50)
    p.add_argument("--eval_interval",  type=int,   default=5_000)
    p.add_argument("--eval_episodes",  type=int,   default=50)
    p.add_argument("--warmup_steps",   type=int,   default=2_000)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--epsilon_frac",   type=float, default=0.35)
    p.add_argument("--capital_ylim",   type=float, default=2_000_000.0)
    p.add_argument("--out_dir",        type=str,   default="./results")
    return p.parse_args()


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def evaluate(agent, args, device):
    """
    Greedy evaluation: manually step episodes (no env auto-reset side-effects).
    Returns (mean_cumulative_pct_reward, mean_final_capital, mean_risky_fraction).
    """
    ep_rewards   = []
    ep_capitals  = []
    ep_risky_frac = []

    for _ in range(args.eval_episodes):
        capital    = 100.0
        ep_reward  = 0.0
        risky_steps = 0

        for t in range(args.episode_length):
            step_frac = t / args.episode_length
            state = torch.FloatTensor(
                [[np.log(capital / 100.0), step_frac]]
            ).to(device)

            with torch.no_grad():
                if hasattr(agent, "n_quantiles_policy"):
                    qv, _ = agent.online(state, agent.n_quantiles_policy)
                    action = qv.mean(1).argmax(1).item()
                else:
                    action = agent.online(state).argmax(1).item()

            if action == 0:
                ret = 0.05
            else:
                ret = 0.12 if np.random.random() < 0.50 else -0.5
                risky_steps += 1

            ep_reward += ret                    # flat per-step reward
            capital   *= (1 + ret)

        ep_reward += np.log(capital / 100.0)    # terminal bonus
        ep_rewards.append(ep_reward)
        ep_capitals.append(capital)
        ep_risky_frac.append(risky_steps / args.episode_length)

    return (float(np.mean(ep_rewards)),
            float(np.mean(ep_capitals)),
            float(np.mean(ep_risky_frac)))


def train_run(agent_class, agent_kwargs, seed, label, args, device, log):
    log.info(f"Starting {label} seed={seed}")
    agent = agent_class(seed=seed, **agent_kwargs)
    env   = BimodalInvestmentEnv(
        n_envs=args.n_envs, episode_length=args.episode_length, device=device
    )
    state = env.reset()

    eval_steps     = []
    eval_rewards   = []
    eval_capitals  = []
    eval_risky_frac = []
    recent_losses  = deque(maxlen=200)
    step = 0
    t0   = time.time()

    while step < args.total_steps:
        actions                    = agent.select_actions(state)
        next_state, rewards, dones = env.step(actions)
        agent.store(
            state.cpu().numpy(), actions,
            rewards.cpu().numpy(), next_state.cpu().numpy(),
            dones.cpu().numpy(),
        )
        state = next_state
        step += args.n_envs
        agent.total_steps = step

        if step >= args.warmup_steps:
            loss = agent.update()
            if loss is not None:
                recent_losses.append(loss)

        if step % args.eval_interval < args.n_envs:
            avg_rew, avg_cap, risky_frac = evaluate(agent, args, device)
            eval_steps.append(step)
            eval_rewards.append(avg_rew)
            eval_capitals.append(avg_cap)
            eval_risky_frac.append(risky_frac)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log.info(
                f"[{label} s{seed}] step={step:7d} | "
                f"capital=${avg_cap:8.2f} | risky={risky_frac:.2f} | "
                f"loss={avg_loss:.5f} | eps={agent.epsilon:.3f} | "
                f"elapsed={time.time()-t0:.0f}s"
            )

    with open(os.path.join(args.out_dir, f"{label}_seed{seed}_log.json"), "w") as f:
        json.dump({"eval_steps": eval_steps, "eval_rewards": eval_rewards,
                   "eval_capitals": eval_capitals, "eval_risky_frac": eval_risky_frac}, f)

    return {
        "label":          label,
        "seed":           seed,
        "eval_steps":     eval_steps,
        "eval_rewards":   eval_rewards,
        "eval_capitals":  eval_capitals,
        "eval_risky_frac": eval_risky_frac,
        "final_capital":  eval_capitals[-1]  if eval_capitals  else None,
        "final_reward":   eval_rewards[-1]   if eval_rewards   else None,
        "final_risky":    eval_risky_frac[-1] if eval_risky_frac else None,
    }


def smooth(arr, window=7):
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    pad    = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(arr)]


def plot_top2_progress(results, algo, args):
    runs = sorted(
        [r for r in results if r["label"] == algo],
        key=lambda r: r["final_capital"] if r["final_capital"] is not None else -np.inf,
        reverse=True,
    )[:2]
    if not runs:
        return

    color = PALETTE[algo]
    fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 5), squeeze=False)
    fig.suptitle(f"{algo} — Top {len(runs)} Run(s) by Final Capital", fontsize=14, fontweight="bold")

    for ax, run in zip(axes[0], runs):
        steps   = np.array(run["eval_steps"])
        caps    = smooth(run["eval_capitals"])
        risky   = smooth(run["eval_risky_frac"])

        ax2 = ax.twinx()
        l2, = ax2.plot(steps, risky * 100, color="gray", alpha=0.5, lw=1.5,
                       linestyle=":", label="% Risky Steps")
        l1, = ax.plot(steps, caps, color=color, lw=2.2, label="Final Capital ($)")

        # Reference lines
        ax.axhline(RISKY_CAPITAL, color="green", lw=1, linestyle="--", alpha=0.5, label=f"Always Risky ${RISKY_CAPITAL:.0f}")
        ax.axhline(SAFE_CAPITAL,  color="orange", lw=1, linestyle="--", alpha=0.5, label=f"Always Safe ${SAFE_CAPITAL:.0f}")

        ax2.set_ylim(0, 110)
        ax2.set_ylabel("% Risky Steps")
        ax.set_ylim(0, args.capital_ylim)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Final Capital ($)")
        ax.set_title(f"Seed {run['seed']}  |  Final Capital: ${run['final_capital']:.0f}  |  Risky: {run['final_risky']*100:.0f}%",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        all_lines = [l1, l2]
        ax.legend(all_lines + ax.get_lines()[1:],
                  [l.get_label() for l in all_lines] + [l.get_label() for l in ax.get_lines()[1:]],
                  loc="upper left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(args.out_dir, f"{algo.lower()}_top2_progress.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison(all_results, args):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    ax_cap   = fig.add_subplot(gs[0, :])   # capital curves
    ax_risky = fig.add_subplot(gs[1, :])   # risky fraction curves
    ax_sc_c  = fig.add_subplot(gs[2, 0])   # scatter final capital
    ax_sc_r  = fig.add_subplot(gs[2, 1])   # scatter risky fraction

    for label in ["DQN", "IQN"]:
        runs  = [r for r in all_results if r["label"] == label]
        color = PALETTE[label]
        ref_len = min(len(r["eval_capitals"]) for r in runs)
        if ref_len == 0:
            continue
        ref_steps = np.array(runs[0]["eval_steps"][:ref_len])

        # Capital curves
        mat_cap = np.array([r["eval_capitals"][:ref_len] for r in runs])
        m, s    = smooth(mat_cap.mean(0)), mat_cap.std(0)
        ax_cap.plot(ref_steps, m, color=color, lw=2.5, label=label)
        ax_cap.fill_between(ref_steps, m - s, m + s, color=color, alpha=0.15)

        # Risky fraction curves
        mat_r = np.array([r["eval_risky_frac"][:ref_len] for r in runs])
        mr, sr = smooth(mat_r.mean(0) * 100), mat_r.std(0) * 100
        ax_risky.plot(ref_steps, mr, color=color, lw=2.5, label=label)
        ax_risky.fill_between(ref_steps, mr - sr, mr + sr, color=color, alpha=0.15)

        # Scatter
        seeds = [r["seed"] for r in runs]
        caps  = [r["final_capital"] for r in runs if r["final_capital"] is not None]
        riskys= [r["final_risky"]*100 for r in runs if r["final_risky"] is not None]
        x_off = 0.0 if label == "DQN" else 0.15
        ax_sc_c.scatter([s + x_off for s in seeds], caps, color=color, s=80, zorder=3, label=label)
        if caps:
            ax_sc_c.hlines(np.mean(caps), -0.5, max(seeds)+0.7,
                           colors=color, linestyles="--", lw=1.5, alpha=0.6)
        ax_sc_r.scatter([s + x_off for s in seeds], riskys, color=color, s=80, zorder=3, label=label)
        if riskys:
            ax_sc_r.hlines(np.mean(riskys), -0.5, max(seeds)+0.7,
                           colors=color, linestyles="--", lw=1.5, alpha=0.6)

    # Reference lines on capital plot
    ax_cap.axhline(RISKY_CAPITAL, color="green", lw=1.5, linestyle="--", alpha=0.7,
                   label=f"Always Risky ${RISKY_CAPITAL:.0f} (optimal)")
    ax_cap.axhline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--", alpha=0.7,
                   label=f"Always Safe ${SAFE_CAPITAL:.0f} (suboptimal)")
    ax_risky.axhline(100, color="green", lw=1, linestyle="--", alpha=0.5, label="Always Risky")
    ax_risky.axhline(0,   color="orange", lw=1, linestyle="--", alpha=0.5, label="Always Safe")

    ax_cap.set_title("Final Capital over Training (mean ± std)", fontsize=13, fontweight="bold")
    ax_cap.set_xlabel("Environment Steps"); ax_cap.set_ylabel("Final Capital ($)")
    ax_cap.set_ylim(0, args.capital_ylim); ax_cap.legend(fontsize=10); ax_cap.grid(True, alpha=0.3)

    ax_risky.set_title("% Risky Steps Chosen by Greedy Policy (mean ± std)", fontsize=13, fontweight="bold")
    ax_risky.set_xlabel("Environment Steps"); ax_risky.set_ylabel("% Risky Steps")
    ax_risky.set_ylim(-5, 115); ax_risky.legend(fontsize=10); ax_risky.grid(True, alpha=0.3)

    ax_sc_c.set_title("Final Capital by Seed", fontsize=11, fontweight="bold")
    ax_sc_c.set_ylim(0, args.capital_ylim); ax_sc_c.set_xlabel("Seed"); ax_sc_c.set_ylabel("Capital ($)")
    ax_sc_c.axhline(RISKY_CAPITAL, color="green", lw=1, linestyle="--", alpha=0.5)
    ax_sc_c.axhline(SAFE_CAPITAL,  color="orange", lw=1, linestyle="--", alpha=0.5)
    ax_sc_c.legend(fontsize=10); ax_sc_c.grid(True, alpha=0.3, axis="y")

    ax_sc_r.set_title("Final % Risky Steps by Seed", fontsize=11, fontweight="bold")
    ax_sc_r.set_ylim(-5, 115); ax_sc_r.set_xlabel("Seed"); ax_sc_r.set_ylabel("% Risky")
    ax_sc_r.legend(fontsize=10); ax_sc_r.grid(True, alpha=0.3, axis="y")

    for label in ["DQN", "IQN"]:
        runs  = [r for r in all_results if r["label"] == label]
        caps  = [r["final_capital"] for r in runs if r["final_capital"]]
        risky = [r["final_risky"]*100 for r in runs if r["final_risky"] is not None]
        if caps:
            print(f"{label}: mean_capital=${np.mean(caps):.0f}  risky={np.mean(risky):.1f}%  "
                  f"(safe=${SAFE_CAPITAL:.0f} risky_opt=${RISKY_CAPITAL:.0f})")

    path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def get_modal_strategy(all_results, label):
    """Get most common risky ratio (5% bins) from last 10% of eval checkpoints."""
    fracs = []
    for r in all_results:
        if r["label"] != label:
            continue
        rf = r["eval_risky_frac"]
        n = len(rf)
        fracs.extend(rf[max(0, n - n // 10):])
    if not fracs:
        return 0.5
    bins = np.arange(0, 1.05, 0.05)
    counts, edges = np.histogram(fracs, bins=bins)
    best = counts.argmax()
    return (edges[best] + edges[best + 1]) / 2.0


def simulate_fixed_strategy(risky_ratio, episode_length, n_episodes=20):
    """Run n_episodes with a fixed risky_ratio, return mean final capital."""
    capitals = []
    for _ in range(n_episodes):
        capital = 100.0
        for t in range(episode_length):
            if np.random.random() < risky_ratio:
                ret = 0.35 if np.random.random() < 0.50 else -0.15
            else:
                ret = 0.10
            capital *= (1 + ret)
        capitals.append(capital)
    return float(np.mean(capitals))


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log    = setup_logging(args.out_dir)

    epsilon_decay = int(args.epsilon_frac * args.total_steps / np.log(20))
    log.info(f"Device: {device} | epsilon reaches 0.05 at step ~{int(args.epsilon_frac*args.total_steps):,} | args: {vars(args)}")
    log.info(f"Theoretical optima: safe=${SAFE_CAPITAL:.0f}  always-risky=${RISKY_CAPITAL:.0f}")

    shared = dict(
        state_dim=2, n_actions=2,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=1.0, epsilon_end=0.02,
        epsilon_decay=epsilon_decay,
        batch_size=args.batch_size, buffer_size=100_000,
        target_update_freq=500, hidden=args.hidden, device=device,
    )
    iqn_kwargs = {
        **shared,
        "n_quantiles":        16,
        "n_quantiles_target": 16,
        "n_quantiles_policy": 64,
        "state_emb_dim":      256,
    }

    all_results = []
    for seed in range(args.n_seeds):
        all_results.append(train_run(DQNAgent, shared, seed, "DQN", args, device, log))
    for seed in range(args.n_seeds):
        all_results.append(train_run(IQNAgent, iqn_kwargs, seed, "IQN", args, device, log))



    plot_top2_progress(all_results, "DQN", args)
    plot_top2_progress(all_results, "IQN", args)
    plot_comparison(all_results, args)

    # Extract modal strategies and simulate
    log.info("=== Fixed-Strategy Comparison ===")
    for label in ["DQN", "IQN"]:
        ratio = get_modal_strategy(all_results, label)
        avg_cap = simulate_fixed_strategy(ratio, args.episode_length, n_episodes=20)
        log.info(f"{label}: modal risky ratio={ratio:.0%} (bin center) | "
                 f"20-episode avg capital=${avg_cap:,.0f}")

    log.info("Done. Results in " + args.out_dir)


if __name__ == "__main__":
    main()
