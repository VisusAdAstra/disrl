"""
Training: DQN vs IQN on Bimodal Investment Environment.

Usage:
  python train.py [options]

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
  --epsilon_frac     FLOAT  Fraction of total_steps for epsilon decay (default: 0.5)
  --capital_ylim     FLOAT  Y-axis upper limit for capital plots (default: 1000)
  --out_dir          STR    Output directory             (default: ./results)
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds",        type=int,   default=4)
    p.add_argument("--total_steps",    type=int,   default=200_000)
    p.add_argument("--n_envs",         type=int,   default=8)
    p.add_argument("--episode_length", type=int,   default=50)
    p.add_argument("--eval_interval",  type=int,   default=5_000)
    p.add_argument("--eval_episodes",  type=int,   default=30)
    p.add_argument("--warmup_steps",   type=int,   default=2_000)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--epsilon_frac",   type=float, default=0.5,
                   help="Fraction of total_steps over which epsilon decays 1.0->0.05")
    p.add_argument("--capital_ylim",   type=float, default=1000.0)
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


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(agent, args, device):
    """
    Greedy policy eval: manually step single episode, no env auto-reset side-effects.
    Returns (mean_log_return_sum, mean_final_capital).
    """
    ep_returns  = []
    ep_capitals = []

    for _ in range(args.eval_episodes):
        capital = 100.0
        log_ret_sum = 0.0

        for t in range(args.episode_length):
            step_frac = t / args.episode_length
            state = torch.FloatTensor(
                [[np.log(capital / 100.0), step_frac]]
            ).to(device)

            with torch.no_grad():
                if hasattr(agent, "n_quantiles_policy"):   # IQN
                    qv, _ = agent.online(state, agent.n_quantiles_policy)
                    action = qv.mean(1).argmax(1).item()
                else:                                       # DQN
                    action = agent.online(state).argmax(1).item()

            if action == 0:
                mult = 1.05
            else:
                mult = 1.20 if np.random.random() < 0.55 else 0.90

            capital     *= mult
            log_ret_sum += np.log(mult)

        ep_returns.append(log_ret_sum)
        ep_capitals.append(capital)

    return float(np.mean(ep_returns)), float(np.mean(ep_capitals))


# ── Single training run ────────────────────────────────────────────────────────
def train_run(agent_class, agent_kwargs, seed, label, args, device, log):
    log.info(f"Starting {label} seed={seed}")
    agent = agent_class(seed=seed, **agent_kwargs)
    env   = BimodalInvestmentEnv(
        n_envs=args.n_envs, episode_length=args.episode_length, device=device
    )
    state = env.reset()

    eval_steps    = []
    eval_returns  = []
    eval_capitals = []
    recent_losses = deque(maxlen=200)
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
            avg_ret, avg_cap = evaluate(agent, args, device)
            eval_steps.append(step)
            eval_returns.append(avg_ret)
            eval_capitals.append(avg_cap)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log.info(
                f"[{label} s{seed}] step={step:7d} | "
                f"log_ret={avg_ret:.4f} | capital=${avg_cap:8.2f} | "
                f"loss={avg_loss:.5f} | eps={agent.epsilon:.3f} | "
                f"elapsed={time.time()-t0:.0f}s"
            )

    with open(os.path.join(args.out_dir, f"{label}_seed{seed}_log.json"), "w") as f:
        json.dump({"eval_steps": eval_steps, "eval_returns": eval_returns,
                   "eval_capitals": eval_capitals}, f)

    return {
        "label":         label,
        "seed":          seed,
        "eval_steps":    eval_steps,
        "eval_returns":  eval_returns,
        "eval_capitals": eval_capitals,
        "final_return":  eval_returns[-1]  if eval_returns  else None,
        "final_capital": eval_capitals[-1] if eval_capitals else None,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def smooth(arr, window=5):
    """Edge-preserving smooth: pad with edge values before convolving."""
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    pad    = window // 2
    padded = np.pad(arr, pad, mode="edge")
    out    = np.convolve(padded, np.ones(window) / window, mode="valid")
    return out[:len(arr)]


def plot_top2_progress(results, algo, args):
    runs = sorted(
        [r for r in results if r["label"] == algo],
        key=lambda r: r["final_return"] if r["final_return"] is not None else -np.inf,
        reverse=True,
    )
    top      = runs[:2]
    n_panels = len(top)
    if n_panels == 0:
        return

    color = PALETTE[algo]
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    fig.suptitle(f"{algo} — Top {n_panels} Run(s) by Final Return",
                 fontsize=14, fontweight="bold")

    for ax, run in zip(axes[0], top):
        steps = np.array(run["eval_steps"])
        rets  = smooth(run["eval_returns"])
        caps  = smooth(run["eval_capitals"])

        ax2 = ax.twinx()
        l2, = ax2.plot(steps, caps, color=color, alpha=0.45, lw=1.5,
                       linestyle="--", label="Final Capital ($)")
        l1, = ax.plot(steps, rets, color=color, lw=2.2, label="Log Return")

        ax2.set_ylim(bottom=0, top=args.capital_ylim)
        ax2.set_ylabel("Final Capital ($)")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Cumulative Log Return")
        ax.set_title(f"Seed {run['seed']}  |  Capital: ${run['final_capital']:.1f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()],
                  loc="upper left", fontsize=9)

    plt.tight_layout()
    path = os.path.join(args.out_dir, f"{algo.lower()}_top2_progress.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison(all_results, args):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_ret  = fig.add_subplot(gs[0, :])
    ax_cap  = fig.add_subplot(gs[1, 0])
    ax_ret2 = fig.add_subplot(gs[1, 1])

    for label in ["DQN", "IQN"]:
        runs  = [r for r in all_results if r["label"] == label]
        color = PALETTE[label]

        ref_len = min(len(r["eval_returns"]) for r in runs)
        if ref_len > 0:
            ref_steps = np.array(runs[0]["eval_steps"][:ref_len])
            mat       = np.array([r["eval_returns"][:ref_len] for r in runs])
            mean_ret  = smooth(mat.mean(axis=0))
            std_ret   = mat.std(axis=0)
            ax_ret.plot(ref_steps, mean_ret, color=color, lw=2.5, label=label)
            ax_ret.fill_between(ref_steps,
                                mean_ret - std_ret, mean_ret + std_ret,
                                color=color, alpha=0.15)

        seeds = [r["seed"] for r in runs if r["final_capital"] is not None]
        caps  = [r["final_capital"] for r in runs if r["final_capital"] is not None]
        rets  = [r["final_return"]  for r in runs if r["final_return"]  is not None]
        x_off = 0.0 if label == "DQN" else 0.15

        ax_cap.scatter([s + x_off for s in seeds], caps,
                       color=color, s=80, zorder=3, label=label)
        if caps:
            ax_cap.hlines(np.mean(caps), -0.5, max(seeds) + 0.7,
                          colors=color, linestyles="--", lw=1.5, alpha=0.6)

        ax_ret2.scatter([s + x_off for s in seeds], rets,
                        color=color, s=80, zorder=3, label=label)
        if rets:
            ax_ret2.hlines(np.mean(rets), -0.5, max(seeds) + 0.7,
                           colors=color, linestyles="--", lw=1.5, alpha=0.6)

    ax_ret.set_title("Training Progress: Mean Cumulative Log Return",
                     fontsize=13, fontweight="bold")
    ax_ret.set_xlabel("Environment Steps")
    ax_ret.set_ylabel("Cumulative Log Return")
    ax_ret.legend(fontsize=12)
    ax_ret.grid(True, alpha=0.3)

    ax_cap.set_title("Final Capital by Seed", fontsize=11, fontweight="bold")
    ax_cap.set_xlabel("Seed")
    ax_cap.set_ylabel("Capital ($)")
    ax_cap.set_ylim(bottom=0, top=args.capital_ylim)
    ax_cap.legend(fontsize=10)
    ax_cap.grid(True, alpha=0.3, axis="y")

    ax_ret2.set_title("Final Log Return by Seed", fontsize=11, fontweight="bold")
    ax_ret2.set_xlabel("Seed")
    ax_ret2.set_ylabel("Cumulative Log Return")
    ax_ret2.legend(fontsize=10)
    ax_ret2.grid(True, alpha=0.3, axis="y")

    for label in ["DQN", "IQN"]:
        runs = [r for r in all_results if r["label"] == label]
        caps = [r["final_capital"] for r in runs if r["final_capital"]]
        rets = [r["final_return"]  for r in runs if r["final_return"]]
        if caps and rets:
            print(f"{label}: mean_capital=${np.mean(caps):.1f}  mean_log_ret={np.mean(rets):.4f}")

    path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log    = setup_logging(args.out_dir)

    # epsilon_decay: number of steps for exp decay from 1.0 to ~0.05
    # set so epsilon reaches 0.05 at epsilon_frac * total_steps
    # exp(-T/decay) = 0.05  =>  decay = T / ln(20)  where T = epsilon_frac * total_steps
    epsilon_decay = int(args.epsilon_frac * args.total_steps / np.log(20))
    log.info(f"Device: {device} | epsilon_decay={epsilon_decay} "
             f"(reaches 0.05 at step ~{int(args.epsilon_frac * args.total_steps):,}) | "
             f"args: {vars(args)}")

    all_results = []

    shared = dict(
        state_dim=2, n_actions=2,
        lr=args.lr, gamma=args.gamma,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=epsilon_decay,          # <-- now scales with total_steps
        batch_size=args.batch_size, buffer_size=100_000,
        target_update_freq=500, hidden=args.hidden, device=device,
    )

    # IQN: more quantiles + deeper state embedding to leverage distributional signal
    iqn_kwargs = {
        **shared,
        "n_quantiles":        16,   # was 8 — more training quantile samples
        "n_quantiles_target": 16,   # was 8
        "n_quantiles_policy": 64,   # was 32 — richer action selection
        "state_emb_dim":      256,  # wider state encoder (was 128 default)
    }
    for seed in range(args.n_seeds):
        all_results.append(train_run(DQNAgent, shared, seed, "DQN", args, device, log))
    for seed in range(args.n_seeds):
        all_results.append(train_run(IQNAgent, iqn_kwargs, seed, "IQN", args, device, log))



    plot_top2_progress(all_results, "DQN", args)
    plot_top2_progress(all_results, "IQN", args)
    plot_comparison(all_results, args)
    log.info("All figures saved to " + args.out_dir)


if __name__ == "__main__":
    main()
