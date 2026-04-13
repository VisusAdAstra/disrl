"""
Training: DQN vs IQN on Bimodal Investment Environment.

Usage: python train.py [options]

  --n_seeds          INT    Seeds per algorithm          (default: 4)
  --total_steps      INT    Env steps per run            (default: 200000)
  --n_envs           INT    Parallel environments        (default: 8)
  --episode_length   INT    Steps per episode            (default: 50)
  --eval_interval    INT    Steps between evaluations    (default: 5000)
  --eval_episodes    INT    Greedy eval episodes         (default: 50)
  --warmup_steps     INT    Steps before training starts (default: 2000)
  --lr               FLOAT  Learning rate                (default: 3e-4)
  --batch_size       INT    Batch size                   (default: 256)
  --gamma            FLOAT  Discount factor              (default: 0.99)
  --hidden           INT    Hidden layer width           (default: 128)
  --epsilon_frac     FLOAT  Fraction of total_steps for eps decay (default: 0.3)
  --out_dir          STR    Output directory             (default: ./results)

Reward design:
  Flat per-step reward = raw return rate (0.05 safe, +0.12/-0.50 risky)
  + terminal bonus = log(final_capital / 100) / 100
  Per-step EV is identical → DQN indifferent.
  Terminal bonus favors safe compounding → IQN can learn this.
"""

import argparse, os, json, time, logging
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import deque

from env import BimodalInvestmentEnv
from dqn import DQNAgent
from iqn import IQNAgent

PALETTE = {"DQN": "#E05C5C", "IQN": "#4A9EE0"}

# Env params (must match env.py)
_SAFE_RET      = 0.05
_RISKY_WIN     = 0.12
_RISKY_LOSS    = -0.5
_RISKY_P_WIN   = 0.9
_EP_LEN        = 50

# Always-safe: deterministic compounding
SAFE_CAPITAL  = 100 * (1 + _SAFE_RET) ** _EP_LEN          # ~$1,147

# Always-risky: geometric mean growth per step = (1+win)^p * (1+loss)^(1-p)
_RISKY_GROWTH = (1 + _RISKY_WIN) ** _RISKY_P_WIN * (1 + _RISKY_LOSS) ** (1 - _RISKY_P_WIN)
RISKY_CAPITAL = 100 * _RISKY_GROWTH ** _EP_LEN             # ~$239 (geometric mean)


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
    Greedy evaluation over eval_episodes full episodes.
    Returns:
      mean_reward       - mean cumulative reward across episodes
      mean_capital      - mean final capital
      mean_risky_frac   - mean fraction of risky steps
      all_capitals      - list of per-episode final capitals (for distribution)
    """
    ep_rewards    = []
    ep_capitals   = []
    ep_risky_frac = []

    for _ in range(args.eval_episodes):
        capital     = 100.0
        ep_reward   = 0.0
        risky_steps = 0

        for t in range(args.episode_length):
            step_frac = t / args.episode_length
            state = torch.FloatTensor([[np.log(capital / 100.0), step_frac]]).to(device)

            with torch.no_grad():
                if hasattr(agent, "n_quantiles_policy"):
                    qv, _ = agent.online(state, agent.n_quantiles_policy)
                    action = qv.mean(1).argmax(1).item()
                else:
                    action = agent.online(state).argmax(1).item()

            if action == 0:
                ret = 0.05
            else:
                ret = 0.12 if np.random.random() < 0.9 else -0.5
                risky_steps += 1

            ep_reward += ret
            capital   *= (1 + ret)

        ep_reward += np.log(capital / 100.0) / 100
        ep_rewards.append(ep_reward)
        ep_capitals.append(capital)
        ep_risky_frac.append(risky_steps / args.episode_length)

    return (
        float(np.mean(ep_rewards)),
        float(np.median(ep_capitals)),   # median: not skewed by lucky tail runs
        float(np.mean(ep_risky_frac)),
        ep_capitals,   # full distribution for box plots / histograms
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_run(agent_class, agent_kwargs, seed, label, args, device, log):
    log.info(f"Starting {label} seed={seed}")
    agent = agent_class(seed=seed, **agent_kwargs)
    env   = BimodalInvestmentEnv(
        n_envs=args.n_envs, episode_length=args.episode_length, device=device
    )
    state = env.reset()

    eval_steps      = []
    eval_rewards    = []
    eval_capitals   = []   # mean per checkpoint
    eval_risky_frac = []
    # Store per-episode capitals only for last 20% of training (converged policy)
    converged_capitals = []
    converge_step = int(args.total_steps * 0.80)

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
            avg_rew, avg_cap, risky_frac, ep_caps = evaluate(agent, args, device)
            eval_steps.append(step)
            eval_rewards.append(avg_rew)
            eval_capitals.append(avg_cap)
            eval_risky_frac.append(risky_frac)

            # Collect full capital distribution once policy has converged
            if step >= converge_step:
                converged_capitals.extend(ep_caps)

            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log.info(
                f"[{label} s{seed}] step={step:7d} | "
                f"median_cap=${avg_cap:8.2f} | risky={risky_frac:.2f} | "
                f"loss={avg_loss:.5f} | eps={agent.epsilon:.3f} | "
                f"elapsed={time.time()-t0:.0f}s"
            )

    # Convergence classification: mean risky frac over last 20% of checkpoints
    n_conv = max(1, len(eval_risky_frac) // 5)
    conv_risky = float(np.mean(eval_risky_frac[-n_conv:]))
    if conv_risky < 0.30:
        conv_label = "safe"
    elif conv_risky > 0.70:
        conv_label = "risky"
    else:
        conv_label = "mixed"

    with open(os.path.join(args.out_dir, f"{label}_seed{seed}_log.json"), "w") as f:
        json.dump({
            "eval_steps":          eval_steps,
            "eval_rewards":        eval_rewards,
            "eval_capitals":       eval_capitals,
            "eval_risky_frac":     eval_risky_frac,
            "converged_capitals":  converged_capitals,
            "conv_risky":          conv_risky,
            "conv_label":          conv_label,
        }, f)

    return {
        "label":               label,
        "seed":                seed,
        "eval_steps":          eval_steps,
        "eval_rewards":        eval_rewards,
        "eval_capitals":       eval_capitals,
        "eval_risky_frac":     eval_risky_frac,
        "converged_capitals":  converged_capitals,   # distribution for box/histogram
        "final_capital":       eval_capitals[-1]  if eval_capitals  else None,
        "final_reward":        eval_rewards[-1]   if eval_rewards   else None,
        "final_risky":         eval_risky_frac[-1] if eval_risky_frac else None,
        "conv_risky":          conv_risky,
        "conv_label":          conv_label,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

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
        steps = np.array(run["eval_steps"])
        caps  = smooth(run["eval_capitals"])
        risky = smooth(run["eval_risky_frac"])

        ax2 = ax.twinx()
        l2, = ax2.plot(steps, risky * 100, color="gray", alpha=0.5, lw=1.5,
                       linestyle=":", label="% Risky Steps")
        l1, = ax.plot(steps, caps, color=color, lw=2.2, label="Final Capital ($)")

        ax.axhline(RISKY_CAPITAL, color="green",  lw=1, linestyle="--", alpha=0.5,
                   label=f"Always Risky ${RISKY_CAPITAL:.0f}")
        ax.axhline(SAFE_CAPITAL,  color="orange", lw=1, linestyle="--", alpha=0.5,
                   label=f"Always Safe ${SAFE_CAPITAL:.0f}")

        ax2.set_ylim(0, 110)
        ax2.set_ylabel("% Risky Steps")
        # Auto-scale capital axis to actual data range
        valid_caps = [c for c in run["eval_capitals"] if c is not None]
        if valid_caps:
            cap_max = max(max(valid_caps) * 1.15, SAFE_CAPITAL * 1.1)
            ax.set_ylim(0, cap_max)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Final Capital ($)")
        ax.set_title(
            f"Seed {run['seed']}  |  Final Capital: ${run['final_capital']:.0f}  "
            f"|  Risky: {run['final_risky']*100:.0f}%  |  Policy: {run['conv_label']}",
            fontsize=11
        )
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
    """
    4-panel comparison figure:
      Row 0: Box plot of converged capital distribution (last 20% of training)
      Row 1: % Risky steps over training (mean ± std across seeds)
      Row 2L: Seed-level convergence classification (safe / mixed / risky)
      Row 2R: Pooled capital histogram (last 20%) with percentile markers
    """
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax_box   = fig.add_subplot(gs[0, :])   # box plot — replaces empty capital curve
    ax_risky = fig.add_subplot(gs[1, :])   # risky fraction curves
    ax_conv  = fig.add_subplot(gs[2, 0])   # seed convergence classification
    ax_hist  = fig.add_subplot(gs[2, 1])   # pooled capital histogram

    # ── Row 0: Box plot of converged-policy capital distribution ──────────────
    box_data   = []
    box_labels = []
    box_colors = []
    for label in ["DQN", "IQN"]:
        runs = [r for r in all_results if r["label"] == label]
        for r in runs:
            caps = r.get("converged_capitals", [])
            if caps:
                box_data.append(caps)
                box_labels.append(f"{label}\ns{r['seed']}\n({r['conv_label']})")
                box_colors.append(PALETTE[label])

    if box_data:
        bp = ax_box.boxplot(
            box_data,
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax_box.set_xticklabels(box_labels, fontsize=8)
        ax_box.axhline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--", alpha=0.7,
                       label=f"Always Safe ${SAFE_CAPITAL:.0f}")
        ax_box.axhline(RISKY_CAPITAL, color="green",  lw=1.5, linestyle="--", alpha=0.7,
                       label=f"Always Risky ${RISKY_CAPITAL:.0f}")
        # Auto y-scale: 5th–95th percentile of all data, with headroom
        all_caps = [c for d in box_data for c in d]
        ymax = np.percentile(all_caps, 97) * 1.2
        ax_box.set_ylim(0, max(ymax, SAFE_CAPITAL * 1.1))
        ax_box.legend(fontsize=9)

    ax_box.set_title(
        "Converged Policy Capital Distribution — last 20% of training (per seed)\n"
        "Training curve tracks median capital; box shows full episode distribution",
        fontsize=13, fontweight="bold"
    )
    ax_box.set_ylabel("Final Capital ($) — median tracked in training log")
    ax_box.grid(True, alpha=0.3, axis="y")

    # ── Row 1: Risky fraction over training ───────────────────────────────────
    for label in ["DQN", "IQN"]:
        runs    = [r for r in all_results if r["label"] == label]
        color   = PALETTE[label]
        ref_len = min(len(r["eval_risky_frac"]) for r in runs)
        if ref_len == 0:
            continue
        ref_steps = np.array(runs[0]["eval_steps"][:ref_len])
        mat_r     = np.array([r["eval_risky_frac"][:ref_len] for r in runs])
        mr        = smooth(mat_r.mean(0) * 100)
        sr        = mat_r.std(0) * 100
        ax_risky.plot(ref_steps, mr, color=color, lw=2.5, label=label)
        ax_risky.fill_between(ref_steps, mr - sr, mr + sr, color=color, alpha=0.15)

    ax_risky.axhline(100, color="green",  lw=1, linestyle="--", alpha=0.5, label="Always Risky")
    ax_risky.axhline(0,   color="orange", lw=1, linestyle="--", alpha=0.5, label="Always Safe")
    ax_risky.set_title("% Risky Steps — Greedy Policy (mean ± std across seeds)",
                        fontsize=13, fontweight="bold")
    ax_risky.set_xlabel("Environment Steps")
    ax_risky.set_ylabel("% Risky Steps")
    ax_risky.set_ylim(-5, 115)
    ax_risky.legend(fontsize=10)
    ax_risky.grid(True, alpha=0.3)

    # ── Row 2L: Seed convergence classification ────────────────────────────────
    conv_map  = {"safe": 0, "mixed": 1, "risky": 2}
    conv_color = {"safe": "#4A9EE0", "mixed": "#AAAAAA", "risky": "#E05C5C"}
    for label in ["DQN", "IQN"]:
        runs   = [r for r in all_results if r["label"] == label]
        x_off  = -0.15 if label == "DQN" else 0.15
        for r in runs:
            cl = r.get("conv_label", "mixed")
            ax_conv.scatter(
                r["seed"] + x_off, conv_map[cl],
                color=PALETTE[label], s=120, zorder=3,
                marker="o" if label == "IQN" else "s",
                edgecolors=conv_color[cl], linewidths=2,
                label=f"{label}" if r["seed"] == runs[0]["seed"] else "",
            )
            # Annotate conv_risky %
            ax_conv.annotate(
                f"{r['conv_risky']*100:.0f}%",
                (r["seed"] + x_off, conv_map[cl]),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7, color=PALETTE[label],
            )

    ax_conv.set_yticks([0, 1, 2])
    ax_conv.set_yticklabels(["Safe\n(<30% risky)", "Mixed\n(30–70%)", "Risky\n(>70%)"])
    ax_conv.set_xticks(range(max(r["seed"] for r in all_results) + 1))
    ax_conv.set_xlabel("Seed")
    ax_conv.set_title("Converged Strategy per Seed", fontsize=11, fontweight="bold")
    ax_conv.legend(fontsize=9, loc="upper right")
    ax_conv.grid(True, alpha=0.3, axis="x")

    # ── Row 2R: Pooled converged capital histogram + stats ────────────────────
    stats_lines = []
    for label in ["DQN", "IQN"]:
        runs  = [r for r in all_results if r["label"] == label]
        caps  = [c for r in runs for c in r.get("converged_capitals", [])]
        if not caps:
            continue
        color = PALETTE[label]
        ax_hist.hist(caps, bins=30, color=color, alpha=0.45, label=label, density=True)
        # Percentile markers
        for pct, ls in [(10, ":"), (50, "--"), (90, "-.")]:
            v = np.percentile(caps, pct)
            ax_hist.axvline(v, color=color, lw=1.2, linestyle=ls, alpha=0.8)
        med   = np.median(caps)
        p10   = np.percentile(caps, 10)
        p90   = np.percentile(caps, 90)
        stats_lines.append(f"{label}: median=${med:.0f}  p10=${p10:.0f}  p90=${p90:.0f}")

    # Mann-Whitney U test between pooled distributions
    dqn_caps = [c for r in all_results if r["label"] == "DQN"
                for c in r.get("converged_capitals", [])]
    iqn_caps = [c for r in all_results if r["label"] == "IQN"
                for c in r.get("converged_capitals", [])]
    if dqn_caps and iqn_caps:
        u_stat, p_val = stats.mannwhitneyu(iqn_caps, dqn_caps, alternative="greater")
        stats_lines.append(f"Mann-Whitney U (IQN>DQN): p={p_val:.3f}")

    ax_hist.axvline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--", alpha=0.7,
                    label=f"Always Safe ${SAFE_CAPITAL:.0f}")
    ax_hist.axvline(RISKY_CAPITAL, color="green",  lw=1.5, linestyle="--", alpha=0.7,
                    label=f"Always Risky ${RISKY_CAPITAL:.0f}")
    ax_hist.set_xlabel("Final Capital ($)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Pooled Converged Capital Distribution\n(last 20% of training, all seeds)",
                      fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Print stats summary below histogram
    fig.text(0.76, 0.02, "\n".join(stats_lines), ha="center", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # Also print to console
    for line in stats_lines:
        print(line)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log    = setup_logging(args.out_dir)

    epsilon_decay = int(args.epsilon_frac * args.total_steps / np.log(20))
    log.info(f"Device: {device} | epsilon_decay={epsilon_decay} | args: {vars(args)}")
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
        all_results.append(train_run(IQNAgent, iqn_kwargs,  seed, "IQN", args, device, log))
    for seed in range(args.n_seeds):
        all_results.append(train_run(DQNAgent, shared,      seed, "DQN", args, device, log))


    plot_top2_progress(all_results, "DQN", args)
    plot_top2_progress(all_results, "IQN", args)
    plot_comparison(all_results, args)

    log.info("Done. Results in " + args.out_dir)


if __name__ == "__main__":
    main()
