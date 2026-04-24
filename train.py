"""
Training: DQN vs IQN vs CVaR-RL on Bimodal Investment Environment.

Usage: python train.py [options]
python train.py --n_seeds 1 --cvar_alpha 0.1 --lr 1e-4 --n_envs 16

  --n_seeds          INT    Seeds per algorithm              (default: 2)
  --total_steps      INT    Env steps per run               (default: 300000)
  --n_envs           INT    Parallel environments           (default: 8)
  --episode_length   INT    Steps per episode               (default: 50)
  --eval_interval    INT    Steps between evaluations       (default: 5000)
  --eval_episodes    INT    Greedy eval episodes            (default: 50)
  --warmup_steps     INT    Steps before training starts   (default: 2000)
  --lr               FLOAT  Learning rate                  (default: 3e-4)
  --batch_size       INT    Batch size                     (default: 256)
  --gamma            FLOAT  Discount factor                (default: 0.99)
  --hidden           INT    Hidden layer width             (default: 128)
  --epsilon_frac     FLOAT  Fraction of steps for eps decay (default: 0.35)
  --cvar_alpha       FLOAT  CVaR risk level (0,1]          (default: 0.25)
  --out_dir          STR    Output directory               (default: ./results)

Algorithms compared:
  DQN      — scalar Q, mean-optimal  -> expects risky (EV=0.058 > 0.05)
  IQN      — distributional, mean-greedy -> implicitly safe (gradient dynamics)
  CVaR-RL  — QR-DQN + CVaR in target -> explicitly risk-averse safe policy
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
from cvar_rl import CVaRAgent

# ── Constants ─────────────────────────────────────────────────────────────────

PALETTE = {"DQN": "#E05C5C", "IQN": "#4A9EE0", "CVaR": "#4CAF50"}
MARKERS = {"DQN": "s", "IQN": "o", "CVaR": "^"}

# Env params — must match env.py
_SAFE_RET    = 0.05
_RISKY_WIN   = 0.12
_RISKY_LOSS  = -0.5
_RISKY_P_WIN = 0.9
_EP_LEN      = 50

SAFE_CAPITAL  = 100 * (1 + _SAFE_RET) ** _EP_LEN
_RISKY_GROWTH = ((1 + _RISKY_WIN) ** _RISKY_P_WIN
                 * (1 + _RISKY_LOSS) ** (1 - _RISKY_P_WIN))
RISKY_CAPITAL = 100 * _RISKY_GROWTH ** _EP_LEN


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds",        type=int,   default=2)
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
    p.add_argument("--cvar_alpha",     type=float, default=0.25)
    p.add_argument("--out_dir",        type=str,   default="./results")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────

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
    Greedy evaluation over eval_episodes full episodes.
    Uses each agent's native action-selection:
      IQN     -> mean of quantile samples
      CVaR-RL -> CVaR_alpha action selection
      DQN     -> scalar argmax

    Returns:
      mean_reward    - mean cumulative reward
      median_capital - median final capital (robust to tail skew)
      mean_risky     - mean fraction of risky steps
      all_capitals   - list of per-episode final capitals
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
            state = torch.FloatTensor(
                [[np.log(capital / 100.0), step_frac]]
            ).to(device)

            with torch.no_grad():
                if hasattr(agent, "n_quantiles_policy"):
                    # IQN: mean of quantile samples
                    qv, _ = agent.online(state, agent.n_quantiles_policy)
                    action = qv.mean(1).argmax(1).item()
                elif hasattr(agent, "cvar_alpha"):
                    # CVaR-RL: CVaR action selection
                    qv     = agent.online(state)
                    action = agent._cvar_values(qv).argmax(1).item()
                else:
                    # DQN: scalar argmax
                    action = agent.online(state).argmax(1).item()

            if action == 0:
                ret = _SAFE_RET
            else:
                ret = _RISKY_WIN if np.random.random() < _RISKY_P_WIN \
                      else _RISKY_LOSS
                risky_steps += 1

            ep_reward += ret
            capital   *= (1 + ret)

        ep_reward += np.log(capital / 100.0) / 100
        ep_rewards.append(ep_reward)
        ep_capitals.append(capital)
        ep_risky_frac.append(risky_steps / args.episode_length)

    return (
        float(np.mean(ep_rewards)),
        float(np.median(ep_capitals)),
        float(np.mean(ep_risky_frac)),
        ep_capitals,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_run(agent_class, agent_kwargs, seed, label, args, device, log):
    log.info(f"Starting {label} seed={seed}")
    agent = agent_class(seed=seed, **agent_kwargs)
    env   = BimodalInvestmentEnv(
        n_envs=args.n_envs,
        episode_length=args.episode_length,
        device=device,
    )
    state = env.reset()

    eval_steps         = []
    eval_rewards       = []
    eval_capitals      = []
    eval_risky_frac    = []
    converged_capitals = []
    converge_step      = int(args.total_steps * 0.80)

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
        state  = next_state
        step  += args.n_envs
        agent.total_steps = step

        if step >= args.warmup_steps:
            loss = agent.update()
            if loss is not None:
                recent_losses.append(loss)

        if step % args.eval_interval < args.n_envs:
            avg_rew, med_cap, risky_frac, ep_caps = evaluate(agent, args, device)
            eval_steps.append(step)
            eval_rewards.append(avg_rew)
            eval_capitals.append(med_cap)
            eval_risky_frac.append(risky_frac)

            if step >= converge_step:
                converged_capitals.extend(ep_caps)

            avg_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            log.info(
                f"[{label} s{seed}] step={step:7d} | "
                f"median_cap=${med_cap:8.2f} | risky={risky_frac:.2f} | "
                f"loss={avg_loss:.5f} | eps={agent.epsilon:.3f} | "
                f"elapsed={time.time()-t0:.0f}s"
            )

    n_conv     = max(1, len(eval_risky_frac) // 5)
    conv_risky = float(np.mean(eval_risky_frac[-n_conv:]))
    conv_label = ("safe"  if conv_risky < 0.30 else
                  "risky" if conv_risky > 0.70 else "mixed")

    result = {
        "label":              label,
        "seed":               seed,
        "eval_steps":         eval_steps,
        "eval_rewards":       eval_rewards,
        "eval_capitals":      eval_capitals,
        "eval_risky_frac":    eval_risky_frac,
        "converged_capitals": converged_capitals,
        "final_capital":      eval_capitals[-1]   if eval_capitals   else None,
        "final_reward":       eval_rewards[-1]    if eval_rewards    else None,
        "final_risky":        eval_risky_frac[-1] if eval_risky_frac else None,
        "conv_risky":         conv_risky,
        "conv_label":         conv_label,
    }

    with open(os.path.join(args.out_dir, f"{label}_seed{seed}_log.json"), "w") as f:
        json.dump(result, f)

    return result


# ── Plotting helpers ───────────────────────────────────────────────────────────

def smooth(arr, window=7):
    arr = np.array(arr, dtype=float)
    if len(arr) < window:
        return arr
    pad    = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(arr)]


def plot_top2_progress(results, algo, args):
    """Per-algorithm top-2 seeds progress plot."""
    runs = sorted(
        [r for r in results if r["label"] == algo],
        key=lambda r: r["final_capital"] if r["final_capital"] is not None else -np.inf,
        reverse=True,
    )[:2]
    if not runs:
        return None

    color = PALETTE[algo]
    fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 5), squeeze=False)
    fig.suptitle(f"{algo} — Top {len(runs)} Run(s) by Final Capital",
                 fontsize=14, fontweight="bold")

    for ax, run in zip(axes[0], runs):
        steps = np.array(run["eval_steps"])
        caps  = smooth(run["eval_capitals"])
        risky = smooth(run["eval_risky_frac"])

        ax2 = ax.twinx()
        l2, = ax2.plot(steps, risky * 100, color="gray", alpha=0.5,
                       lw=1.5, linestyle=":", label="% Risky Steps")
        l1, = ax.plot(steps, caps, color=color, lw=2.2, label="Median Capital ($)")
        ax.axhline(SAFE_CAPITAL,  color="orange", lw=1, linestyle="--",
                   alpha=0.6, label=f"Always Safe ${SAFE_CAPITAL:.0f}")
        ax.axhline(RISKY_CAPITAL, color="green",  lw=1, linestyle="--",
                   alpha=0.6, label=f"Always Risky ${RISKY_CAPITAL:.0f}")

        ax2.set_ylim(0, 110)
        ax2.set_ylabel("% Risky Steps")
        valid_caps = [c for c in run["eval_capitals"] if c is not None]
        if valid_caps:
            ax.set_ylim(0, max(max(valid_caps) * 1.15, SAFE_CAPITAL * 1.1))

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Median Capital ($)")
        ax.set_title(
            f"Seed {run['seed']}  |  Capital: ${run['final_capital']:.0f}  "
            f"|  Risky: {run['final_risky']*100:.0f}%  |  Policy: {run['conv_label']}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        lines = [l1, l2] + ax.get_lines()[1:]
        ax.legend(lines, [l.get_label() for l in lines],
                  loc="upper left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(args.out_dir, f"{algo.lower()}_top2_progress.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison(all_results, args):
    """
    5-panel comparison figure across DQN / IQN / CVaR-RL:
      Row 0: Box plot of converged capital distribution per seed
      Row 1: % Risky steps over training (mean +/- std)
      Row 2L: Seed convergence classification
      Row 2R: Pooled capital histogram + statistical tests
      Row 3: Risk profile bar chart (p10 / median / p90)
    """
    agents = [la for la in ["DQN", "IQN", "CVaR"]
              if any(r["label"] == la for r in all_results)]

    fig = plt.figure(figsize=(16, 17))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax_box   = fig.add_subplot(gs[0, :])
    ax_risky = fig.add_subplot(gs[1, :])
    ax_conv  = fig.add_subplot(gs[2, 0])
    ax_hist  = fig.add_subplot(gs[2, 1])
    ax_risk  = fig.add_subplot(gs[3, :])

    # ── Box plot ──────────────────────────────────────────────────────────────
    box_data, box_labels, box_colors = [], [], []
    for label in agents:
        for r in [r for r in all_results if r["label"] == label]:
            caps = r.get("converged_capitals", [])
            if caps:
                box_data.append(caps)
                box_labels.append(f"{label}\ns{r['seed']}\n({r['conv_label']})")
                box_colors.append(PALETTE[label])

    if box_data:
        bp = ax_box.boxplot(
            box_data, patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax_box.set_xticklabels(box_labels, fontsize=8)
        all_caps = [c for d in box_data for c in d]
        ymax = np.percentile(all_caps, 97) * 1.2
        ax_box.set_ylim(0, max(ymax, SAFE_CAPITAL * 1.1))

    ax_box.axhline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--",
                   alpha=0.7, label=f"Always Safe ${SAFE_CAPITAL:.0f}")
    ax_box.axhline(RISKY_CAPITAL, color="green",  lw=1.5, linestyle="--",
                   alpha=0.7, label=f"Always Risky ${RISKY_CAPITAL:.0f}")
    ax_box.set_title(
        "Converged Policy Capital Distribution — last 20% of training (per seed)\n"
        "Training curve tracks median capital; box shows full episode distribution",
        fontsize=12, fontweight="bold",
    )
    ax_box.set_ylabel("Final Capital ($)")
    ax_box.legend(fontsize=9)
    ax_box.grid(True, alpha=0.3, axis="y")

    # ── Risky fraction curves ─────────────────────────────────────────────────
    for label in agents:
        runs = [r for r in all_results if r["label"] == label]
        color = PALETTE[label]
        ref_len = min(len(r["eval_risky_frac"]) for r in runs)
        if ref_len == 0:
            continue
        ref_steps = np.array(runs[0]["eval_steps"][:ref_len])
        mat = np.array([r["eval_risky_frac"][:ref_len] for r in runs])
        mr  = smooth(mat.mean(0) * 100)
        sr  = mat.std(0) * 100
        ax_risky.plot(ref_steps, mr, color=color, lw=2.5, label=label)
        ax_risky.fill_between(ref_steps, mr - sr, mr + sr, color=color, alpha=0.15)

    ax_risky.axhline(100, color="green",  lw=1, linestyle="--",
                     alpha=0.5, label="Always Risky")
    ax_risky.axhline(0,   color="orange", lw=1, linestyle="--",
                     alpha=0.5, label="Always Safe")
    ax_risky.set_title("% Risky Steps — Greedy Policy (mean ± std across seeds)",
                       fontsize=12, fontweight="bold")
    ax_risky.set_xlabel("Environment Steps")
    ax_risky.set_ylabel("% Risky Steps")
    ax_risky.set_ylim(-5, 115)
    ax_risky.legend(fontsize=10)
    ax_risky.grid(True, alpha=0.3)

    # ── Convergence classification ────────────────────────────────────────────
    conv_map   = {"safe": 0, "mixed": 1, "risky": 2}
    conv_color = {"safe": "#4A9EE0", "mixed": "#AAAAAA", "risky": "#E05C5C"}
    n_agents   = len(agents)
    offsets    = {la: (i - n_agents // 2) * 0.2
                  for i, la in enumerate(agents)}
    for label in agents:
        runs  = [r for r in all_results if r["label"] == label]
        x_off = offsets.get(label, 0.0)
        for r in runs:
            cl = r.get("conv_label", "mixed")
            ax_conv.scatter(
                r["seed"] + x_off, conv_map[cl],
                color=PALETTE[label], s=130, zorder=3,
                marker=MARKERS[label],
                edgecolors=conv_color[cl], linewidths=2,
                label=label if r["seed"] == runs[0]["seed"] else "",
            )
            ax_conv.annotate(
                f"{r['conv_risky']*100:.0f}%",
                (r["seed"] + x_off, conv_map[cl]),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=7, color=PALETTE[label],
            )

    ax_conv.set_yticks([0, 1, 2])
    ax_conv.set_yticklabels(["Safe\n(<30% risky)", "Mixed\n(30-70%)", "Risky\n(>70%)"])
    ax_conv.set_xticks(range(max(r["seed"] for r in all_results) + 1))
    ax_conv.set_xlabel("Seed")
    ax_conv.set_title("Converged Strategy per Seed", fontsize=11, fontweight="bold")
    ax_conv.legend(fontsize=9, loc="upper right")
    ax_conv.grid(True, alpha=0.3, axis="x")

    # ── Pooled histogram + stats ──────────────────────────────────────────────
    stats_lines = []
    pooled = {}
    for label in agents:
        runs = [r for r in all_results if r["label"] == label]
        caps = [c for r in runs for c in r.get("converged_capitals", [])]
        if not caps:
            continue
        pooled[label] = caps
        ax_hist.hist(caps, bins=30, color=PALETTE[label], alpha=0.45,
                     label=label, density=True)
        for pct, ls in [(10, ":"), (50, "--"), (90, "-.")]:
            ax_hist.axvline(np.percentile(caps, pct), color=PALETTE[label],
                            lw=1.2, linestyle=ls, alpha=0.8)
        stats_lines.append(
            f"{label}: med=${np.median(caps):.0f} "
            f"p10=${np.percentile(caps,10):.0f} "
            f"p90=${np.percentile(caps,90):.0f}"
        )

    # Pairwise Mann-Whitney U tests
    pw_labels = list(pooled.keys())
    for i in range(len(pw_labels)):
        for j in range(i + 1, len(pw_labels)):
            la, lb = pw_labels[i], pw_labels[j]
            _, p_val = stats.mannwhitneyu(
                pooled[la], pooled[lb], alternative="two-sided"
            )
            stats_lines.append(f"MW-U {la} vs {lb}: p={p_val:.3f}")

    ax_hist.set_xlim(0, 5000)
    ax_hist.axvline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--",
                    alpha=0.7, label=f"Always Safe ${SAFE_CAPITAL:.0f}")
    ax_hist.axvline(RISKY_CAPITAL, color="green",  lw=1.5, linestyle="--",
                    alpha=0.7, label=f"Always Risky ${RISKY_CAPITAL:.0f}")
    ax_hist.set_xlabel("Final Capital ($)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Pooled Converged Capital Distribution\n"
                      "(last 20% of training, all seeds)",
                      fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    fig.text(0.76, 0.30, "\n".join(stats_lines), ha="center", va="top",
             fontsize=7.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))

    # ── Risk profile bar chart ────────────────────────────────────────────────
    metric_labels = ["p10  (downside risk)", "Median", "p90  (upside)"]
    x     = np.arange(len(metric_labels))
    width = 0.22
    bar_offsets = {la: (i - len(agents) // 2) * width
                   for i, la in enumerate(agents)}

    for label in agents:
        caps = pooled.get(label, [])
        if not caps:
            continue
        values = [np.percentile(caps, 10), np.median(caps), np.percentile(caps, 90)]
        bars = ax_risk.bar(
            x + bar_offsets.get(label, 0), values, width * 0.88,
            label=label, color=PALETTE[label], alpha=0.82,
        )
        for bar, val in zip(bars, values):
            ax_risk.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"${val:.0f}", ha="center", va="bottom",
                fontsize=7, color=PALETTE[label], fontweight="bold",
            )

    ax_risk.axhline(SAFE_CAPITAL,  color="orange", lw=1.5, linestyle="--",
                    alpha=0.7, label=f"Always Safe ${SAFE_CAPITAL:.0f}")
    ax_risk.axhline(RISKY_CAPITAL, color="green",  lw=1.5, linestyle="--",
                    alpha=0.7, label=f"Always Risky ${RISKY_CAPITAL:.0f}")
    ax_risk.set_xticks(x)
    ax_risk.set_xticklabels(metric_labels, fontsize=11)
    ax_risk.set_ylabel("Final Capital ($)")
    ax_risk.set_title(
        "Risk Profile — p10 / Median / p90 of Converged Capital\n"
        "Lower p10 = more downside exposure   |   "
        "Higher median = better typical outcome",
        fontsize=12, fontweight="bold",
    )
    ax_risk.legend(fontsize=10)
    ax_risk.grid(True, alpha=0.3, axis="y")

    path = os.path.join(args.out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    for line in stats_lines:
        print(line)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log    = setup_logging(args.out_dir)

    epsilon_decay = int(args.epsilon_frac * args.total_steps / np.log(20))
    log.info(f"Device: {device} | eps_decay={epsilon_decay} | args: {vars(args)}")
    log.info(f"Baselines: safe=${SAFE_CAPITAL:.0f}  risky_geometric=${RISKY_CAPITAL:.0f}")

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
    cvar_kwargs = {
        **shared,
        "n_quantiles": 32,
        "cvar_alpha":  args.cvar_alpha,
    }

    all_results = []
    algos = []

    log.info(f"=== Training CVaR-RL (alpha={args.cvar_alpha}) ===")
    for seed in range(args.n_seeds):
        all_results.append(
            train_run(CVaRAgent, cvar_kwargs, seed, "CVaR", args, device, log)
        )
    algos.append("CVaR")

    log.info("=== Training DQN ===")
    for seed in range(args.n_seeds):
        all_results.append(
            train_run(DQNAgent, shared, seed, "DQN", args, device, log)
        )
    algos.append("DQN")

    log.info("=== Training IQN ===")
    for seed in range(args.n_seeds):
        all_results.append(
            train_run(IQNAgent, iqn_kwargs, seed, "IQN", args, device, log)
        )
    algos.append("IQN")

    log.info("=== Plotting ===")
    for algo in algos:
        plot_top2_progress(all_results, algo, args)
    plot_comparison(all_results, args)

    log.info(f"Done. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
