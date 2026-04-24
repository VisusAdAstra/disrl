"""
Patch: add CVaR-RL to train.py

In train.py, make the following changes:

1. Import CVaRAgent at the top:
   from cvar_rl import CVaRAgent

2. Add "CVaR" to PALETTE:
   PALETTE = {"DQN": "#E05C5C", "IQN": "#4A9EE0", "CVaR": "#4CAF50"}

3. In main(), add CVaR kwargs and training runs.
   Replace the existing main() training block with the version below.
"""

# ── Drop-in replacement for the training block in main() ──────────────────────
#
#   shared = dict(...)          # unchanged
#   iqn_kwargs = {...}          # unchanged
#
#   cvar_kwargs = {
#       **shared,
#       "n_quantiles": 32,      # fixed quantile grid (QR-DQN style)
#       "cvar_alpha":  0.25,    # CVaR at 25%: focus on worst-quarter outcomes
#   }
#
#   all_results = []
#   for seed in range(args.n_seeds):
#       all_results.append(train_run(DQNAgent,  shared,       seed, "DQN",  args, device, log))
#   for seed in range(args.n_seeds):
#       all_results.append(train_run(IQNAgent,  iqn_kwargs,   seed, "IQN",  args, device, log))
#   for seed in range(args.n_seeds):
#       all_results.append(train_run(CVaRAgent, cvar_kwargs,  seed, "CVaR", args, device, log))
#
# ── evaluate() patch ───────────────────────────────────────────────────────────
# CVaRAgent has no n_quantiles_policy attribute so the existing evaluate()
# dispatch already handles it correctly via the hasattr check:
#
#   if hasattr(agent, "n_quantiles_policy"):    # IQN branch
#       qv, _ = agent.online(state, agent.n_quantiles_policy)
#       action = qv.mean(1).argmax(1).item()
#   else:                                       # DQN and CVaR-RL branch
#       action = agent.online(state).argmax(1).item()
#
# BUT for CVaR-RL we want CVaR action selection, not argmax of mean.
# Replace the evaluate() dispatch with:
#
#   if hasattr(agent, "n_quantiles_policy"):            # IQN
#       qv, _ = agent.online(state, agent.n_quantiles_policy)
#       action = qv.mean(1).argmax(1).item()
#   elif hasattr(agent, "cvar_alpha"):                  # CVaR-RL
#       qv = agent.online(state)                        # (1, N, A)
#       action = agent._cvar_values(qv).argmax(1).item()
#   else:                                               # DQN
#       action = agent.online(state).argmax(1).item()
#
# ── Suggested cvar_alpha values to run ────────────────────────────────────────
#
#   alpha = 1.00  ->  risk-neutral QR-DQN  (should match DQN behaviour)
#   alpha = 0.25  ->  CVaR-25%             (moderately risk-averse)
#   alpha = 0.10  ->  CVaR-10%             (strongly risk-averse)
#
# Running all three alongside IQN and DQN gives a full risk-sensitivity spectrum.

# ── Quick sanity check (run standalone) ───────────────────────────────────────
if __name__ == "__main__":
    import torch
    import numpy as np
    from cvar_rl import CVaRAgent

    device = "cpu"
    agent  = CVaRAgent(
        state_dim=2, n_actions=2,
        cvar_alpha=0.25, n_quantiles=32,
        device=device, seed=0,
    )

    # Fake a batch to check forward + update works
    states      = torch.randn(8, 2)
    actions_np  = np.zeros(8, dtype=np.int64)
    rewards_np  = np.random.randn(8).astype(np.float32)
    next_states = torch.randn(8, 2)
    dones_np    = np.zeros(8, dtype=np.float32)

    agent.store(states.numpy(), actions_np, rewards_np,
                next_states.numpy(), dones_np)

    # Fill buffer enough to trigger an update
    for _ in range(40):
        agent.store(
            np.random.randn(8, 2).astype(np.float32), actions_np,
            rewards_np, np.random.randn(8, 2).astype(np.float32), dones_np,
        )
        agent.total_steps += 8

    loss = agent.update()
    print(f"loss = {loss:.6f}" if loss is not None else "buffer not full yet")

    # Check action selection
    s = torch.FloatTensor([[0.0, 0.5]])
    a = agent.select_actions(s)
    print(f"selected action = {a[0]}  (CVaR-alpha={agent.cvar_alpha})")

    # Check cvar_alpha=1.0 matches mean (risk-neutral)
    agent2 = CVaRAgent(state_dim=2, n_actions=2,
                       cvar_alpha=1.0, n_quantiles=32,
                       device=device, seed=0)
    qv  = agent2.online(s)            # (1, N, A)
    cvar_val = agent2._cvar_values(qv)  # should equal mean of all quantiles
    mean_val = qv.mean(dim=1)
    assert torch.allclose(cvar_val, mean_val, atol=1e-5), \
        "alpha=1.0 should recover mean"
    print("alpha=1.0 correctly recovers mean Q  ✓")
    print("All checks passed.")
