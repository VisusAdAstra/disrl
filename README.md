## Implementation Challenges and Their Resolution

Three design problems were identified and corrected:

### (i) Itô leakage in the terminal bonus

A terminal bonus of $\log(C_T / C_0)$ implicitly penalises variance via the Itô correction $E[\log(1 + r_r)] \approx E[r_r] - \tfrac{1}{2}\mathrm{Var}[r_r] = 0.040 < 0.05$, making safe *genuinely* optimal in expected log-return even for DQN and   confounding the comparison; the bonus was scaled by $1/100$ to reduce its influence to the order of a single per-step reward.

### (ii) Exploration contamination

With a slow $\varepsilon$-decay, the replay buffer accumulated far more $-0.50$ outcomes than the converged policy would generate, artificially biasing both agents toward safe during early training; the decay constant was tightened so that near-greedy transitions dominate the buffer before the bulk of gradient updates occur.

### (iii) Capital-scaled reward leakage

Rewarding the agent with $r\_t \times C\_t$ encodes the current wealth level into the reward signal: after risky losses $C\_t$ is small and future rewards are depressed regardless of action, so DQN associates the risky action with low value through a spurious wealth-level correlation rather than through return-distribution reasoning. Using the raw ratio $r\_t$ as the per-step reward removes this leakage entirely.
