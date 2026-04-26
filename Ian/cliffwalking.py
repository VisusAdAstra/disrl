import random
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

ENV_NAME = "CliffWalking-v1"
NUM_EPISODES = 300 #increase if needed
NUM_SEEDS = 3

GAMMA = 0.99
ALPHA = 0.1

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# Progress printing
VERBOSE = True
PRINT_EVERY_EPISODES = 10
PRINT_STEPS = False

# Distributional support (this use categorical support not quantile)
N_ATOMS = 101
V_MIN = -120.0
V_MAX = 0.0
Z_ATOMS = np.linspace(V_MIN, V_MAX, N_ATOMS)
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)

#utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def moving_average(x, window=30):
    x = np.asarray(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))

def expected_q_under_epsilon_greedy(Q, state, epsilon):
    n_actions = Q.shape[1]
    greedy = np.argmax(Q[state])

    probs = np.ones(n_actions) * epsilon / n_actions
    probs[greedy] += 1.0 - epsilon

    return np.dot(probs, Q[state])

def count_cliff_fall(reward):
    return int(reward <= -100)

def log_episode(method, seed, ep, total_reward, total_falls, epsilon, start_time):
    if VERBOSE and (ep % PRINT_EVERY_EPISODES == 0 or ep == NUM_EPISODES - 1):
        elapsed = time.time() - start_time
        print(
            f"[{method:26s}] "
            f"Seed={seed:2d} | "
            f"Ep={ep + 1:4d}/{NUM_EPISODES} | "
            f"Return={total_reward:7.1f} | "
            f"Falls={total_falls:2d} | "
            f"Eps={epsilon:.3f} | "
            f"Time={elapsed:6.1f}s"
        )

def log_step(method, seed, ep, step, state, action, reward, next_state):
    if PRINT_STEPS:
        print(
            f"[{method}] "
            f"Seed={seed} | Ep={ep + 1} | Step={step} | "
            f"s={state}, a={action}, r={reward}, s'={next_state}"
        )





# SARSA
def train_sarsa(seed):
    set_seed(seed)
    env = gym.make(ENV_NAME)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    returns = []
    falls = []
    epsilon = EPS_START

    start_time = time.time()

    for ep in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed + ep)
        action = epsilon_greedy(Q, state, epsilon)

        done = False
        total_reward = 0
        total_falls = 0
        step = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = epsilon_greedy(Q, next_state, epsilon)

            target = reward
            if not done:
                target += GAMMA * Q[next_state, next_action]

            Q[state, action] += ALPHA * (target - Q[state, action])

            log_step("SARSA", seed, ep, step, state, action, reward, next_state)

            state = next_state
            action = next_action

            total_reward += reward
            total_falls += count_cliff_fall(reward)
            step += 1

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        returns.append(total_reward)
        falls.append(total_falls)

        log_episode("SARSA", seed, ep, total_reward, total_falls, epsilon, start_time)

    env.close()
    return returns, falls


# Q-Learning
def train_q_learning(seed):
    set_seed(seed)
    env = gym.make(ENV_NAME)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    returns = []
    falls = []
    epsilon = EPS_START

    start_time = time.time()

    for ep in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed + ep)

        done = False
        total_reward = 0
        total_falls = 0
        step = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward
            if not done:
                target += GAMMA * np.max(Q[next_state])

            Q[state, action] += ALPHA * (target - Q[state, action])

            log_step("Q-learning", seed, ep, step, state, action, reward, next_state)

            state = next_state

            total_reward += reward
            total_falls += count_cliff_fall(reward)
            step += 1

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        returns.append(total_reward)
        falls.append(total_falls)

        log_episode("Q-learning", seed, ep, total_reward, total_falls, epsilon, start_time)

    env.close()
    return returns, falls



# Expected SARSA
def train_expected_sarsa(seed):
    set_seed(seed)
    env = gym.make(ENV_NAME)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    returns = []
    falls = []
    epsilon = EPS_START

    start_time = time.time()

    for ep in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed + ep)

        done = False
        total_reward = 0
        total_falls = 0
        step = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward
            if not done:
                target += GAMMA * expected_q_under_epsilon_greedy(Q, next_state, epsilon)

            Q[state, action] += ALPHA * (target - Q[state, action])

            log_step("Expected SARSA", seed, ep, step, state, action, reward, next_state)

            state = next_state

            total_reward += reward
            total_falls += count_cliff_fall(reward)
            step += 1

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        returns.append(total_reward)
        falls.append(total_falls)

        log_episode("Expected SARSA", seed, ep, total_reward, total_falls, epsilon, start_time)

    env.close()
    return returns, falls


# Distributional Q-Learning
def project_distribution(reward, done, next_dist):
    projected = np.zeros(N_ATOMS)

    for j in range(N_ATOMS):
        z_j = Z_ATOMS[j]

        if done:
            tz_j = reward
        else:
            tz_j = reward + GAMMA * z_j

        tz_j = np.clip(tz_j, V_MIN, V_MAX)

        b = (tz_j - V_MIN) / DELTA_Z
        l = int(np.floor(b))
        u = int(np.ceil(b))

        if l == u:
            projected[l] += next_dist[j]
        else:
            projected[l] += next_dist[j] * (u - b)
            projected[u] += next_dist[j] * (b - l)

    projected_sum = projected.sum()
    if projected_sum > 0:
        projected /= projected_sum
    else:
        projected[:] = 1.0 / N_ATOMS

    return projected


def train_distributional_q(seed):
    set_seed(seed)
    env = gym.make(ENV_NAME)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # ateegorical Approximation
    eta = np.ones((n_states, n_actions, N_ATOMS)) / N_ATOMS

    returns = []
    falls = []
    epsilon = EPS_START

    start_time = time.time()

    for ep in range(NUM_EPISODES):
        state, _ = env.reset(seed=seed + ep)

        done = False
        total_reward = 0
        total_falls = 0
        step = 0

        while not done:
            Q_mean = np.sum(eta[state] * Z_ATOMS, axis=1)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q_mean))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Greedy wrt distributions
            if done:
                next_dist = np.ones(N_ATOMS) / N_ATOMS
            else:
                # Next Q_eta (expected value)
                next_Q_mean = np.sum(eta[next_state] * Z_ATOMS, axis=1)
                # Action greedy with respect to next_Q_eta
                next_action = int(np.argmax(next_Q_mean))
                next_dist = eta[next_state, next_action]

            target_dist = project_distribution(reward, done, next_dist)

            eta[state, action] = (1 - ALPHA) * eta[state, action] + ALPHA * target_dist
            eta[state, action] = np.maximum(eta[state, action], 0.0)
            eta[state, action] /= eta[state, action].sum()

            log_step(
                "Distributional Q-learning",
                seed,
                ep,
                step,
                state,
                action,
                reward,
                next_state,
            )

            state = next_state

            total_reward += reward
            total_falls += count_cliff_fall(reward)
            step += 1

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        returns.append(total_reward)
        falls.append(total_falls)

        log_episode(
            "Distributional Q-learning",
            seed,
            ep,
            total_reward,
            total_falls,
            epsilon,
            start_time,
        )

    env.close()
    return returns, falls


# Experiment
def run_experiment():
    methods = {
        "SARSA": train_sarsa,
        "Expected SARSA": train_expected_sarsa,
        "Q-learning": train_q_learning,
        "Distributional Q-learning": train_distributional_q,
    }

    all_returns = {}
    all_falls = {}

    total_start_time = time.time()

    for name, fn in methods.items():
        print("\n" + "=" * 72)
        print(f"Training method: {name}")
        print("=" * 72)

        seed_returns = []
        seed_falls = []

        for seed in range(NUM_SEEDS):
            print(f"\n--- Starting {name}, seed {seed + 1}/{NUM_SEEDS} ---")
            seed_start = time.time()

            returns, falls = fn(seed)

            seed_returns.append(returns)
            seed_falls.append(falls)

            seed_elapsed = time.time() - seed_start
            print(
                f"--- Finished {name}, seed {seed + 1}/{NUM_SEEDS} "
                f"in {seed_elapsed:.1f}s ---"
            )

        all_returns[name] = np.array(seed_returns)
        all_falls[name] = np.array(seed_falls)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 72)
    print(f"All experiments finished in {total_elapsed:.1f}s")
    print("=" * 72)

    return all_returns, all_falls



# Plots
def plot_metric(all_results, ylabel, title, window=30):
    plt.figure(figsize=(12, 7))

    for name, data in all_results.items():
        mean_curve = data.mean(axis=0)
        std_curve = data.std(axis=0)

        ma_mean = moving_average(mean_curve, window)
        ma_std = moving_average(std_curve, window)

        x = np.arange(len(ma_mean))

        plt.plot(x, ma_mean, label=name)
        plt.fill_between(x, ma_mean - ma_std, ma_mean + ma_std, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel(f"{ylabel}, moving average window = {window}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    all_returns, all_falls = run_experiment()

    plot_metric(
        all_returns,
        ylabel="Return per episode",
        title="CliffWalking-v1: SARSA, Expected SARSA, Q-learning, Distributional Q-learning",
        window=30,
    )

    plot_metric(
        all_falls,
        ylabel="Cliff falls per episode",
        title="CliffWalking-v1: Cliff fall frequency",
        window=30,
    )
