import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch


def smooth_curve(data, window=10):
    """Smooth data using moving average."""
    if len(data) < window:
        return np.array(data)  # Return as numpy array for consistency
    if window < 2:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    if len(smoothed) == 0:  # Safety check
        return np.array(data)
    return smoothed


def plot_training_curves(results_dict, title="Training Comparison", save_path=None):
    """
    Plot training curves for multiple agents.
    
    Args:
        results_dict: Dictionary with agent names as keys and training logs as values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    for idx, (agent_name, logs) in enumerate(results_dict.items()):
        color = colors[idx]
        rewards = logs['episode_rewards']
        
        # Plot 1: Episode rewards
        axes[0, 0].plot(rewards, alpha=0.3, color=color, linewidth=0.5)
        smoothed = smooth_curve(rewards, window=min(20, len(rewards) // 5))
        axes[0, 0].plot(smoothed, label=agent_name, color=color, linewidth=2)
        
        # Plot 2: Cumulative rewards
        cumulative = np.cumsum(rewards)
        axes[0, 1].plot(cumulative, label=agent_name, color=color, linewidth=2)
        
        # Plot 3: Moving average over episodes
        moving_avg = smooth_curve(rewards, window=min(50, len(rewards) // 3))
        axes[1, 0].plot(moving_avg, label=agent_name, color=color, linewidth=2)
        
        # Plot 4: Distribution of final rewards (last 100 episodes)
        final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        axes[1, 1].hist(final_rewards, alpha=0.5, label=agent_name, bins=20, color=color)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards (with smoothing)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].set_title('Cumulative Rewards')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Moving Average Reward')
    axes[1, 0].set_title('Moving Average (50 episodes)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Final Reward Distribution (last 100 episodes)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def compute_statistics(results_dict):
    """
    Compute and display statistics for each agent.
    
    Args:
        results_dict: Dictionary with agent names as keys and training logs as values
    """
    print("\n" + "="*80)
    print("TRAINING STATISTICS")
    print("="*80)
    
    for agent_name, logs in results_dict.items():
        rewards = logs['episode_rewards']
        
        # Get last 100 episodes or all if less than 100
        final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        
        print(f"\n{agent_name}:")
        print(f"  Total Episodes: {logs['episode_count']}")
        print(f"  Training Duration: {logs['duration']:.2f}s")
        print(f"  Episodes per Second: {logs['episode_count'] / logs['duration']:.2f}")
        print(f"\n  Overall Performance:")
        print(f"    Mean Reward: {np.mean(rewards):.2f}")
        print(f"    Std Reward: {np.std(rewards):.2f}")
        print(f"    Min Reward: {np.min(rewards):.2f}")
        print(f"    Max Reward: {np.max(rewards):.2f}")
        print(f"\n  Final Performance (last {len(final_rewards)} episodes):")
        print(f"    Mean Reward: {np.mean(final_rewards):.2f}")
        print(f"    Std Reward: {np.std(final_rewards):.2f}")
        print(f"    Min Reward: {np.min(final_rewards):.2f}")
        print(f"    Max Reward: {np.max(final_rewards):.2f}")
    
    # Statistical comparison
    if len(results_dict) == 2:
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON")
        print("="*80)
        
        agents = list(results_dict.keys())
        rewards1 = results_dict[agents[0]]['episode_rewards'][-100:]
        rewards2 = results_dict[agents[1]]['episode_rewards'][-100:]
        
        # T-test
        if len(rewards1) < 2 or len(rewards2) < 2:
            print(f"\nT-test: Skipped (need at least 2 samples per agent)")
        elif np.std(rewards1) == 0 and np.std(rewards2) == 0:
            print(f"\nT-test: Skipped (both agents have zero variance)")
            if np.mean(rewards1) > np.mean(rewards2):
                print(f"  Result: {agents[0]} has higher mean ({np.mean(rewards1):.2f} vs {np.mean(rewards2):.2f})")
            elif np.mean(rewards2) > np.mean(rewards1):
                print(f"  Result: {agents[1]} has higher mean ({np.mean(rewards2):.2f} vs {np.mean(rewards1):.2f})")
            else:
                print(f"  Result: Both agents have identical performance")
        elif np.std(rewards1) == 0 or np.std(rewards2) == 0:
            print(f"\nT-test: Skipped (one agent has zero variance)")
            print(f"  {agents[0]} mean: {np.mean(rewards1):.2f}, {agents[1]} mean: {np.mean(rewards2):.2f}")
        else:
            t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
            print(f"\nT-test (last 100 episodes):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                better_agent = agents[0] if np.mean(rewards1) > np.mean(rewards2) else agents[1]
                print(f"  Result: {better_agent} is significantly better (p < 0.05)")
            else:
                print(f"  Result: No significant difference (p >= 0.05)")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
        print(f"\nMann-Whitney U test (last 100 episodes):")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {u_p_value:.4f}")


def visualize_q_distributions(agent, env, state, num_samples=1000):
    """
    Visualize the learned Q-value distributions for IQN vs point estimates for DQN.
    Only works with IQN agents.
    
    Args:
        agent: IQN agent
        env: Environment
        state: State to visualize
        num_samples: Number of quantile samples to use
    """
    if not hasattr(agent, 'online_network') or not hasattr(agent.online_network, 'generate_taus'):
        print("Agent doesn't support quantile distribution visualization (not IQN)")
        return
    
    state_tensor = torch.tensor(state, device=agent.config['device']).unsqueeze(0)
    
    # Sample many quantiles
    taus = torch.rand((1, num_samples)).to(agent.config['device'])
    
    with torch.no_grad():
        q_values = agent.online_network(state_tensor, taus).squeeze(0).cpu().numpy()
    
    # Plot distribution for each action
    num_actions = env.action_space.n
    fig, axes = plt.subplots(1, num_actions, figsize=(5*num_actions, 4))
    
    if num_actions == 1:
        axes = [axes]
    
    for action in range(num_actions):
        action_q_values = q_values[action, :]
        
        axes[action].hist(action_q_values, bins=50, alpha=0.7, edgecolor='black')
        axes[action].axvline(np.mean(action_q_values), color='red', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(action_q_values):.2f}')
        axes[action].axvline(np.median(action_q_values), color='green', 
                           linestyle='--', linewidth=2, label=f'Median: {np.median(action_q_values):.2f}')
        axes[action].set_xlabel('Q-value')
        axes[action].set_ylabel('Frequency')
        axes[action].set_title(f'Action {action}')
        axes[action].legend()
        axes[action].grid(True, alpha=0.3)
    
    plt.suptitle('Q-value Distributions (IQN)', fontsize=14)
    plt.tight_layout()
    
    return fig


def analyze_risk_sensitivity(results_dict):
    """
    Analyze risk sensitivity by looking at variance and worst-case performance.
    
    Args:
        results_dict: Dictionary with agent names as keys and training logs as values
    """
    print("\n" + "="*80)
    print("RISK SENSITIVITY ANALYSIS")
    print("="*80)
    
    for agent_name, logs in results_dict.items():
        rewards = logs['episode_rewards']
        final_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        
        # Compute various risk metrics
        mean_reward = np.mean(final_rewards)
        std_reward = np.std(final_rewards)
        cv = std_reward / mean_reward if mean_reward != 0 else float('inf')  # Coefficient of variation
        
        # Value at Risk (VaR) - 5th percentile
        var_5 = np.percentile(final_rewards, 5)
        
        # Conditional Value at Risk (CVaR) - mean of worst 5%
        worst_5_percent = np.sort(final_rewards)[:max(1, len(final_rewards) // 20)]
        cvar_5 = np.mean(worst_5_percent)
        
        # Sharpe-like ratio (assuming risk-free rate = 0)
        sharpe = mean_reward / std_reward if std_reward > 0 else 0
        
        print(f"\n{agent_name}:")
        print(f"  Coefficient of Variation: {cv:.4f}")
        print(f"  Value at Risk (5%): {var_5:.2f}")
        print(f"  Conditional VaR (5%): {cvar_5:.2f}")
        print(f"  Sharpe-like Ratio: {sharpe:.4f}")
        print(f"  Worst Episode Reward: {np.min(final_rewards):.2f}")
        print(f"  Best Episode Reward: {np.max(final_rewards):.2f}")


def plot_learning_speed(results_dict, target_reward=None, save_path=None):
    """
    Plot learning speed comparison - how fast agents reach certain performance levels.
    
    Args:
        results_dict: Dictionary with agent names as keys and training logs as values
        target_reward: Target reward threshold to analyze
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    has_data = False
    for idx, (agent_name, logs) in enumerate(results_dict.items()):
        color = colors[idx]
        rewards = logs['episode_rewards']
        
        if len(rewards) < 2:
            print(f"Warning: {agent_name} has only {len(rewards)} episode(s), skipping plot")
            continue
            
        has_data = True
        
        # Compute moving average
        window = min(20, max(2, len(rewards) // 10))
        if len(rewards) >= window:
            moving_avg = smooth_curve(rewards, window=window)
            episodes = np.arange(window//2, window//2 + len(moving_avg))
            ax.plot(episodes, moving_avg, label=agent_name, color=color, linewidth=2)
            
            # Mark when agent reaches target
            if target_reward is not None:
                reached_indices = np.where(moving_avg >= target_reward)[0]
                if len(reached_indices) > 0:
                    first_reach = reached_indices[0]
                    ax.axvline(episodes[first_reach], color=color, linestyle='--', alpha=0.5)
                    ax.text(episodes[first_reach], target_reward, 
                          f' {agent_name}\n Episode {episodes[first_reach]}',
                          rotation=90, verticalalignment='bottom')
    
    if not has_data:
        ax.text(0.5, 0.5, 'Insufficient data for learning curve\n(need multiple episodes)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        if target_reward is not None:
            ax.axhline(target_reward, color='black', linestyle=':', linewidth=1, label=f'Target ({target_reward})')
        ax.legend()
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Moving Average Reward (window={window if has_data else "N/A"})')
    ax.set_title('Learning Speed Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning speed plot saved to {save_path}")
    
    return fig


def plot_episode_by_episode_comparison(results_dict, save_path=None):
    """
    Plot episode-by-episode reward comparison between agents.
    Shows raw rewards and smoothed curves.
    
    Args:
        results_dict: Dictionary with agent names as keys and training logs as values
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    has_data = False
    max_episodes = 0
    
    for idx, (agent_name, logs) in enumerate(results_dict.items()):
        color = colors[idx]
        rewards = logs['episode_rewards']
        
        if len(rewards) < 2:
            print(f"Warning: {agent_name} has only {len(rewards)} episode(s)")
            continue
        
        has_data = True
        episodes = np.arange(len(rewards))
        max_episodes = max(max_episodes, len(rewards))
        
        # Plot 1: Raw rewards
        ax1.plot(episodes, rewards, alpha=0.4, color=color, linewidth=0.8)
        window = min(20, max(2, len(rewards) // 10))
        smoothed = smooth_curve(rewards, window=window)
        if len(smoothed) > 0:
            smooth_episodes = np.arange(window//2, window//2 + len(smoothed))
            ax1.plot(smooth_episodes, smoothed, label=f'{agent_name} (smoothed)', 
                    color=color, linewidth=2.5)
        
        # Plot 2: Cumulative reward
        cumulative = np.cumsum(rewards)
        ax2.plot(episodes, cumulative, label=agent_name, color=color, linewidth=2.5)
    
    if not has_data:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'Insufficient data\n(need multiple episodes)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    else:
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward per Episode')
        ax1.set_title('Episode Rewards (Raw + Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Total Accumulated Reward Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode-by-episode comparison saved to {save_path}")
    
    return fig
