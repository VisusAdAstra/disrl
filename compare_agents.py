"""
Main script to compare IQN vs DQN performance on bimodal reward distributions.

This script demonstrates that IQN (Implicit Quantile Networks) can better model
reward distributions compared to standard DQN, especially in environments with
bimodal or multimodal reward structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from iqn_agent import IQN
from dqn_agent import DQN
from bimodal_env import SimpleBimodalEnv, BimodalRewardEnv, MultiModalChainEnv
from utils import (plot_training_curves, compute_statistics, 
                   analyze_risk_sensitivity, plot_learning_speed,
                   visualize_q_distributions, plot_episode_by_episode_comparison)
from device_config import get_device, print_gpu_memory_usage
from optimized_configs import get_configs_for_env
import argparse


def train_and_compare(env_name='SimpleBimodal', num_runs=3, total_steps=30000, verbose=True, device='auto'):
    """
    Train both IQN and DQN agents and compare their performance.
    
    Args:
        env_name: Name of environment to use ('SimpleBimodal', 'BimodalReward', 'MultiModalChain', or any Gym env)
        num_runs: Number of independent runs for each agent
        total_steps: Number of training steps per run
        verbose: Whether to print training progress
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
    
    Returns:
        Dictionary with results for both agents
    """
    
    # Get device
    device_obj = get_device(device)
    device_str = str(device_obj)
    
    # Register custom environments
    if env_name == 'SimpleBimodal':
        gym.register(id='SimpleBimodal-v0', entry_point=SimpleBimodalEnv)
        env_id = 'SimpleBimodal-v0'
    elif env_name == 'BimodalReward':
        gym.register(id='BimodalReward-v0', entry_point=BimodalRewardEnv)
        env_id = 'BimodalReward-v0'
    elif env_name == 'MultiModalChain':
        gym.register(id='MultiModalChain-v0', entry_point=MultiModalChainEnv)
        env_id = 'MultiModalChain-v0'
    else:
        env_id = env_name
    
    print(f"\n{'='*80}")
    print(f"COMPARING IQN vs DQN ON {env_id}")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Environment: {env_id}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Steps per run: {total_steps:,}")
    print(f"\n{'='*80}\n")
    
    # Store results for all runs
    all_results = {
        'IQN': {'all_runs': [], 'best_run': None, 'best_mean': -np.inf},
        'DQN': {'all_runs': [], 'best_run': None, 'best_mean': -np.inf}
    }
    
    # Run experiments
    for run in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1}/{num_runs}")
        print(f"{'='*80}\n")
        
        # Get optimized configs for this environment
        iqn_cfg, dqn_cfg = get_configs_for_env(env_id)
        
        # Update with user parameters (keep target_reward from config)
        iqn_cfg.update({
            'env_name': env_id,
            'total_steps': total_steps,
            'verbose': verbose,
            'device': device_str
        })
        
        dqn_cfg.update({
            'env_name': env_id,
            'total_steps': total_steps,
            'verbose': verbose,
            'device': device_str
        })
        
        # Train IQN
        print(f"\n--- Training IQN (Run {run + 1}) ---")
        iqn_agent = IQN(iqn_cfg)
        iqn_logs = iqn_agent.train()
        all_results['IQN']['all_runs'].append(iqn_logs)
        
        # Print GPU memory if using GPU
        if 'cuda' in device_str:
            print_gpu_memory_usage()
        
        # Check if this is the best IQN run
        mean_reward = np.mean(iqn_logs['episode_rewards'][-100:])
        if mean_reward > all_results['IQN']['best_mean']:
            all_results['IQN']['best_mean'] = mean_reward
            all_results['IQN']['best_run'] = iqn_logs
            all_results['IQN']['best_agent'] = iqn_agent
        
        # Train DQN
        print(f"\n--- Training DQN (Run {run + 1}) ---")
        dqn_agent = DQN(dqn_cfg)
        dqn_logs = dqn_agent.train()
        all_results['DQN']['all_runs'].append(dqn_logs)
        
        # Print GPU memory if using GPU
        if 'cuda' in device_str:
            print_gpu_memory_usage()
        
        # Check if this is the best DQN run
        mean_reward = np.mean(dqn_logs['episode_rewards'][-100:])
        if mean_reward > all_results['DQN']['best_mean']:
            all_results['DQN']['best_mean'] = mean_reward
            all_results['DQN']['best_run'] = dqn_logs
            all_results['DQN']['best_agent'] = dqn_agent
    
    return all_results, env_id


def analyze_results(all_results, env_id):
    """Analyze and visualize results from multiple runs."""
    
    # Create figures directory
    import os
    os.makedirs('./figures', exist_ok=True)
    
    print("\n" + "="*80)
    print("AGGREGATE RESULTS ACROSS ALL RUNS")
    print("="*80)
    
    # Compute aggregate statistics
    for agent_name in ['IQN', 'DQN']:
        all_runs = all_results[agent_name]['all_runs']
        
        # Get final performance across all runs
        final_means = [np.mean(run['episode_rewards'][-100:]) for run in all_runs]
        final_stds = [np.std(run['episode_rewards'][-100:]) for run in all_runs]
        
        print(f"\n{agent_name}:")
        print(f"  Mean final reward: {np.mean(final_means):.2f} ± {np.std(final_means):.2f}")
        print(f"  Best run final reward: {np.max(final_means):.2f}")
        print(f"  Worst run final reward: {np.min(final_means):.2f}")
        print(f"  Mean variance: {np.mean(final_stds):.2f}")
    
    # Compare best runs
    print("\n" + "="*80)
    print("BEST RUN COMPARISON")
    print("="*80)
    
    best_runs = {
        'IQN (best)': all_results['IQN']['best_run'],
        'DQN (best)': all_results['DQN']['best_run']
    }
    
    # Plot best runs
    fig1 = plot_training_curves(best_runs, 
                                title=f"Best Run Comparison - {env_id}",
                                save_path='./figures/best_runs_comparison.png')
    
    # Compute statistics
    compute_statistics(best_runs)
    
    # Risk sensitivity analysis
    analyze_risk_sensitivity(best_runs)
    
    # Episode-by-episode comparison (NEW - detailed learning curves)
    print("\n" + "="*80)
    print("DETAILED LEARNING CURVES")
    print("="*80)
    fig2 = plot_episode_by_episode_comparison(best_runs,
                                              save_path='./figures/episode_comparison.png')
    
    # Learning speed comparison
    fig3 = plot_learning_speed(best_runs, 
                               target_reward=None,
                               save_path='./figures/learning_speed.png')
    
    # Visualize Q-value distributions for IQN
    if 'best_agent' in all_results['IQN']:
        print("\n" + "="*80)
        print("Q-VALUE DISTRIBUTION VISUALIZATION (IQN)")
        print("="*80)
        
        try:
            iqn_agent = all_results['IQN']['best_agent']
            test_env = gym.make(env_id)
            test_state, _ = test_env.reset()
            
            fig4 = visualize_q_distributions(iqn_agent, test_env, test_state, num_samples=1000)
            if fig4 is not None:
                fig4.savefig('./figures/q_distributions.png', dpi=300, bbox_inches='tight')
                print("Q-value distribution plot saved to ./figures/q_distributions.png")
            test_env.close()
        except Exception as e:
            print(f"Could not visualize Q-distributions: {e}")
    
    return fig1, fig2, fig3


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compare IQN vs DQN on bimodal reward distributions')
    parser.add_argument('--env', type=str, default='SimpleBimodal',
                      choices=['SimpleBimodal', 'BimodalReward', 'MultiModalChain', 'CartPole-v1'],
                      help='Environment to use')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of independent runs')
    parser.add_argument('--steps', type=int, default=30000,
                      help='Training steps per run')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Print training progress')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cuda', 'mps', 'cpu'],
                      help='Device to use (auto=detect GPU automatically)')
    
    args = parser.parse_args()
    
    # Run comparison
    all_results, env_id = train_and_compare(
        env_name=args.env,
        num_runs=args.runs,
        total_steps=args.steps,
        verbose=args.verbose,
        device=args.device
    )
    
    # Analyze results
    analyze_results(all_results, env_id)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nResults saved in ./figures/:")
    print("  - best_runs_comparison.png (4-panel statistical comparison)")
    print("  - episode_comparison.png (detailed learning curves - NEW!)")
    print("  - learning_speed.png (moving average progress)")
    print("  - q_distributions.png (IQN distributional visualization)")
    print("\n" + "="*80)
    
    plt.show()


if __name__ == '__main__':
    main()
