"""
Quick demo to show IQN vs DQN performance difference.
Runs a short comparison (1 run, 10k steps) for fast results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import gymnasium as gym
from iqn_agent import IQN
from dqn_agent import DQN
from bimodal_env import SimpleBimodalEnv
from utils import plot_training_curves, compute_statistics, analyze_risk_sensitivity


def quick_demo():
    """Run a quick comparison demo."""
    print("\n" + "="*80)
    print("QUICK DEMO: IQN vs DQN on Bimodal Rewards")
    print("="*80)
    print("\nThis demo runs a short comparison to demonstrate the key differences.")
    print("For robust results, run: python compare_agents.py --runs 3\n")
    print("="*80 + "\n")
    
    # Register environment
    gym.register(id='SimpleBimodal-v0', entry_point=SimpleBimodalEnv)
    
    # Configuration for quick demo
    steps = 10000
    
    iqn_cfg = {
        'env_name': 'SimpleBimodal-v0',
        'device': 'cpu',
        'total_steps': steps,
        'hidden_dim': 32,
        'batch_size': 64,
        'buffer_capacity': 50000,
        'target_update': 100,
        'tau': 0.1,
        'eps_start': 1.0,
        'eps_final': 0.05,
        'eps_fraction': 0.5,
        'learning_starts': 200,
        'train_frequency': 1,
        'lr': 1e-3,
        'gamma': 0.99,
        'num_steps': 1,
        'kappa': 1.0,
        'num_eval_quantiles': 16,
        'cosine_embedding_dim': 16,
        'target_reward': 999999,  # Don't stop early
        'verbose': True
    }
    
    dqn_cfg = {
        'env_name': 'SimpleBimodal-v0',
        'device': 'cpu',
        'total_steps': steps,
        'hidden_dim': 32,
        'batch_size': 64,
        'buffer_capacity': 50000,
        'target_update': 100,
        'tau': 0.1,
        'eps_start': 1.0,
        'eps_final': 0.05,
        'eps_fraction': 0.5,
        'learning_starts': 200,
        'train_frequency': 1,
        'lr': 1e-3,
        'gamma': 0.99,
        'num_steps': 1,
        'target_reward': 999999,  # Don't stop early
        'verbose': True
    }
    
    # Train IQN
    print("\n" + "-"*80)
    print("Training IQN...")
    print("-"*80)
    iqn_agent = IQN(iqn_cfg)
    iqn_logs = iqn_agent.train()
    
    # Train DQN
    print("\n" + "-"*80)
    print("Training DQN...")
    print("-"*80)
    dqn_agent = DQN(dqn_cfg)
    dqn_logs = dqn_agent.train()
    
    # Compare results
    results = {
        'IQN': iqn_logs,
        'DQN': dqn_logs
    }
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING RESULTS")
    print("="*80 + "\n")
    
    fig = plot_training_curves(results, 
                               title="Quick Demo: IQN vs DQN on Bimodal Rewards",
                               save_path='./figures/demo_comparison.png')
    
    # Compute statistics
    compute_statistics(results)
    
    # Risk analysis
    analyze_risk_sensitivity(results)
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    iqn_final = np.mean(iqn_logs['episode_rewards'][-50:])
    dqn_final = np.mean(dqn_logs['episode_rewards'][-50:])
    improvement = ((iqn_final - dqn_final) / dqn_final) * 100
    
    print(f"\nFinal Performance (last 50 episodes):")
    print(f"  IQN: {iqn_final:.2f}")
    print(f"  DQN: {dqn_final:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    if improvement > 5:
        print(f"\n✓ IQN shows superior performance!")
        print(f"  IQN better captures the bimodal reward distribution")
        print(f"  and learns the optimal risky strategy.")
    elif improvement < -5:
        print(f"\n⚠ DQN performed better in this run")
        print(f"  Try running multiple times or longer training")
    else:
        print(f"\n~ Similar performance in this short demo")
        print(f"  Run longer: python compare_agents.py --runs 3")
    
    print("\n" + "="*80)
    print("UNDERSTANDING THE RESULTS")
    print("="*80)
    print("""
In the SimpleBimodal environment:
- Action 0 (Safe): Always gives +1.0 reward
- Action 1 (Risky): 80% chance of +5.0, 20% chance of -10.0
  Expected value: 0.8×5 + 0.2×(-10) = 2.0

Optimal Strategy: Choose Action 1 (risky)
  - It has higher expected value (2.0 vs 1.0)
  - Per step: Risky gives 2x the reward of Safe

Key Differences:
- DQN learns only the mean Q-value, may be conservative
- IQN learns the full distribution, can assess true risk/reward
- IQN should converge to the risky action faster

Why IQN is Better Here:
1. Captures bimodal nature of risky rewards
2. Can distinguish between "sometimes bad" and "always mediocre"
3. Makes more informed risk-aware decisions
    """)
    
    print("\nVisualization saved to: demo_comparison.png")
    print("\nFor comprehensive comparison with statistics:")
    print("  python compare_agents.py --env SimpleBimodal --runs 3 --steps 30000")
    print("="*80 + "\n")


if __name__ == '__main__':
    quick_demo()
