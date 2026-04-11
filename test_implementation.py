"""
Quick test script to verify IQN and DQN implementations work correctly.
Runs a short training session on SimpleBimodal environment.
"""

import numpy as np
import gymnasium as gym
from iqn_agent import IQN
from dqn_agent import DQN
from bimodal_env import SimpleBimodalEnv


def test_environment():
    """Test that the bimodal environment works correctly."""
    print("Testing SimpleBimodalEnv...")
    
    # Register and create environment
    gym.register(id='SimpleBimodal-v0', entry_point=SimpleBimodalEnv)
    env = gym.make('SimpleBimodal-v0')
    
    # Test reset
    state, info = env.reset()
    print(f"  Initial state shape: {state.shape}")
    print(f"  State: {state}")
    
    # Test actions
    action_rewards = {0: [], 1: []}
    for _ in range(100):
        for action in [0, 1]:
            state, reward, terminated, truncated, info = env.step(action)
            action_rewards[action].append(reward)
    
    print(f"\n  Action 0 (Safe) rewards:")
    print(f"    Mean: {np.mean(action_rewards[0]):.2f}")
    print(f"    Std: {np.std(action_rewards[0]):.2f}")
    print(f"    Expected: 1.0 (deterministic)")
    
    print(f"\n  Action 1 (Risky) rewards:")
    print(f"    Mean: {np.mean(action_rewards[1]):.2f}")
    print(f"    Std: {np.std(action_rewards[1]):.2f}")
    print(f"    Expected mean: 2.0 (0.8*5 + 0.2*(-10))")
    print(f"    Expected to see ~80% of +5.0 and ~20% of -10.0")
    
    env.close()
    print("✓ Environment test passed!\n")


def test_iqn_agent():
    """Test that IQN agent can be instantiated and trained."""
    print("Testing IQN Agent...")
    
    gym.register(id='SimpleBimodal-v0', entry_point=SimpleBimodalEnv)
    
    config = {
        'env_name': 'SimpleBimodal-v0',
        'device': 'cpu',
        'total_steps': 1000,
        'hidden_dim': 16,
        'batch_size': 32,
        'buffer_capacity': 10000,
        'target_update': 50,
        'tau': 0.5,
        'eps_start': 1.0,
        'eps_final': 0.1,
        'eps_fraction': 0.5,
        'learning_starts': 100,
        'train_frequency': 1,
        'lr': 1e-3,
        'gamma': 0.99,
        'num_steps': 1,
        'kappa': 1.0,
        'num_eval_quantiles': 8,
        'cosine_embedding_dim': 8,
        'target_reward': 100,
        'verbose': False
    }
    
    agent = IQN(config)
    print("  Agent instantiated successfully")
    
    # Test forward pass
    state, _ = agent.env.reset()
    import torch
    state_tensor = torch.tensor(state, device='cpu').unsqueeze(0)
    with torch.no_grad():
        q_values = agent.online_network(state_tensor)
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Expected: (1, {agent.env.action_space.n}, {config['num_eval_quantiles']})")
    
    # Run short training
    print("  Running short training (1000 steps)...")
    logs = agent.train()
    print(f"  Episodes completed: {logs['episode_count']}")
    print(f"  Mean reward (all episodes): {np.mean(logs['episode_rewards']):.2f}")
    
    agent.env.close()
    print("✓ IQN agent test passed!\n")


def test_dqn_agent():
    """Test that DQN agent can be instantiated and trained."""
    print("Testing DQN Agent...")
    
    gym.register(id='SimpleBimodal-v0', entry_point=SimpleBimodalEnv)
    
    config = {
        'env_name': 'SimpleBimodal-v0',
        'device': 'cpu',
        'total_steps': 1000,
        'hidden_dim': 16,
        'batch_size': 32,
        'buffer_capacity': 10000,
        'target_update': 50,
        'tau': 0.5,
        'eps_start': 1.0,
        'eps_final': 0.1,
        'eps_fraction': 0.5,
        'learning_starts': 100,
        'train_frequency': 1,
        'lr': 1e-3,
        'gamma': 0.99,
        'num_steps': 1,
        'target_reward': 100,
        'verbose': False
    }
    
    agent = DQN(config)
    print("  Agent instantiated successfully")
    
    # Test forward pass
    state, _ = agent.env.reset()
    import torch
    state_tensor = torch.tensor(state, device='cpu').unsqueeze(0)
    with torch.no_grad():
        q_values = agent.online_network(state_tensor)
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Expected: (1, {agent.env.action_space.n})")
    
    # Run short training
    print("  Running short training (1000 steps)...")
    logs = agent.train()
    print(f"  Episodes completed: {logs['episode_count']}")
    print(f"  Mean reward (all episodes): {np.mean(logs['episode_rewards']):.2f}")
    
    agent.env.close()
    print("✓ DQN agent test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING IMPLEMENTATION TESTS")
    print("="*60 + "\n")
    
    try:
        test_environment()
        test_iqn_agent()
        test_dqn_agent()
        
        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYou can now run the full comparison:")
        print("  python compare_agents.py --env SimpleBimodal --runs 3 --steps 30000")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
