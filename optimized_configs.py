"""
Optimized hyperparameters for SimpleBimodal environment.
"""

# Optimized IQN configuration for SimpleBimodal
iqn_bimodal_config = {
    'env_name': 'SimpleBimodal-v0',
    'device': 'cpu',
    'total_steps': 50000,
    'hidden_dim': 64,
    'batch_size': 128,
    'buffer_capacity': 100000,
    'target_update': 200,
    'tau': 0.005,
    'eps_start': 1.0,
    'eps_final': 0.01,
    'eps_fraction': 0.6,
    'learning_starts': 500,
    'train_frequency': 1,
    'lr': 3e-4,  # Reduced for more stable IQN learning
    'gamma': 0.99,
    'num_steps': 3,
    'kappa': 1.0,
    'num_eval_quantiles': 16,  # Fewer quantiles = less variance
    'cosine_embedding_dim': 32,  # Smaller embedding = more stable
    'target_reward': float('inf'),  # Train full duration
    'verbose': True
}

# Optimized DQN configuration for SimpleBimodal
dqn_bimodal_config = {
    'env_name': 'SimpleBimodal-v0',
    'device': 'cpu',
    'total_steps': 50000,
    'hidden_dim': 64,  # Match IQN
    'batch_size': 128,  # Match IQN
    'buffer_capacity': 100000,
    'target_update': 200,  # Match IQN
    'tau': 0.005,  # Match IQN (soft updates)
    'eps_start': 1.0,
    'eps_final': 0.01,  # Match IQN
    'eps_fraction': 0.6,  # Match IQN
    'learning_starts': 500,  # Match IQN
    'train_frequency': 1,
    'lr': 5e-4,  # Match IQN
    'gamma': 0.99,
    'num_steps': 3,  # Match IQN
    'use_huber': True,  # Use Huber loss like IQN
    'grad_clip': 10.0,  # Add gradient clipping
    'target_reward': float('inf'),  # Train full duration
    'verbose': True
}

# Configs for other environments
iqn_cartpole_config = {
    'env_name': 'CartPole-v1',
    'device': 'cpu',
    'total_steps': 50000,
    'hidden_dim': 64,
    'batch_size': 64,
    'buffer_capacity': 100000,
    'target_update': 100,
    'tau': 0.01,
    'eps_start': 1.0,
    'eps_final': 0.05,
    'eps_fraction': 0.3,
    'learning_starts': 100,
    'train_frequency': 1,
    'lr': 1e-3,
    'gamma': 0.99,
    'num_steps': 1,
    'kappa': 1.0,
    'num_eval_quantiles': 16,
    'cosine_embedding_dim': 32,
    'target_reward': 195,
    'verbose': True
}

dqn_cartpole_config = {
    'env_name': 'CartPole-v1',
    'device': 'cpu',
    'total_steps': 50000,
    'hidden_dim': 64,
    'batch_size': 64,
    'buffer_capacity': 100000,
    'target_update': 100,
    'tau': 0.01,
    'eps_start': 1.0,
    'eps_final': 0.05,
    'eps_fraction': 0.3,
    'learning_starts': 100,
    'train_frequency': 1,
    'lr': 1e-3,
    'gamma': 0.99,
    'num_steps': 1,
    'use_huber': True,
    'target_reward': 195,
    'verbose': True
}


def get_configs_for_env(env_name):
    """
    Get optimized configs for a specific environment.
    
    Returns:
        tuple: (iqn_config, dqn_config)
    """
    if 'SimpleBimodal' in env_name:
        return iqn_bimodal_config.copy(), dqn_bimodal_config.copy()
    elif 'CartPole' in env_name:
        return iqn_cartpole_config.copy(), dqn_cartpole_config.copy()
    else:
        # Use SimpleBimodal configs as default for other custom envs
        return iqn_bimodal_config.copy(), dqn_bimodal_config.copy()
