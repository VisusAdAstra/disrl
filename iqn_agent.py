import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import time


### IQN Q-Network ###
class IQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_eval_quantiles=16, cosine_embedding_dim=16):
        super().__init__()
        self.num_actions = action_dim
        self.hidden_dim = hidden_dim
        self.num_eval_quantiles = num_eval_quantiles     
        self.cosine_embedding_dim = cosine_embedding_dim
        
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.tau_embedding_layer = nn.Linear(cosine_embedding_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.advantage_layer = nn.Linear(hidden_dim, self.num_actions)
        self.advantage_layer.weight.data *= 0.01
        
    def forward(self, state, taus=None):
        if taus is None:
            taus = self.generate_taus(batch_size=state.shape[0], uniform=True).to(state.device)
        
        assert state.shape[0] == taus.shape[0], "Use same batch sizes for state and tau tensors."
        batch_size, n_quantiles = taus.shape
        
        # State encoding
        state_enc = F.relu(self.input_layer(state))
        state_enc = F.relu(self.hidden_layer(state_enc))
        
        # Tau encoding
        pi_i = torch.pi * torch.arange(self.cosine_embedding_dim, device=state.device)
        cos_pi_i_tau = torch.cos(taus.unsqueeze(-1) * pi_i)
        tau_enc = F.relu(self.tau_embedding_layer(cos_pi_i_tau.view(batch_size * n_quantiles, self.cosine_embedding_dim)))
        
        # Combine encodings with Hadamard product
        combined_enc = state_enc.unsqueeze(-1) * tau_enc.view(batch_size, self.hidden_dim, n_quantiles)
        
        # Dueling
        value = self.value_layer(combined_enc.view(batch_size * n_quantiles, self.hidden_dim))
        advantages = self.advantage_layer(combined_enc.view(batch_size * n_quantiles, self.hidden_dim))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values.view(batch_size, self.num_actions, n_quantiles)
    
    def generate_taus(self, batch_size=1, uniform=False):
        if uniform:
            return torch.linspace(0, 1, self.num_eval_quantiles + 2)[1:-1].expand((batch_size, self.num_eval_quantiles))
        return torch.rand((batch_size, self.num_eval_quantiles))
    
    
### Experience Replay Buffer ###
class ReplayBuffer:
    def __init__(self, capacity, num_steps=1, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.num_steps = num_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=num_steps)
        
    def add(self, transition):
        """Pushes transition to buffer and handles n-step logic if required."""
        assert len(transition) == 6, "Use new Gym step API: (s, a, r, s', ter, tru)"
        if self.num_steps == 1:
            state, action, reward, next_state, terminated, _ = transition
            self.buffer.append((state, action, reward, next_state, terminated))
        else:
            self.n_step_buffer.append(transition)
            
            # Calculate n-step reward
            _, _, _, final_state, final_termination, final_truncation = transition
            n_step_reward = 0.
            for _, _, reward, _, _, _ in reversed(self.n_step_buffer):
                n_step_reward = n_step_reward * self.gamma + reward
            state, action, _, _, _, _ = self.n_step_buffer[0]

            # If n-step buffer is full, append to main buffer
            if len(self.n_step_buffer) == self.num_steps:
                self.buffer.append((state, action, n_step_reward, final_state, final_termination))
            
            # If done, clear n-step buffer
            if final_termination or final_truncation:
                self.n_step_buffer.clear()
        
    def sample(self, batch_size):
        """Samples a batch of experiences for learner to learn from."""
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        states = torch.tensor(np.stack(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)
    
    
### Linear Scheduler ###
class LinearScheduler:
    """Used to create variables whose values are linearly annealed over time."""
    def __init__(self, start, end, total_duration, fraction=1.):
        self.start = start
        self.end = end
        self.total_duration = total_duration
        self.duration = int(total_duration * fraction)
        self.step = 0
        
    def get(self):
        """Gets current value without incrementing step counter."""
        if self.step < self.duration:
            current_value = self.start + (self.end - self.start) * (self.step / self.duration)
        else:
            current_value = self.end
        return current_value

    def __call__(self):
        """Gets current value and increments step counter."""
        current_value = self.get()
        self.step += 1
        return current_value    

    
### IQN Agent Class ###
class IQN:
    def __init__(self, config):
        self.device = config['device']
        self.env = gym.make(config['env_name'])
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = self.env.action_space.n
        self.online_network = IQNetwork(state_dim, 
                                        action_dim, 
                                        config['hidden_dim'], 
                                        config['num_eval_quantiles'], 
                                        config['cosine_embedding_dim']).to(self.device)
        self.target_network = IQNetwork(state_dim, 
                                        action_dim, 
                                        config['hidden_dim'], 
                                        config['num_eval_quantiles'], 
                                        config['cosine_embedding_dim']).to(self.device)
        self.update_target_network(1.)
        self.optimizer = torch.optim.AdamW(self.online_network.parameters(), lr=config['lr'])
        self.buffer = ReplayBuffer(config['buffer_capacity'], config['num_steps'], config['gamma'])
        self.epsilon = LinearScheduler(config['eps_start'], config['eps_final'], 
                                       config['total_steps'], config['eps_fraction'])
        self.config = config
        
    def update_target_network(self, tau):
        """Updates the parameters of the target network, tau controls how fully the weights are copied."""
        for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1. - tau) * target_param.data)
                
    def select_action(self, state, epsilon):
        """Epsilon greedy action selection."""
        if random.random() < epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, device=self.config['device']).unsqueeze(0)
        with torch.no_grad():
            return self.online_network(state_tensor).mean(-1).argmax().item()
    
    def learn(self):
        # Load batch and create tensors
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config['batch_size'])
        states = states.to(self.device)
        actions = actions.to(self.device).view(-1, 1, 1)
        rewards = rewards.to(self.device).view(-1, 1, 1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).view(-1, 1, 1)
        
        # Get number of quantiles from config
        n_quantiles = self.config['num_eval_quantiles']
        
        # Sample random quantiles for training (key to IQN)
        taus = self.online_network.generate_taus(batch_size=self.config['batch_size'], uniform=False).to(self.device)
        
        # Predicted Q-value quantiles for current state
        current_state_q_values = self.online_network(states, taus)
        
        # Gather Q-value quantiles of actions actually taken
        current_action_q_values = torch.gather(current_state_q_values, dim=1, index=actions.expand(-1, -1, n_quantiles))
        current_action_q_values = current_action_q_values.squeeze(1)  # (batch, N)
        
        # Compute targets
        with torch.no_grad():
            # Get best actions in next state with double DQN (use uniform taus for action selection)
            next_state_best_actions = torch.argmax(self.online_network(next_states).mean(dim=2), dim=1, keepdims=True).unsqueeze(-1)
            
            # Get target Q-values using SAME sampled taus (simplified, less variance)
            next_state_q_values = self.target_network(next_states, taus)
            next_state_max_q_values = torch.gather(next_state_q_values, dim=1, index=next_state_best_actions.expand(-1, -1, n_quantiles))
            next_state_max_q_values = next_state_max_q_values.squeeze(1)  # (batch, N)
            
            # Bellman equation to compute target Q-values for not done states
            target_q_values = rewards.squeeze(1).squeeze(1).unsqueeze(1) + \
                            self.config['gamma'] ** self.config['num_steps'] * \
                            next_state_max_q_values * (1 - dones.squeeze(1).squeeze(1).unsqueeze(1))
        
        # Calculate TD error and Quantile Huber loss (element-wise, not N×N' grid)
        kappa = self.config['kappa']
        td_error = target_q_values - current_action_q_values  # (batch, N)
        
        # Huber loss
        huber_loss = torch.where(td_error.abs() <= kappa, 
                                 0.5 * td_error.pow(2), 
                                 kappa * (td_error.abs() - 0.5 * kappa))
        
        # Quantile regression loss (element-wise)
        # taus shape: (batch, N)
        quantile_weight = torch.abs(taus - (td_error < 0).float())
        quantile_loss = quantile_weight * huber_loss
        
        # Average over quantiles and batch
        loss = quantile_loss.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Trains agent for a given number of steps according to given configuration."""
        print("Training IQN agent\n")
            
        # Logging information
        logs = {'episode_count': 0, 'episodic_reward': 0., 'episode_rewards': [], 'start_time': time.time()}
        
        # Reset episode
        state, _ = self.env.reset()
        
        # Main training loop
        for step in range(1, self.config['total_steps'] + 1):
            # Get action and execute in environment
            action = self.select_action(state, self.epsilon())
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update logs
            logs['episodic_reward'] += reward
            
            # Push experience to buffer
            self.buffer.add((state, action, reward, next_state, terminated, truncated))

            if terminated or truncated:
                state, _ = self.env.reset()
                
                # Update logs
                logs['episode_count'] += 1
                logs['episode_rewards'].append(logs['episodic_reward'])
                logs['episodic_reward'] = 0.
            else:
                state = next_state
            
            # Perform learning step
            if len(self.buffer) >= self.config['batch_size'] and step > self.config['learning_starts']:
                self.learn()
            
            # Update target network
            if step % self.config['target_update'] == 0:
                self.update_target_network(self.config['tau'])
                
            # If mean of last 20 rewards exceed target, end training
            if len(logs['episode_rewards']) > 0 and np.mean(logs['episode_rewards'][-20:]) >= self.config['target_reward']:
                break
            
            # Print training info if verbose
            if self.config['verbose'] and step % 100 == 0 and len(logs['episode_rewards']) > 0:
                print(f"\r--- {100 * step / self.config['total_steps']:.1f}%"
                      f"\t Step: {step:,}"
                      f"\t Mean Reward: {np.mean(logs['episode_rewards'][-20:]):.2f}"
                      f"\t Epsilon: {self.epsilon.get():.2f}"
                      f"\t Episode: {logs['episode_count']:,}"
                      f"\t Duration: {time.time() - logs['start_time']:,.1f}s  ---", end='')
                if step % 10000 == 0:
                    print()
                    
        # Training ended
        print("\n\nTraining done")
        logs['end_time'] = time.time()
        logs['duration'] = logs['end_time'] - logs['start_time']
        return logs
    
        
### IQN Configuration ###
iqn_config = {
    'env_name': 'CartPole-v1',  # Gym environment to use
    'device': 'cpu',  # Device used for learning
    'total_steps': 50000,  # Total training steps
    'hidden_dim': 16,  # Number of neurons in Q-network hidden layer
    'batch_size': 64,  # Number of experience tuples sampled per learning update
    'buffer_capacity': 100000,  # Maximum length of replay buffer
    'target_update': 50,  # How often to perform target network weight synchronizations
    'tau': 0.5,  # When copying online network weights to target network
    'eps_start': 0.8,  # Initial epsilon to use
    'eps_final': 0.1,  # Lowest possible epsilon value
    'eps_fraction': 0.25,  # Fraction of training period for exploration decay
    'learning_starts': 80,  # Step to begin learning at
    'train_frequency': 1,  # Performs a learning update every `train_frequency` steps
    'lr': 1e-3,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'num_steps': 1,  # Multistep reward steps
    'kappa': 1.0,  # Huber loss kappa
    'num_eval_quantiles': 8,  # N in the IQN paper, resolution of the quantile distribution
    'cosine_embedding_dim': 8,  # N' in IQN paper, dimensionality of cosine embeddings
    'target_reward': 195,  # Stop training when mean reward exceeds this
    'verbose': True,  # Prints steps and rewards in output
}
