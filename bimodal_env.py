import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BimodalRewardEnv(gym.Env):
    """
    A custom environment with bimodal reward distribution.
    
    The agent navigates a grid and can take risky or safe actions.
    Risky actions lead to bimodal rewards (high or low), while safe actions
    give consistent moderate rewards. This tests if IQN can better model
    the reward distribution compared to DQN.
    """
    
    def __init__(self, grid_size=10, max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # State: [x, y, steps_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([grid_size-1, grid_size-1, max_steps]),
            dtype=np.float32
        )
        
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=risky_move, 5=safe_move
        self.action_space = spaces.Discrete(6)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at bottom-left
        self.pos = np.array([0, 0])
        self.steps = 0
        # Goal at top-right
        self.goal = np.array([self.grid_size-1, self.grid_size-1])
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([self.pos[0], self.pos[1], self.max_steps - self.steps], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # Regular movement actions (0-3)
        if action < 4:
            moves = [np.array([0, 1]), np.array([1, 0]), 
                    np.array([0, -1]), np.array([-1, 0])]
            new_pos = self.pos + moves[action]
            # Keep within bounds
            new_pos = np.clip(new_pos, 0, self.grid_size-1)
            self.pos = new_pos
            reward = -0.01  # Small step penalty
            
        # Risky move (action 4) - Bimodal reward distribution
        elif action == 4:
            # 50% chance of high reward, 50% chance of penalty
            if np.random.random() < 0.5:
                reward = 10.0  # High reward mode
            else:
                reward = -5.0  # Penalty mode
                
        # Safe move (action 5) - Consistent moderate reward
        elif action == 5:
            reward = 2.0  # Consistent moderate reward
        
        # Check if reached goal
        if np.array_equal(self.pos, self.goal):
            reward += 50.0
            terminated = True
            
        # Check if max steps exceeded
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.pos[1], self.pos[0]] = 1
        grid[self.goal[1], self.goal[0]] = 2
        return grid


class SimpleBimodalEnv(gym.Env):
    """
    A simpler bimodal environment - single state with two actions.
    Action 0: Deterministic reward of 1.0
    Action 1: Bimodal reward - 80% chance of 5.0, 20% chance of -10.0
    
    The expected value of action 1 is: 0.8*5 + 0.2*(-10) = 4 - 2 = 2.0
    So action 1 is better on average, but riskier.
    
    IQN should learn the full distribution and choose action 1,
    while DQN might be more conservative.
    """
    
    def __init__(self, max_steps=200):
        super().__init__()
        self.max_steps = max_steps
        
        # Simple state space (just a counter)
        self.observation_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        # Two actions: safe vs risky
        self.action_space = spaces.Discrete(2)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Normalized step counter
        return np.array([self.steps / self.max_steps], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        
        if action == 0:
            # Safe action: deterministic reward
            reward = 1.0
        else:
            # Risky action: bimodal distribution
            # 80% chance of +5, 20% chance of -10
            # Expected value: 0.8*5 + 0.2*(-10) = 2.0 (better than safe!)
            if np.random.random() < 0.8:
                reward = 5.0
            else:
                reward = -10.0
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}


class MultiModalChainEnv(gym.Env):
    """
    Chain environment with multiple reward modalities.
    Agent must navigate a chain and collect rewards that follow
    different distributions at different states.
    """
    
    def __init__(self, chain_length=10, max_steps=50):
        super().__init__()
        self.chain_length = chain_length
        self.max_steps = max_steps
        
        # State: [position, steps_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([chain_length-1, max_steps]),
            dtype=np.float32
        )
        
        # Actions: 0=left, 1=right
        self.action_space = spaces.Discrete(2)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([self.position, self.max_steps - self.steps], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        
        # Move left or right
        if action == 0:  # left
            self.position = max(0, self.position - 1)
        else:  # right
            self.position = min(self.chain_length - 1, self.position + 1)
        
        # Reward structure with bimodal distributions at certain positions
        reward = 0.0
        
        if self.position == self.chain_length // 3:
            # First special state: bimodal low-high
            if np.random.random() < 0.5:
                reward = 1.0
            else:
                reward = 5.0
                
        elif self.position == 2 * self.chain_length // 3:
            # Second special state: bimodal negative-positive
            if np.random.random() < 0.6:
                reward = 8.0
            else:
                reward = -3.0
                
        elif self.position == self.chain_length - 1:
            # End state: high reward
            reward = 20.0
        else:
            # Other states: small negative to encourage progress
            reward = -0.1
        
        terminated = self.position == self.chain_length - 1
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
