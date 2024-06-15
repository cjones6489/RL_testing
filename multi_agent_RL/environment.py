import numpy as np
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(4,), dtype=np.int32)
    
    def reset(self):
        self.agent1_pos = np.array([0, 0])
        self.agent2_pos = np.array([4, 4])
        self.goal_pos = np.array([2, 2])
        return self._get_obs()
    
    def step(self, action):
        self._move_agent(self.agent1_pos, action[0])
        self._move_agent(self.agent2_pos, action[1])
        reward, done = self._get_reward()
        return self._get_obs(), reward, done, {}
    
    def _move_agent(self, agent_pos, action):
        if action == 0: # Up
            agent_pos[0] = max(agent_pos[0] - 1, 0)
        elif action == 1: # Down
            agent_pos[0] = min(agent_pos[0] + 1, self.grid_size - 1)
        elif action == 2: # Left
            agent_pos[1] = max(agent_pos[1] - 1, 0)
        elif action == 3: # Right
            agent_pos[1] = min(agent_pos[1] + 1, self.grid_size - 1)

    def _get_obs(self):
        return np.concatenate([self.agent1_pos, self.agent2_pos])
    
    def _get_reward(self):
        if np.array_equal(self.agent1_pos, self.goal_pos) and np.array_equal(self.agent2_pos, self.goal_pos):
            return 1, True
        return 0, False
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = "."
        grid[self.agent1_pos[0], self.agent1_pos[1]] = "A1"
        grid[self.agent2_pos[0], self.agent2_pos[1]] = "A2"
        grid[self.goal_pos[0], self.goal_pos[1]] = "G"
        print("\n".join([" ".join(row) for row in grid]))
        print()