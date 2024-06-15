import numpy as np
import matplotlib.pyplot as plt
import random

# Define the grid environment
class GridEnvironment:
    def __init__(self, size, start, goal):
        self.size = size
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        x, y = self.agent_position
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.size - 1)
        
        self.agent_position = (x, y)
        reward = 1 if self.agent_position == self.goal else -0.1
        done = self.agent_position == self.goal
        return self.agent_position, reward, done

# Initialize the environment
size = 10
start = (0, 0)
goal = (size - 1, size - 1)
env = GridEnvironment(size, start, goal)

# Set up the Matplotlib figure for real-time plotting
plt.ion()
fig, ax = plt.subplots()
grid = np.zeros((size, size))

def update_plot(agent_position, episode, cumulative_reward):
    ax.clear()
    ax.imshow(grid, cmap='gray', origin='lower', extent=[-0.5, size-0.5, -0.5, size-0.5])
    ax.plot(agent_position[1], agent_position[0], 'ro', markersize=12)  # Plot agent position
    # Plot the goal marker
    ax.plot(goal[1], goal[0], 'gs', markersize=12)  # 'gs' stands for green square
    ax.text(goal[1], goal[0], 'G', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.set_title(f"Episode: {episode}")
    ax.set_xticks(np.arange(-0.5, size, 1))
    ax.set_yticks(np.arange(-0.5, size, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Display the cumulative reward
    ax.text(0.02, 1.02, f"Reward: {cumulative_reward:.2f}", ha='left', va='center', fontsize=12, color='black', transform=ax.transAxes)
    plt.draw()
    plt.pause(0.1)

# Random agent
episodes = 50
for episode in range(episodes):
    state = env.reset()
    done = False
    steps = 0
    cumulative_reward = 0
    
    while not done and steps < 50:
        action = random.choice([0, 1, 2, 3])  # Choose a random action
        next_state, reward, done = env.step(action)
        cumulative_reward += reward
        update_plot(next_state, episode + 1, cumulative_reward)
        steps += 1

print("Simulation finished.")
plt.ioff()
plt.show()
