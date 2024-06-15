import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10, nonstationary=False):
        self.k = k
        self.q_true = np.zeros(k)  # True action values
        self.q_est = np.zeros(k)  # Estimated action values
        self.action_count = np.zeros(k)  # Number of times each action is taken
        self.nonstationary = nonstationary

    def get_reward(self, action):
        reward = np.random.randn() + self.q_true[action]
        return reward

    def update_q_true(self):
        if self.nonstationary:
            self.q_true += np.random.randn(self.k) * 0.01  # Random walk for q*(a)

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_est)

def run_experiment(bandit, steps, epsilon, alpha=None):
    rewards = np.zeros(steps)
    for step in range(steps):
        action = bandit.select_action(epsilon)
        reward = bandit.get_reward(action)
        rewards[step] = reward

        # Update q_true to simulate nonstationary problem
        bandit.update_q_true()

        # Update estimated action values
        if alpha is not None:
            bandit.q_est[action] += alpha * (reward - bandit.q_est[action])
        else:
            bandit.action_count[action] += 1
            bandit.q_est[action] += (reward - bandit.q_est[action]) / bandit.action_count[action]

    return rewards

def main():
    steps = 10000
    epsilon = 0.1
    alpha = 0.1

    # Stationary environment
    bandit_stationary_avg = Bandit(nonstationary=False)
    rewards_stationary_avg = run_experiment(bandit_stationary_avg, steps, epsilon)
    
    bandit_stationary_const = Bandit(nonstationary=False)
    rewards_stationary_const = run_experiment(bandit_stationary_const, steps, epsilon, alpha)
    
    # Nonstationary environment
    bandit_nonstationary_avg = Bandit(nonstationary=True)
    rewards_nonstationary_avg = run_experiment(bandit_nonstationary_avg, steps, epsilon)
    
    bandit_nonstationary_const = Bandit(nonstationary=True)
    rewards_nonstationary_const = run_experiment(bandit_nonstationary_const, steps, epsilon, alpha)

    # Apply dark theme
    plt.style.use('dark_background')

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.plot(np.cumsum(rewards_stationary_avg) / (np.arange(steps) + 1), label='Sample Average', color='#1f77b4')  # blue
    ax1.plot(np.cumsum(rewards_stationary_const) / (np.arange(steps) + 1), label='Constant Step-Size', color='#ff7f0e')  # orange
    ax1.set_xlabel('Steps', color='white')
    ax1.set_ylabel('Average Reward', color='white')
    ax1.set_title('Stationary Environment', color='white')
    ax1.legend()

    ax2.plot(np.cumsum(rewards_nonstationary_avg) / (np.arange(steps) + 1), label='Sample Average', color='#1f77b4')  # blue
    ax2.plot(np.cumsum(rewards_nonstationary_const) / (np.arange(steps) + 1), label='Constant Step-Size', color='#ff7f0e')  # orange
    ax2.set_xlabel('Steps', color='white')
    ax2.set_ylabel('Average Reward', color='white')
    ax2.set_title('Nonstationary Environment', color='white')
    ax2.legend()

    # Set the tick colors
    ax1.tick_params(colors='white')
    ax2.tick_params(colors='white')

    # Set the spines color
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('white')

    plt.show()

if __name__ == '__main__':
    main()
