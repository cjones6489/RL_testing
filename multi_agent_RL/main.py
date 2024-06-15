import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorldEnv
from model import DQN
from replay_buffer import ReplayBuffer
from trainer import MultiAgentTrainer

def main():
    env = GridWorldEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent1_dqn = DQN(input_dim, output_dim)
    agent2_dqn = DQN(input_dim, output_dim)
    replay_buffer = ReplayBuffer(capacity=10000)

    trainer = MultiAgentTrainer(env, agent1_dqn, agent2_dqn, replay_buffer)
    rewards = trainer.train(num_episodes=50)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes")
    plt.show()

    trainer.evaluate(num_episodes=10)

if __name__ == "__main__":
    main()
