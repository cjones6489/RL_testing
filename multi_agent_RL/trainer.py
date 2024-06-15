import torch
import torch.optim as optim
import random

class MultiAgentTrainer:
    def __init__(self, env, agent1_dqn, agent2_dqn, replay_buffer, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.agent1_dqn = agent1_dqn
        self.agent2_dqn = agent2_dqn
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.agent1_optimizer = optim.Adam(self.agent1_dqn.parameters())
        self.agent2_optimizer = optim.Adam(self.agent2_dqn.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent1_dqn.to(self.device)
        self.agent2_dqn.to(self.device)

    def train(self, num_episodes=10):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                self.env.render()

                if random.random() > self.epsilon:
                    action1 = torch.argmax(self.agent1_dqn(torch.FloatTensor(state).to(self.device))).item()
                    action2 = torch.argmax(self.agent2_dqn(torch.FloatTensor(state).to(self.device))).item()
                else:
                    action1 = self.env.action_space.sample()
                    action2 = self.env.action_space.sample()
                
                next_state, reward, done, _ = self.env.step((action1, action2))
                self.replay_buffer.push(state, (action1, action2), reward, next_state, done)
                state = next_state
                episode_reward += reward

                if done:
                    break

                if len(self.replay_buffer) > self.batch_size:
                    self._update_networks()

            rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon}")
            
        self.env.close()
        return rewards
    
    def _update_networks(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values1 = self.agent1_dqn(states)
        q_values2 = self.agent2_dqn(states)
        next_q_values1 = self.agent1_dqn(next_states)
        next_q_values2 = self.agent2_dqn(next_states)

        q_value1 = q_values1.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        q_value2 = q_values2.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)
        next_q_value1 = next_q_values1.max(1)[0]
        next_q_value2 = next_q_values2.max(1)[0]

        expected_q_value1 = rewards + self.gamma * next_q_value1 * (1 - dones)
        expected_q_value2 = rewards + self.gamma * next_q_value2 * (1 - dones)

        loss1 = (q_value1 - expected_q_value1.detach()).pow(2).mean()
        loss2 = (q_value2 - expected_q_value2.detach()).pow(2).mean()

        self.agent1_optimizer.zero_grad()
        self.agent2_optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        self.agent1_optimizer.step()
        self.agent2_optimizer.step()

    def evaluate(self, num_episodes=10):
        self.agent1_dqn.eval()
        self.agent2_dqn.eval()
        total_rewards = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                self.env.ender()

                action1 = torch.argmax(self.agent1_dqn(torch.FloatTensor(state).to(self.device))).itme()
                action2 = torch.argmax(self.agent2_dqn(torch.FloatTensor(state).to(self.device))).item()
            
                state, reward, done, _ = self.env.step((action1, action2))

                episode_reward += reward

                if done:
                    break

            total_rewards += episode_reward
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        avg_reward = total_rewards / num_episodes
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        self.env.close()

