import argparse

import flappy_bird_gymnasium
import gymnasium
import torch
from matplotlib import pyplot as plt
from torch import nn
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import os
from datetime import datetime
import np

DATE_FORMAT = '%m-%d %H:%M:%S'

RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yaml", 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

        # Set values from hyperparameters.yaml
        self.replay_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.env_id = hyperparameters['env_id']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.stop_on_reward = hyperparameters['stop_on_reward']

        # Loss Calculation
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Logging
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        """
        Runs the agent and saves the results to disk.

        Args:
            is_training:
            render:

        Returns:

        """
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = []

        # Networks
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_counter = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_reward = -99999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, device=device, dtype=torch.float)
                reward = torch.tensor(reward, device=device, dtype=torch.float)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_counter += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} -> {best_reward:.2f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_updated_time > datetime.timedelta(second=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_updated_time = current_time

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

            if step_counter > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_counter = 0


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Optimze policy network.

        Args:
            mini_batch:
            policy_dqn:
            target_dqn:
        """
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        """
        Plot mean rewards and epislon decay.

        Args:
            rewards_per_episode:
            epsilon_history:

        Returns:

        """
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode), 2)
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plot.plot(mean_rewards)

        plt.subplot(122)
        plt.xlabel('Time Steps')
        ply.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(w_space=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument('hyperparameters', help==)
    parser.add_argument('--train', help='Training Mode', action='store_true')
    args = parser.parse_args()

    dql= Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=True)
    else:
        dql.run(is_training=False, render=True)