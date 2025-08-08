import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yaml", 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

        self.replay_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.env_id = hyperparameters['env_id']

    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        rewards_per_episode = []
        epsilon_history = []
        policy_dqn = DQN(env.observation_space.shape[0], num_actions).to_device(device)

        if is_training:
            memory = ReplayMemory(self.replay_size)

            epsilon = self.epsilon_init


        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device, dtype=torch.float)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, device=device, dtype=torch.float)
                reward = torch.tensor(reward, device=device, dtype=torch.float)

            rewards_per_episode.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == "__main__":
    agent = Agent("flappybird")
    agent.run(is_training=True, render=True)