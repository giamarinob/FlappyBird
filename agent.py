import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools

device = torch.device("mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"))

class Agent:
    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        rewards_per_episode = []
        policy_dqn = DQN(env.observation_space.shape[0], num_actions).to_device(device)

        if is_training:
            memory = ReplayMemory(10000)

        for epsiode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)
                episode_reward += reward

                state = new_state

            rewards_per_episode.append(episode_reward)
