import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # DQN
        # self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Dueling DQN
        # Value stream
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        # Advantages stream
        self.fc_advantage = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # DQN
        # return self.fc3(x)

        # Dueling DQN
        # Value Calc
        fc_v = F.relu(self.fc_value(x))
        value = self.value(fc_v)

        # Advantage Calc
        fc_a = F.relu(self.fc_advantage(x))
        advantage = self.advantage(fc_a)

        q = value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        return q


if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
