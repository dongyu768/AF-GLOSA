from torch import nn
from torch.nn import Sequential
from agent.orthogonal_initialization import orthogonal_init

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        orthogonal_init(self.l1)
        self.l2 = nn.Linear(256, 128)
        orthogonal_init(self.l2)
        self.l3 = nn.Linear(128, 64)
        orthogonal_init(self.l3)
        self.l4 = nn.Linear(64, 1)
        orthogonal_init(self.l4)
        self.net = Sequential(
            self.l1,
            nn.Tanh(),
            self.l2,
            nn.Tanh(),
            self.l3,
            nn.Tanh(),
            self.l4
        )

    def forward(self, x):
        return self.net(x)


