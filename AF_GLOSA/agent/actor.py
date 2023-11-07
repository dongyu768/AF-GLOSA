import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Beta
from torch.distributions import Normal
from agent.orthogonal_initialization import orthogonal_init
from args import parse_args

args = parse_args()

class HybridActor(nn.Module):
    def __init__(self, state_dim, discrete_dim, continuous_dim):
        super(HybridActor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.l1 = nn.Linear(128, 64)
        orthogonal_init(self.l1)
        self.l2 = nn.Linear(64, discrete_dim)
        orthogonal_init(self.l2)
        self.softmax = nn.Softmax(dim=-1)
        self.discrete_net = nn.Sequential(
            self.l1,
            nn.Tanh(),
            self.l2,
            nn.Tanh()
        )
        self.l3 = nn.Linear(128 + 1, 64)
        orthogonal_init(self.l3)
        self.l4 = nn.Linear(64, 64)
        orthogonal_init(self.l4)
        self.l5 = nn.Linear(64, 32)
        orthogonal_init(self.l5)
        self.mean_layer = nn.Linear(32, continuous_dim)
        orthogonal_init(self.mean_layer, 0.01)
        self.std_layer = nn.Linear(32, continuous_dim)
        orthogonal_init(self.std_layer, 0.01)
        # self.log_std = nn.Parameter(torch.zeros(1, continuous_dim))
        self.continuous_net = nn.Sequential(
            self.l3,
            nn.Tanh(),
            self.l4,
            nn.Tanh(),
            self.l5,
            nn.Tanh()
        )

    def forward(self, x):
        state_encoder = self.encoder(x)
        return state_encoder

    def get_discrete_dist(self, s):
        s_encoder = self.forward(s)
        probability_distribution = self.discrete_net(s_encoder)
        probability_distribution = self.softmax(probability_distribution)
        dist = Categorical(probability_distribution)
        return dist
    def eval_discrete_dist(self, s):
        s_encoder = self.forward(s)
        probability_distribution = self.discrete_net(s_encoder)
        return probability_distribution
    def get_continuous_dist(self, s, a_dis):
        s_encoder = self.forward(s)
        s_encoder = torch.concat([s_encoder, a_dis], dim=-1)
        continuous_action = self.continuous_net(s_encoder)
        mean = torch.tanh(self.mean_layer(continuous_action)) * 2.5
        # if mean.item() > 0:
        #     mean = mean * args.amax
        # else:
        #     mean = mean * args.amin
        std = F.softplus(self.std_layer(continuous_action)) + 0.001
        dist = Normal(mean, std)
        return dist

    def eval_continuous_dist(self, s, a_dis):
        s_encoder = self.forward(s)
        s_encoder = torch.concat([s_encoder, a_dis], dim=-1)
        continuous_action = self.continuous_net(s_encoder)
        mean = torch.tanh(self.mean_layer(continuous_action)) * 3
        std = F.softplus(self.std_layer(continuous_action)) + 0.001
        return mean, std


