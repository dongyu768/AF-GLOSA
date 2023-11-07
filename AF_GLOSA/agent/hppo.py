import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from agent.critic import Critic
from agent.actor import HybridActor
from agent.replay_buffer import ReplayBuffer
import math
import numpy as np
class HPPO(object):
    def __init__(self, args, writer):
        super(HPPO, self).__init__()
        self.dis_lr = args.discrete_LR
        self.con_lr = args.continuous_LR
        self.actor = HybridActor(args.state_dim, args.discrete_dim, args.continuous_dim)
        self.optimizer_actor_dis = Adam(params=self.actor.parameters(), lr=self.dis_lr, eps=1e-6)
        self.optimizer_actor_con = Adam(params=self.actor.parameters(), lr=self.con_lr, eps=1e-6)

        self.critic_lr = args.critic_LR
        self.critic = Critic(args.state_dim)
        self.optimizer_critic = Adam(params=self.critic.parameters(), lr=self.critic_lr, eps=1e-5)
        self.critic_criterion = nn.MSELoss()

        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.lamda = args.lamda
        self.gamma = args.gamma
        self.epochs = args.epochs
        self.entropy_coef = args.entropy_coef
        self.epsilon = args.epsilon
        self.a_max = args.amax
        self.a_min = args.amin

        self.training_step = 0
        self.max_grad_norm = 0.5
        self.total_steps = args.EPISODES
        self.max_step = 1e8
        self.writer = writer
        self.buffer = ReplayBuffer(self.buffer_size)
        self.path = args.modelPath
    def choose_action(self, state):
        with torch.no_grad():
            discrete_dis = self.actor.get_discrete_dist(state)
            # continuous_dis = self.actor.get_continuous_dist(state)
        # discrete action
        discrete_action = discrete_dis.sample()
        discrete_log_prob = discrete_dis.log_prob(discrete_action)
        discrete_log_prob = discrete_log_prob.numpy()

        # continuous action
        with torch.no_grad():
            continuous_dis = self.actor.get_continuous_dist(state, torch.Tensor([discrete_action]))
        continuous_action = continuous_dis.sample()
        continuous_log_prob = continuous_dis.log_prob(continuous_action)
        continuous_log_prob = continuous_log_prob.numpy()
        continuous_action = continuous_action.clamp(-self.a_min, self.a_max)
        return discrete_action, discrete_log_prob, continuous_action, continuous_log_prob

    def model_save(self, i):
        """
        :param episode: 当前智能体所处回合
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        os.makedirs(self.path + '/' + str(i)+ "/actor_model")
        os.makedirs(self.path +  '/' + str(i)+ "/critic_model")
        torch.save(self.actor.state_dict(), self.path + '/' + str(i)+ '/actor_model/' + 'actor.pkl')
        torch.save(self.critic.state_dict(), self.path + '/' + str(i)+ '/critic_model/' + 'critic.pkl')
    def model_load(self):
        self.actor.load_state_dict(torch.load("E:/Program Files/PyWorkstation/AF_GLOSA/models/model081401/20000/actor_model/actor.pkl"))
        self.critic.load_state_dict(torch.load("E:/Program Files/PyWorkstation/AF_GLOSA/models/model081401/20000/critic_model/critic.pkl"))

    def update(self): #
        state, a_dis, a_dis_log, a_con, a_con_log, reward, state_, done = self.buffer.sample(self.batch_size)
        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(state)  # value function of state
            vs_next = self.critic(state_)  # value function of next state
            deltas = (reward.view(-1, 1) + self.gamma * (1.0 - done.view(-1, 1)) * vs_next) - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-6))
        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                self.training_step += 1
                # lane change control
                new_discrete_dist = self.actor.get_discrete_dist(state[index])
                new_discrete_entropy = new_discrete_dist.entropy().view(-1, 1)
                new_discrete_prob_log = new_discrete_dist.log_prob(a_dis[index].squeeze()).view(-1, 1)
                discrete_ratios = torch.exp(new_discrete_prob_log - a_dis_log[index])
                discrete_surr1 = discrete_ratios * adv[index]
                discrete_surr2 = torch.clamp(discrete_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                discrete_loss = -(torch.min(discrete_surr1, discrete_surr2) + self.entropy_coef * new_discrete_entropy).mean()
                self.optimizer_actor_dis.zero_grad()
                discrete_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor_dis.step()
                # self.writer.add_scalar('loss/discrete_actorSc_loss', discrete_loss, global_step=self.training_step)
                self.writer.add_scalar('discrete_entropy', new_discrete_entropy.mean(), global_step=self.training_step)
                # continuous acc control
                new_dist = self.actor.get_continuous_dist(state[index], a_dis[index].view(len(a_dis[index]), -1))
                new_continuous_entropy = new_dist.entropy().sum(1, keepdim=True)
                new_prob_log = new_dist.log_prob(a_con[index])
                continuous_ratios = torch.exp(new_prob_log.sum(1, keepdim=True) - a_con_log[index].sum(1, keepdim=True))
                continuous_surr1 = continuous_ratios * adv[index]
                continuous_surr2 = torch.clamp(continuous_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                continuous_loss = -(torch.min(continuous_surr1, continuous_surr2) + self.entropy_coef * new_continuous_entropy).mean()
                self.optimizer_actor_con.zero_grad()
                continuous_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor_con.step()
                # self.writer.add_scalar('loss/continuous_actor_loss', continuous_loss, global_step=self.training_step)
                self.writer.add_scalar('continuous_entropy', new_continuous_entropy.mean(), global_step=self.training_step)
                # critic
                v_s = self.critic(state[index])
                critic_loss = self.critic_criterion(v_s, v_target[index]).mean()
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()
                # self.writer.add_scalar('loss/critic_loss', critic_loss, global_step=self.training_step)

        self.lr_decay(self.total_steps)  # decay of learn rate
        self.buffer.clear()

    def evaluate(self, s):
        dis_action = self.actor.eval_discrete_dist(s)
        dis_action = torch.argmax(dis_action)
        mean, std = self.actor.eval_continuous_dist(s, torch.Tensor([dis_action]))
        con_action = (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-((mean - mean) ** 2) / (2 * (std ** 2)))
        return dis_action, con_action

    def lr_decay(self, total_steps):
        lr_dis_sc_now = self.dis_lr * (1 - total_steps / self.max_step)
        lr_con_sc_now = self.con_lr * (1 - total_steps / self.max_step)
        lr_cri_now = self.critic_lr * (1 - total_steps / self.max_step)
        for p in self.optimizer_actor_dis.param_groups:
            p['lr'] = lr_dis_sc_now
        for p in self.optimizer_actor_con.param_groups:
            p['lr'] = lr_con_sc_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_cri_now
