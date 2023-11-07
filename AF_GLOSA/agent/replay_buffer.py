import collections
import random
import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, buffer_size, update_frequency=2, start_read_memory_step=700):
        super(ReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.start_read_memory_step = start_read_memory_step
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def is_full(self):
        return self.__len__() == self.buffer_size

    def append(self, state, a_dis, a_prob_dis, a_con, a_prob_con, reward, state_, done):
        state = state.numpy().tolist()
        a_dis = a_dis.tolist()
        a_prob_dis = a_prob_dis.tolist()
        a_con = a_con.tolist()
        a_prob_con = a_prob_con.tolist()
        state_ = np.array(state_)
        step_data = (state, a_dis, a_prob_dis, a_con, a_prob_con, reward, state_, done)
        self.buffer.append(step_data)

    def sample(self, batch):
        batch_data = random.sample(self.buffer, batch)
        obs, discrete_action, discrete_action_log_prob, continuous_action, continuous_action_log_prob, reward, obs_next, done = zip(*batch_data)
        obs = self.list_to_tensor(obs)
        discrete_action = self.list_to_tensor(discrete_action)
        discrete_action_log_prob = self.list_to_tensor(discrete_action_log_prob)
        continuous_action = self.list_to_tensor(continuous_action)
        # continuous_action = continuous_action.unsqueeze()
        continuous_action_log_prob = self.list_to_tensor(continuous_action_log_prob)
        reward = self.list_to_tensor(reward)
        obs_next = self.list_to_tensor(obs_next)
        done = self.list_to_tensor(done)
        return obs, discrete_action, discrete_action_log_prob, continuous_action, continuous_action_log_prob, reward, obs_next, done

    @classmethod
    def list_to_tensor(cls, x):
        x = list(x)
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.type(torch.float32)
        return x
