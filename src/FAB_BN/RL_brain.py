import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from src.FAB_BN.Actor_Critic import Actor, Critic

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class DDPG():
    def __init__(
            self,
            feature_nums,
            action_nums,
            lr_A,
            lr_C,
            reward_decay,
            memory_size,
            batch_size = 32,
            tau = 0.005, # for target network soft update
            device = 'cuda:0',
    ):
        self.feature_numbers = feature_nums
        self.action_numbers = action_nums
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        # 设置随机数种子
        setup_seed(1)

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + self.action_numbers + 1 + 1))

        self.Actor = Actor(self.feature_numbers, self.action_numbers).to(self.device)
        self.Critic = Critic(self.feature_numbers, self.action_numbers).to(self.device)

        self.Actor_ = copy.deepcopy(self.Actor)
        self.Critic_ = copy.deepcopy(self.Critic)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.learn_iter = 0

    def store_transition(self, transition):
        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换
        self.memory_counter += 1

    def choose_action(self, state, ):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state)
            action = torch.clamp(action + torch.randn_like(action) * 0.1, -0.99, 0.99)

        self.Actor.train()

        return action.cpu().numpy()[0][0]

    def choose_best_action(self, state, ):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        self.Actor.eval()
        with torch.no_grad():
            action = torch.clamp(self.Actor.forward(state), -0.99, 0.99)

        self.Actor.train()

        return action.cpu().numpy()[0][0]

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self):
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = random.sample(range(self.memory_size), self.batch_size)
        else:
            sample_index = random.sample(range(self.memory_counter), self.batch_size)

        batch_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch_memory[:, :self.feature_numbers]).to(self.device)
        b_a = torch.FloatTensor(batch_memory[:, self.feature_numbers: self.feature_numbers + self.action_numbers]).to(self.device)
        b_r = torch.FloatTensor(batch_memory[:, self.feature_numbers + self.action_numbers]).unsqueeze(1).to(self.device)
        b_s_ = torch.FloatTensor(batch_memory[:, self.feature_numbers + self.action_numbers + 1: 2 * self.feature_numbers + self.action_numbers + 1]).to(self.device)
        b_dones = torch.FloatTensor(batch_memory[:, -1]).unsqueeze(1).to(self.device)
        # print(b_r)

        with torch.no_grad():
            a_b_s_ = self.Actor_(b_s_)
            q1_target, q2_target = \
                self.Critic_.evaluate(b_s_, torch.clamp(a_b_s_ + torch.clamp(torch.randn_like(a_b_s_) * 0.2, -0.5, 0.5), -0.99, 0.99))
            # print(q1_target, q2_target)
            q_target = torch.min(q1_target, q2_target)
            q_target = b_r + self.gamma * torch.mul(q_target, 1 - b_dones)

        q1, q2 = self.Critic.evaluate(b_s, b_a)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.Critic.parameters(), 0.5)
        self.optimizer_c.step()

        self.learn_iter += 1

        a_loss_r = 0
        if self.learn_iter % 3 == 0:
            a_b_s = self.Actor(b_s)
            # print(a_b_s)
            # print(a_b_s)
            a_loss = -self.Critic.evaluate_1(b_s, a_b_s).mean() + (a_b_s ** 2).mean() * 1e-2

            a_loss_r = a_loss.item()
            self.optimizer_a.zero_grad()
            a_loss.backward()
            # nn.utils.clip_grad_norm_(self.Actor.parameters(), 0.5)
            # for name, parms in net.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #           ' -->grad_value:', parms.grad)
            self.optimizer_a.step()

            self.soft_update(self.Actor, self.Actor_)
            self.soft_update(self.Critic, self.Critic_)

        td_error_r = critic_loss.item()
        return td_error_r, a_loss_r

    # 只存储获得最优收益（点击）那一轮的参数
    def para_store_iter(self, test_results):
        max = 0
        if len(test_results) >= 1:
            for i in range(len(test_results)):
                if i == 0:
                    max = test_results[i][3]
                elif i != len(test_results) - 1:
                    if test_results[i][3] > test_results[i - 1][3] and test_results[i][3] > test_results[i + 1][3]:
                        if max < test_results[i][3]:
                            max = test_results[i][3]
                else:
                    if test_results[i][3] > max:
                        max = test_results[i][3]
        return max

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x