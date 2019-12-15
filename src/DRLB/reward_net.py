import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

class Net(nn.Module):
    def __init__(self, feature_numbers, action_numbers, reward_numbers):
        super(Net, self).__init__()

        # 第一层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_1 = 100
        # 第二层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_2 = 100

        self.fc1 = nn.Linear(feature_numbers + action_numbers, neuron_numbers_1)
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.fc2 = nn.Linear(neuron_numbers_1, neuron_numbers_2)
        self.fc2.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化
        self.out = nn.Linear(neuron_numbers_1, reward_numbers)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x_1 = self.fc1(input)
        x_1 = F.relu(x_1)
        x_2 = self.fc2(x_1)
        x_2 = F.relu(x_2)
        actions_value = self.out(x_2)

        return actions_value

class RewardNet:
    def __init__(
        self,
        action_space,
        action_numbers,
        reward_numbers,
        feature_numbers,
        learning_rate = 0.01,
        memory_size = 500,
        batch_size = 32,
        device = 'cuda:0',
    ):
        self.action_space = action_space
        self.action_numbers = action_numbers
        self.reward_numbers = reward_numbers
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device

        if not hasattr(self, 'memory_S_counter'):
            self.memory_S_counter = 0

        if not hasattr(self, 'memory_D_counter'):
            self.memory_D_counter = 0

        # 将经验池<状态-动作-累积奖励>中的转换组初始化为0
        self.memory_S = defaultdict()

        # 将经验池<状态-动作-累积奖励中最大>中的转换组初始化为0
        self.memory_D = np.zeros((self.memory_size, self.feature_numbers + 2))

        self.model_reward, self.real_reward = Net(self.feature_numbers, self.action_numbers, self.reward_numbers).to(self.device), Net(self.feature_numbers, self.action_numbers, self.reward_numbers).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model_reward.parameters(), lr=self.lr, weight_decay=1e-3)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

    def return_model_reward(self, state_action):
        # 统一 observation 的 shape (1, size_of_observation)
        state_action = torch.unsqueeze(torch.FloatTensor(state_action), 0).to(self.device)

        with torch.no_grad():
            model_reward = self.model_reward.forward(state_action).cpu().numpy()
        return model_reward

    def store_S_pair(self, state_action_pair, reward):
        self.memory_S[state_action_pair] = reward

    def get_reward_from_S(self, state_action_pair):
        return self.memory_S.get(state_action_pair, 0)
    
    def store_D_pair(self, state, action, reward):
        state_action_reward_pair = np.hstack((state, action, reward))
        
        index = self.memory_D_counter % self.memory_size
        self.memory_D[index, :] = state_action_reward_pair
        self.memory_D_counter += 1

    def learn(self):
        if self.memory_D_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_D_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory_D[sample_index, :]

        state_actions = torch.FloatTensor(batch_memory[:, :self.feature_numbers+1]).to(self.device)
        real_reward = torch.unsqueeze(torch.FloatTensor(batch_memory[:, self.feature_numbers + 1]), 1).to(self.device)

        model_reward = self.model_reward.forward(state_actions)

        loss = self.loss_func(model_reward, real_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
