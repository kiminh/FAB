import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from src.FAB_NO_BN.config import config
from src.data_type import config as data_type

log_path = data_type['campaign_id'] + data_type['type']
if not os.path.exists(data_type['campaign_id']):
    os.mkdir(data_type['campaign_id'])

if not os.path.exists(log_path):
    os.mkdir(log_path)

if not os.path.exists(log_path + '/result_reward_1/'):
    os.mkdir(log_path + '/result_reward_1/')

if not os.path.exists(log_path + '/result_reward_2/'):
    os.mkdir(log_path + '/result_reward_2/')

if not os.path.exists(log_path + '/result_reward_3/'):
    os.mkdir(log_path + '/result_reward_3/')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

neural_nums_a_1 = config['neuron_nums_a_1']
neural_nums_a_2 = config['neuron_nums_a_2']
neural_nums_c_1 = config['neuron_nums_c_1']
neural_nums_c_2 = config['neuron_nums_c_2']

class Actor(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_numbers, neural_nums_a_1)
        self.fc2 = nn.Linear(neural_nums_a_1, neural_nums_a_2)
        self.out = nn.Linear(neural_nums_a_2, action_numbers)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x_ = F.relu(self.fc2(x))
        out = torch.tanh(self.out(x_))

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(feature_numbers, neural_nums_c_1)
        self.fc_a = nn.Linear(action_numbers, neural_nums_c_1)
        self.fc_q = nn.Linear(2 * neural_nums_c_1, neural_nums_c_2)
        self.fc_ = nn.Linear(neural_nums_c_2, 1)

    def forward(self, input, action):
        f_s = F.relu(self.fc_s(input))
        f_a = self.fc_a(action)
        cat = torch.cat([f_s, f_a], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_(q)

        return q