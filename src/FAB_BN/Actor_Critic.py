import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from src.FAB_BN.config import config
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
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.out.weight.data)

        self.batch_norm_input = nn.BatchNorm1d(feature_numbers,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

        self.batch_norm_layer_1 = nn.BatchNorm1d(neural_nums_a_1,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

        self.batch_norm_layer_2 = nn.BatchNorm1d(neural_nums_a_2,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 track_running_stats=True)

        self.batch_norm_action = nn.BatchNorm1d(action_numbers,
                     eps=1e-05,
                     momentum=0.1,
                     affine=True,
                     track_running_stats=True)

    def forward(self, input):
        x = F.relu(self.batch_norm_layer_1(self.fc1(self.batch_norm_input(input))))
        x_ = F.relu(self.batch_norm_layer_2(self.fc2(x)))
        out = torch.tanh(self.batch_norm_action(self.out(x_)))

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(feature_numbers, neural_nums_c_1)
        self.fc_a = nn.Linear(action_numbers, neural_nums_c_1)
        self.fc_q = nn.Linear(2 * neural_nums_c_1, neural_nums_c_2)
        self.fc_ = nn.Linear(neural_nums_c_2, action_numbers)
        nn.init.xavier_normal_(self.fc_s.weight.data)
        nn.init.xavier_normal_(self.fc_a.weight.data)
        nn.init.xavier_normal_(self.fc_q.weight.data)
        nn.init.xavier_normal_(self.fc_.weight.data)

        self.batch_norm_input = nn.BatchNorm1d(feature_numbers,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        self.batch_norm_layer_1 = nn.BatchNorm1d(neural_nums_c_1,
                                               eps=1e-05,
                                               momentum=0.1,
                                               affine=True,
                                               track_running_stats=True)

        self.batch_norm_layer_2 = nn.BatchNorm1d(neural_nums_c_2,
                                                 eps=1e-05,
                                                 momentum=0.1,
                                                 affine=True,
                                                 track_running_stats=True)

    def forward(self, input, action):
        xs = F.relu(self.fc_s(self.batch_norm_input(input)))
        x = torch.cat([self.batch_norm_layer_1(xs), self.fc_a(action)], dim=1)
        q = F.relu(self.batch_norm_layer_2(self.fc_q(x)))
        q = self.fc_(q)

        return q