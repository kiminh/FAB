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

def hidden_init(layer):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (0, lim)

class Actor(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Actor, self).__init__()

        neuron_nums = [neural_nums_a_1, neural_nums_a_2]
        self.batch_input = nn.BatchNorm1d(feature_numbers)
        self.batch_input.weight.data.fill_(1)
        self.batch_input.bias.data.fill_(0)

        self.mlp = nn.Sequential(
            nn.Linear(feature_numbers, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], action_numbers),
            # nn.BatchNorm1d(action_numbers),
            nn.Tanh()
        )

        self.reset_parameters()

    def reset_parameters(self):
        # self.mlp[1].weight.data.uniform_(*hidden_init(self.mlp[1]))
        #
        # self.mlp[3].weight.data.uniform_(*hidden_init(self.mlp[3]))

        self.mlp[4].weight.data.uniform_(-0.003, 0.003)

    def forward(self, input):
        bn_input = self.batch_input(input)
        out = self.mlp(bn_input)

        return out

class Critic(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Critic, self).__init__()

        self.batch_input = nn.BatchNorm1d(feature_numbers)
        self.batch_input.weight.data.fill_(1)
        self.batch_input.bias.data.fill_(0)

        neuron_nums = [neural_nums_c_1, neural_nums_c_2]
        self.mlp_1 = nn.Sequential(
            nn.Linear(feature_numbers + action_numbers, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], 1)
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(feature_numbers + action_numbers, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], 1)
        )

        # self.reset_parameters()

    def reset_parameters(self):
        self.mlp_1[0].weight.data.uniform_(*hidden_init(self.mlp_1[0]))
        self.mlp_2[0].weight.data.uniform_(*hidden_init(self.mlp_2[0]))

        self.mlp_1[2].weight.data.uniform_(*hidden_init(self.mlp_1[2]))
        self.mlp_2[2].weight.data.uniform_(*hidden_init(self.mlp_2[2]))

        # self.mlp_1[4].weight.data.uniform_(-0.003, 0.003)
        # self.mlp_2[4].weight.data.uniform_(-0.003, 0.003)

    def evaluate(self, input, action):
        bn_input = self.batch_input(input)
        cat_x = torch.cat([bn_input, action], dim=-1)

        q_1 = self.mlp_1(cat_x)
        q_2 = self.mlp_2(cat_x)

        return q_1, q_2

    def evaluate_1(self, input, action):
        bn_input = self.batch_input(input)
        cat_x = torch.cat([bn_input, action], dim=-1)

        q_1 = self.mlp_1(cat_x)

        return q_1