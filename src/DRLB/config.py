'''
定义一些超参数
'''
import os
from src.data_type import config as data_type

log_path = 'result/' + data_type['campaign_id'] + data_type['type'] + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

config = {
    'e_greedy': 1,
    'learning_rate': 0.001,
    'pg_learning_rate': 1e-3,
    'reward_decay': 1,
    'feature_num': 7,
    'state_feature_num': 7,
    'campaign_id': '1458',
    'budget_para': [1/16],
    'train_episodes': 1000,
    'neuron_nums': 100,
    'relace_target_iter': 100,
    'memory_size': 100000,
    'batch_size': 32,
}