'''
定义一些超参数
'''
import os
from src.data_type import config as data_type

log_path = 'result/' + data_type['campaign_id'] + data_type['type'] + '/'
if not os.path.exists('result/'):
    os.mkdir('result/')
if not os.path.exists('result/' + data_type['campaign_id']):
    os.mkdir('result/' + data_type['campaign_id'])
if not os.path.exists(log_path):
    os.mkdir(log_path)

config = {
    'e_greedy': 1,
    'learning_rate': 0.001,
    'pg_learning_rate': 1e-3,
    'reward_decay': 1.0,
    'feature_num': 7,
    'state_feature_num': 7,
    'budget_para': [1/2],
    'train_episodes': 1000,
    'neuron_nums': 100,
    'relace_target_iter': 100,
    'memory_size': 100000,
    'batch_size': 32,
    'device': 'cuda:0',
}