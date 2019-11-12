config = {
    'learning_rate_a': 1e-4,
    'learning_rate_c': 1e-3,
    'reward_decay': 1,
    'exploration_rate': 1,
    'tau': 0.001,
    'feature_num': 4, # 153,3
    'data_pctr_index': 4, # 0
    'data_hour_index': 3, # 17:train-fm,3
    'data_clk_index': 1, # 15:train-fm,1
    'data_marketprice_index': 2, # 16:train-fm,2
    'data_feature_index': 1, # 15:train-fm,1
    'state_feature_num': 1, #,1
    # ctr 预测参数：./ffm-train -l 0.00001 -k 10 -t 20 -r 0.03 -s {nr_thread} {save}train_{data_name}_{day}.ffm
    'budget_para': [1/16],
    'train_episodes': 50000,
    'neuron_nums_c_1': 50,
    'neuron_nums_c_2': 40,
    'neuron_nums_a_1': 30,
    'neuron_nums_a_2': 20,
    'GPU_fraction': 1,
    'learn_iter': 24,
    'observation_size': 240,
    'memory_size': 1000000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}