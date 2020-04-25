import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from src.data_type import config as data_type

def list_metrics(budget_para, result_directory):
    bid_records = pd.read_csv(result_directory + '/bids_' + str(budget_para) + '.csv')
    columns = ['actions', 'prices', 'clicks', 'hours', 'thetas']
    bid_records = bid_records[columns]
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 1:4] = bid_records.iloc[:, 1:4].astype(int)

    hour_clk_records = []
    hour_cost_records = []
    hour_cpc_records = []
    hour_imp_records = []
    hour_bid_nums_records = []
    for hour_clip in range(24):
        hour_records = bid_records[bid_records.iloc[:, 3].isin([hour_clip])]
        hour_records = hour_records.values

        win_records = hour_records[hour_records[:, 0] >= hour_records[:, 1]]

        hour_bid_nums = len(hour_records)
        hour_clks = np.sum(win_records[:, 2])
        hour_costs = np.sum(win_records[:, 1])
        hour_cpc = hour_costs / hour_clks if hour_clks > 0 else 0
        hour_imps = len(win_records)

        hour_bid_nums_records.append(hour_bid_nums)
        hour_clk_records.append(hour_clks)
        hour_cost_records.append(hour_costs)
        hour_cpc_records.append(hour_cpc)
        hour_imp_records.append(hour_imps)

    print(hour_clk_records)
    print(hour_cost_records)
    print(hour_cpc_records)
    print(hour_imp_records)
    print(hour_bid_nums_records)

    records = [hour_clk_records, hour_cost_records, hour_cpc_records, hour_imp_records, hour_bid_nums_records]

    for k in range(5):
        current_str = ''
        for m in range(len(hour_clk_records)):
            current_str = current_str + str(records[k][m]) + '\t'

        print(current_str)

def action_distribution(budget_para, result_directory):
    bid_records = pd.read_csv(result_directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 1:4] = bid_records.iloc[:, 1:4].astype(int)

    # 平均价格
    avg_actions = np.average(bid_records.iloc[:, 0])
    avg_market_prices = np.average(bid_records.iloc[:, 1])
    print(avg_actions, avg_market_prices)

    # 0-300价格区间的每个价格的个数
    action_dicts = {}
    market_price_dicts = {}
    for i in range(301):
        action_dicts.setdefault(i, 0)
        market_price_dicts.setdefault(i, 0)

    action_records = bid_records.iloc[:, 0].values
    market_price_records = bid_records.iloc[:, 1].values
    for i in range(len(action_records)):
        action_dicts[math.floor(action_records[i])] += 1 # math.ceil(x) 向上取整， math.floor(x)向上取整
        market_price_dicts[market_price_records[i]] += 1
    print(action_dicts)
    print(market_price_dicts)

    x_axis = list(action_dicts.keys())
    action_y_axis = list(action_dicts.values())
    market_price_y_axis = list(market_price_dicts.values())

    plt.plot(x_axis, action_y_axis, 'r')
    plt.plot(x_axis, market_price_y_axis, 'b')
    plt.legend()
    plt.show()

    action_i_sum = [0 for k in range(301)]
    market_price_i_sum = [0 for m in range(301)]

    # 0-300各个价格的累加
    for i in range(301):
        action_i_sum[i] = np.sum(list(action_y_axis)[:i])
        market_price_i_sum[i] = np.sum(list(market_price_y_axis)[:i])

    plt.plot(x_axis, action_i_sum, 'r')
    plt.plot(x_axis, market_price_i_sum, 'b')
    plt.legend()
    plt.show()

    price_nums = {'action_nums':action_y_axis, 'market_price_nums': market_price_y_axis, 'action_cumulative_nums': action_i_sum, 'market_price_cumulative_nums': market_price_i_sum}
    price_nums_df = pd.DataFrame(data=price_nums)
    price_nums_df.to_csv('price_nums.csv')

    f_i = open('price_nums.csv')
    for line in f_i:
        print(line.replace(',', '\t').strip())

def clk_frequency(budget_para, result_directory):
    bid_records = pd.read_csv(result_directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 0] = np.ceil(bid_records.iloc[:, 0].values) # 向下取整
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(int)
    bid_records.iloc[:, 1:4] = bid_records.iloc[:, 1:4].astype(int)

    record_clk_indexs = [300]
    hour_appear_arrays = bid_records[bid_records.iloc[:, 0].isin(record_clk_indexs)].iloc[:, 3]
    is_in_indexs = bid_records[bid_records.iloc[:, 0].isin(record_clk_indexs)].iloc[:, 3].unique() # 有哪些时段出现了record_clk_index
    index_dicts = {}
    for index in is_in_indexs:
        index_dicts.setdefault(index, 0)

    for hour_appear in hour_appear_arrays.values:
        index_dicts[hour_appear] += 1

    print('出价价格300出现次数最多时段排序')
    sort_300_index = np.argsort(-np.array(list(index_dicts.values())))
    dict_keys = []
    for i in range(len(sort_300_index)):
        dict_keys.append(list(index_dicts.keys())[sort_300_index[i]])
    print(dict_keys) # 出价价格300出现次数最多时段排序
    print(index_dicts)

def ctr_statistics(budget_para, result_directory):
    bid_records = pd.read_csv(result_directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 0] = np.ceil(bid_records.iloc[:, 0].values)  # 向下取整
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(int)
    bid_records.iloc[:, 1:4] = bid_records.iloc[:, 1:4].astype(int)
    bid_records.iloc[:, 4] = bid_records.iloc[:, 4].astype(float)

    print(bid_records.iloc[:, 4].median())  # 返回中位数

    bid_price_index = [300]

    ctr_records = bid_records[bid_records.iloc[:, 0].isin(bid_price_index)].iloc[:, 4]
    print(np.sum(ctr_records) / len(ctr_records))

# 由启发式算法得到最优eCPC 1458-60920.22773088766,38767.41764692851,33229.21512593873, 22152.81008395915‬
# 3386-77901.22125145316‬,47939.21307781733,35954.409808363,23969.60653890866‬

budget_paras = [0.0625]
campaign_id = data_type['campaign_id']
project_name = 'RLB'

result_file = data_type['type']

log_path = '../' + project_name + '/result/'

result_directory = log_path + campaign_id + result_file

heuristic_result_path = '../heuristic_algo/result/' + campaign_id + result_file + '/results_train.best.perf.txt'

print('\n##########To Bids.csv files##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    clk_frequency(budget_para, result_directory)

print('\n##########Time slots’ctr statistics##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    ctr_statistics(budget_para, result_directory)

print('\n##########List Metrics##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    list_metrics(budget_para, result_directory)

# print('\n##########Action Distribution##########')
# for budget_para in budget_paras:
#     print('\n------budget_para:{}------'.format(budget_para))
#     action_distribution(budget_para, result_directory)
#
# os.remove('price_nums.csv')
# print('price_nums.csv file has been removed')