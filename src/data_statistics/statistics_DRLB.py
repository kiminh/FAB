import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from src.DRLB.config import config
from src.data_type import config as data_type

def bid_func(auc_pCTRS, lamda):
    return auc_pCTRS/ lamda

def statistics(B_t, origin_t_spent, origin_t_win_imps,
               origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t,  auc_t_datas, bid_arrays, remain_auc_num, t):
    cpc = 30000
    if B_t[t] > 0:
        if B_t[t] - origin_t_spent <= 0 or remain_auc_num[t] - origin_t_auctions <= 0:
            temp_t_auctions = 0
            temp_t_spent = 0
            temp_t_win_imps = 0
            temp_reward_t = 0
            temp_t_clks = 0
            temp_profit_t = 0
            for i in range(len(auc_t_datas)):
                temp_t_auctions += 1
                if remain_auc_num[t] - temp_t_auctions >= 0:
                    if B_t[t] - temp_t_spent >= 0:
                        if auc_t_datas.iloc[i, 2] <= bid_arrays[i]:
                            temp_t_spent += auc_t_datas.iloc[i, 2]
                            temp_t_win_imps += 1
                            temp_t_clks += auc_t_datas.iloc[i, 0]
                            temp_profit_t += (auc_t_datas.iloc[i, 1] * cpc - auc_t_datas.iloc[i, 2])
                            temp_reward_t += auc_t_datas.iloc[i, 0]
                    else:
                        break
                else:
                    break
            t_auctions = temp_t_auctions
            t_spent = temp_t_spent if temp_t_spent > 0 else 0
            t_win_imps = temp_t_win_imps
            t_clks = temp_t_clks
            reward_t = temp_reward_t
            profit_t = temp_profit_t
        else:
            t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t \
                = origin_t_spent, origin_t_win_imps, origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t
    else:
        t_auctions = 0
        t_spent = 0
        t_win_imps = 0
        reward_t = 0
        t_clks = 0
        profit_t = 0

    return t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t

def state_(budget, auc_num, auc_t_datas, auc_t_data_pctrs, lamda, B_t, time_t, remain_auc_num):
    cpc = 30000
    bid_arrays = bid_func(auc_t_data_pctrs, lamda)  # 出价
    bid_arrays = np.where(bid_arrays >= 300, 300, bid_arrays)
    win_auc_datas = auc_t_datas[auc_t_datas.iloc[:, 2] <= bid_arrays].values  # 赢标的数据
    t_spent = np.sum(win_auc_datas[:, 2])  # 当前t时段花费
    t_auctions = len(auc_t_datas)  # 当前t时段参与拍卖次数
    t_win_imps = len(win_auc_datas)  # 当前t时段赢标曝光数
    t_clks = np.sum(win_auc_datas[:, 0])
    profit_t = np.sum(win_auc_datas[:, 1] * cpc - win_auc_datas[:, 2])  # RewardNet
    reward_t = np.sum(np.multiply(win_auc_datas[:, 0], win_auc_datas[:, 1])) # 按论文中的奖励设置，作为直接奖励
    origin_reward_t = reward_t

    done = 0

    # BCR_t = 0
    if time_t == 0:
        if remain_auc_num[0] > 0:
            if remain_auc_num[0] - t_auctions <= 0:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, 0)
            else:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, 0)
        else:
            t_win_imps = 0
            t_spent = 0
            t_auctions = 0
            reward_t = 0
            t_clks = 0
            profit_t = 0

        B_t[0] = budget - t_spent
        if B_t[0] < 0:
            B_t[0] = 0
        remain_auc_num[0] = auc_num - t_auctions
        if remain_auc_num[0] < 0:
            remain_auc_num[0] = 0
        BCR_t_0 = (B_t[0] - budget) / budget
        BCR_t = BCR_t_0
    else:
        if remain_auc_num[time_t - 1] > 0:
            if remain_auc_num[time_t - 1] - t_auctions <= 0:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, time_t - 1)
            else:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, time_t - 1)
        else:
            t_auctions = 0
            t_spent = 0
            t_win_imps = 0
            reward_t = 0
            t_clks = 0
            profit_t = 0

        B_t[time_t] = B_t[time_t - 1] - t_spent
        if B_t[time_t] < 0:
            done = 1
            B_t[time_t] = 0
        remain_auc_num[time_t] = remain_auc_num[time_t - 1] - t_auctions
        if remain_auc_num[time_t] < 0:
            done = 1
            remain_auc_num[time_t] = 0
        BCR_t_current = (B_t[time_t] - B_t[time_t - 1]) / B_t[time_t - 1] if B_t[time_t - 1] > 0 else 0
        BCR_t = BCR_t_current

    ROL_t = 96 - time_t - 1
    CPM_t = t_spent / t_win_imps if t_spent != 0 else 0
    WR_t = t_win_imps / t_auctions if t_auctions > 0 else 0
    state_t = [time_t+1, B_t[time_t], ROL_t, BCR_t, CPM_t, WR_t, reward_t]

    t_real_clks = np.sum(auc_t_datas.iloc[:, 0])

    t_real_imps = len(auc_t_datas)

    return state_t, lamda, B_t, reward_t, origin_reward_t, profit_t, t_clks, bid_arrays, remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent, done, bid_arrays

def test_env(directory, budget_para, test_data, init_lamda, actions):
    test_data.iloc[:, [0, 2, 3]] = test_data.iloc[:, [0, 2, 3]].astype(int)
    test_data.iloc[:, [1]] = test_data.iloc[:, [1]].astype(float)

    # config['test_budget'] = np.sum(test_data.iloc[:, 2])
    config['test_budget'] = 32000000
    config['test_auc_num'] = len(test_data)

    auc_num = config['test_auc_num']
    budget = config['test_budget'] * budget_para

    B_t = [0 for i in range(96)]
    B_t[0] = budget * budget_para

    remain_auc_num = [0 for i in range(96)]
    remain_auc_num[0] = auc_num

    e_bids = []
    e_market_prices = []
    e_real_labels = []
    e_hours = []
    e_ctrs = []
    e_clks = []
    temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = 0, [], []

    for t in range(96):
        time_t = t

        # auc_data[0] 是否有点击；auc_data[1] pCTR；auc_data[2] 市场价格； auc_data[3] t划分[1-96]
        auc_t_datas = test_data[test_data.iloc[:, 3].isin([t + 1])]  # t时段的数据
        auc_t_data_pctrs = auc_t_datas.iloc[:, 1].values  # ctrs

        market_prices = auc_t_datas.iloc[:, 2].values
        hours = auc_t_datas.iloc[:, 3].values
        real_labels = auc_t_datas.iloc[:, 0].values

        if t == 0:
            state_t, lamda_t, B_t, reward_t, origin_reward_t, profit_t, t_clks, bid_arrays, t_remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent, done, bid_arrays = state_(
                budget, auc_num, auc_t_datas, auc_t_data_pctrs,
                init_lamda, B_t, time_t, remain_auc_num)  # 1时段
            action = actions[t]

            lamda_t_next = lamda_t * (1 + action)

            temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = lamda_t_next, B_t, t_remain_auc_num
        else:
            state_t, lamda_t, B_t, reward_t, origin_reward_t, profit_t, t_clks, bid_arrays, t_remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent, done, bid_arrays = state_(
                budget, auc_num, auc_t_datas, auc_t_data_pctrs,
                temp_lamda_t_next, temp_B_t_next, time_t, temp_remain_t_auctions)

            action = actions[t]

            lamda_t_next = lamda_t * (1 + action)

            temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = lamda_t_next, B_t, t_remain_auc_num

        e_clks = np.hstack((e_clks, t_clks))
        e_bids = np.hstack((e_bids, bid_arrays))
        e_real_labels = np.hstack((e_real_labels, real_labels))
        e_market_prices = np.hstack((e_market_prices, market_prices))
        e_hours = np.hstack((e_hours, hours))
        e_ctrs = np.hstack((e_ctrs, auc_t_data_pctrs))

        if done == 1:
            break

    print(np.sum(e_clks))
    records = {'bids': e_bids.tolist(), 'market_prices':e_market_prices.tolist(), 'clks': e_real_labels, 'hours': e_hours, 'ctrs': e_ctrs}
    records_df = pd.DataFrame(data=records)
    records_df.to_csv(directory + '/bids_' + str(budget_para) + '.csv', index=None)

def max_train_index(directory, para):
    train_results = pd.read_csv(directory + '/train_episode_results_' + str(para) + '.csv').values
    train_clks = train_results[:, [0 ,3]]
    test_results = pd.read_csv(directory + '/test_episode_results_' + str(para) + '.csv').values
    test_clks = test_results[:, [0, 3]]

    every_ten_times = []
    for i in range(1000):
        if i == 0:
            continue
        if (i + 1) % 10 == 0:
            every_ten_times.append(i)

    train_clks_temp = train_clks[every_ten_times, :]

    # 每10轮测试一轮
    max_value = train_clks_temp[train_clks[every_ten_times, 1].argsort()][-1, 1]
    max_value_indexs = train_clks_temp[train_clks[every_ten_times, 1] == max_value]

    max_test_value = []
    max_test_value_index = []
    for index in max_value_indexs:
        max_test_value.append(test_clks[int(index[0]), 1])
        max_test_value_index.append(int(index[0]))

    test_value_max_index = np.argmax(max_test_value)  # max_test_value 最大值的索引

    max_test_result = test_results[max_test_value_index[test_value_max_index], :].tolist()
    print(max_test_result)
    return int(max_test_result[0])

def choose_init_lamda(budget_para, campaign, original_ctr, heuristic_result_path):
    results_train_best = open(heuristic_result_path, 'r')
    train_best_bid = {}
    for i, line in enumerate(results_train_best):
        if i == 0:
            continue
        line_array = line.strip().split('\t')
        train_best_bid.setdefault(int(line_array[0]), int(line_array[-1]))

    if budget_para == 0.5:
        init_lamda = original_ctr / train_best_bid[2]
    elif budget_para == 0.25:
        init_lamda = original_ctr / train_best_bid[4]
    elif budget_para == 0.125:
        init_lamda = original_ctr / train_best_bid[8]
    else:
        init_lamda = original_ctr / train_best_bid[16]

    return init_lamda

def to_bids(is_sample, budget_para, campaign_id, result_directory):
    train_data = pd.read_csv('../DRLB/data/' + campaign_id + 'train_DRLB_' + data_type['type'] + '.csv',
                             header=None).drop([0])
    train_data.iloc[:, [0, 2, 3]] = train_data.iloc[:, [0, 2, 3]].astype(int)
    train_data.iloc[:, [1]] = train_data.iloc[:, [1]].astype(float)

    # config['train_budget'] = np.sum(train_data.iloc[:, 2])
    config['train_budget'] = 32000000
    config['train_auc_num'] = len(train_data)
    original_ctr = np.sum(train_data.iloc[:, 0]) / len(train_data)

    test_data = pd.read_csv('../DRLB/data/' + campaign_id + 'test_DRLB_' + is_sample + '.csv', header=None).drop([0])

    max_result_index = max_train_index(result_directory, budget_para)
    print(max_result_index)
    init_lamda = choose_init_lamda(budget_para, campaign_id, original_ctr, heuristic_result_path)

    action_file = result_directory + '/test_episode_actions_' + str(budget_para) + '.csv'
    actions_df = pd.read_csv(action_file, header=None).drop([0])

    actions = actions_df.iloc[max_result_index, 1:].tolist()

    test_env(result_directory, budget_para, test_data, init_lamda, actions)

def time_fraction_to_time_slot(records):
    new_records = []
    for i, record in enumerate(records):
        if i == 0:
            continue

        if i % 4 == 0:
            new_records.append(np.sum(records[i-3:i+1]))
        elif i == 95:
            new_records.append(np.sum(records[93:i+1]))

    return new_records

def list_metrics(budget_para, result_directory):
    bid_records = pd.read_csv(result_directory + '/bids_' + str(budget_para) + '.csv', header=None).drop([0])
    bid_records.iloc[:, 0] = bid_records.iloc[:, 0].astype(float)
    bid_records.iloc[:, 1:4] = bid_records.iloc[:, 1:4].astype(int)

    hour_clk_records = []
    hour_cost_records = []
    hour_cpc_records = []
    hour_imp_records = []
    hour_bid_nums_records = []

    current_auc_nums = [0 for i in range(96)]
    current_costs = [0 for i in range(96)]
    for hour_clip in range(96):
        hour_records = bid_records[bid_records.iloc[:, 3].isin([hour_clip])]
        hour_records = hour_records.values

        win_records = hour_records[hour_records[:, 0] >= hour_records[:, 1]]

        hour_bid_nums = len(hour_records)
        hour_clks = np.sum(win_records[:, 2])
        hour_costs = np.sum(win_records[:, 1])
        hour_cpc = hour_costs / hour_clks if hour_clks > 0 else 0
        hour_imps = len(win_records)

        current_auc_nums[hour_clip] = len(hour_records)
        current_costs[hour_clip] = hour_costs
        if np.sum(current_costs) >= config['test_budget'] * budget_para or np.sum(current_auc_nums) > config['test_auc_num']:
            hour_clks = 0
            hour_costs = 0
            hour_imps = 0
            hour_bid_nums = 0
            current_auc_nums[hour_clip] = 0
            current_costs[hour_clip] = 0
            for hour_record in hour_records:
                if np.sum(current_costs) >= config['test_budget'] * budget_para or np.sum(current_auc_nums) > config[
                    'test_auc_num']:
                    break
                hour_bid_nums += 1

                if hour_record[0] >= hour_record[1]:
                    hour_clks += hour_record[2]
                    hour_costs += hour_record[1]
                    current_costs[hour_clip] += hour_record[1]
                    hour_imps += 1
                current_auc_nums[hour_clip] += 1
            hour_cpc = hour_costs / hour_clks if hour_clks > 0 else 0

        hour_bid_nums_records.append(hour_bid_nums)
        hour_clk_records.append(hour_clks)
        hour_cost_records.append(hour_costs)
        hour_cpc_records.append(hour_cpc)
        hour_imp_records.append(hour_imps)

    hour_bid_nums_records = time_fraction_to_time_slot(hour_bid_nums_records)
    hour_clk_records = time_fraction_to_time_slot(hour_clk_records)
    hour_cost_records = time_fraction_to_time_slot(hour_cost_records)
    hour_cpc_records = time_fraction_to_time_slot(hour_cpc_records)
    hour_imp_records = time_fraction_to_time_slot(hour_imp_records)

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

    x_axis = action_dicts.keys()
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
        index = time_slot_indexs[index]
        index_dicts.setdefault(index, 0)

    for hour_appear in hour_appear_arrays.values:
        hour_appear = time_slot_indexs[hour_appear]
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

time_slot_indexs = {} # 将96个时段还原为24个时段
time_slots = 24
for k in range(time_slots):
    times_of_four = 4 * (k + 1)
    for l in range(4):
        time_slot_indexs.setdefault(times_of_four - l, int(times_of_four / 4))

budget_paras = [0.0625]
campaign_id = data_type['campaign_id']
project_name = 'DRLB'

result_file = data_type['type']

log_path = '../' + project_name + '/result/'

result_directory = log_path + campaign_id + result_file

heuristic_result_path = '../heuristic_algo/result/' + campaign_id + result_file + '/results_train.best.perf.txt'

print('\n##########To Bids.csv files##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    to_bids(data_type['type'], budget_para, campaign_id, result_directory)
    clk_frequency(budget_para, result_directory)

print('\n##########Time slots’ctr statistics##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    ctr_statistics(budget_para, result_directory)

print('\n##########List Metrics##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    list_metrics(budget_para, result_directory)

print('\n##########Action Distribution##########')
for budget_para in budget_paras:
    print('\n------budget_para:{}------'.format(budget_para))
    action_distribution(budget_para, result_directory)

os.remove('price_nums.csv')
print('price_nums.csv file has been removed')

