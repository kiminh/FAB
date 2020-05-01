import pandas as pd
import numpy as np
import datetime
import os
from src.FAB_BN.config import config
from src.data_type import config as data_type
from src.FAB_BN.RL_brain import DDPG, OrnsteinUhlenbeckNoise

def setup_seed(seed):
    np.random.seed(seed)
# 设置随机数种子
setup_seed(1)

# 由启发式算法得到的eCPC
def choose_eCPC(campaign, original_ctr):
    results_train_best = open('../heuristic_algo/result/' + campaign + data_type['type'] + '/results_train.best.perf.txt', 'r')
    train_best_bid = {}
    for i, line in enumerate(results_train_best):
        if i == 0:
            continue
        line_array = line.strip().split('\t')
        train_best_bid.setdefault(int(line_array[0]), int(line_array[-1]))

    if config['budget_para'][0] == 0.5:
        eCPC = train_best_bid[2] / original_ctr
    elif config['budget_para'][0] == 0.25:
        eCPC = train_best_bid[4] / original_ctr
    elif config['budget_para'][0] == 0.125:
        eCPC = train_best_bid[8] / original_ctr
    else:
        eCPC = train_best_bid[16] / original_ctr

    return eCPC

# 奖励函数type1
# def adjust_reward(auc_len, e_true_value, e_miss_true_value, bid_win_t, market_price_win_t, e_win_imp_with_clk_value, e_cost, e_win_imp_without_clk_cost, e_clks, real_clks,
#                   e_lose_imp_with_clk_value,
#                   e_clk_aucs,
#                   e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs, e_no_clk_no_win_aucs, no_win_imps_market_prices_t, budget, total_clks, t):
#     if auc_len > 0:
#
#         # print(e_clks[t], real_clks[t])
#         reward_degree = np.square(1 - np.mean(np.true_divide(np.subtract(bid_win_t, market_price_win_t), bid_win_t)))
#         reward_win_imp_with_clk = (e_clks[t] / real_clks[t]) * reward_degree if real_clks[t] > 0 else 0
#
#         # print(e_clks[t] / real_clks[t])
#
#         # remain_budget = (budget - np.sum(e_cost[:t+1])) / budget
#         # remain_budget = remain_budget if remain_budget > 0 else 1e-1 # 1e-1防止出现除0错误
#         # remain_clks = (total_clks - np.sum(real_clks[:t+1])) / total_clks
#         # punish_win_rate = remain_clks / remain_budget
#         reward_win_imp_without_clk = - e_win_imp_without_clk_cost[t] / e_cost[t] if e_cost[t] > 0 else 0
#
#         temp_rate = (e_clk_no_win_aucs[t] / e_clk_aucs[t]) if e_clk_aucs[t] > 0 else 1
#         punish_no_win_clk_rate = 1 - temp_rate if temp_rate != 1 else 1
#         base_punishment = - (real_clks[t] - e_clks[t]) / real_clks[t] if real_clks[t] > 0 else 0
#         # print(punish_no_win_clk_rate)
#         reward_lose_imp_with_clk = base_punishment / punish_no_win_clk_rate
#
#         base_encourage = e_lose_imp_without_clk_cost[t] / no_win_imps_market_prices_t if no_win_imps_market_prices_t > 0 else 0
#         encourage_rate = e_no_clk_no_win_aucs[t] / e_no_clk_aucs[t] if e_no_clk_aucs[t] > 0 else 0
#         reward_lose_imp_without_clk = base_encourage * encourage_rate if encourage_rate > 0 else 0
#
#         reward_positive = reward_win_imp_with_clk
#         reward_negative = reward_lose_imp_with_clk
#         # print(reward_win_imp_with_clk, reward_lose_imp_without_clk)
#         # print(reward_win_imp_without_clk, reward_lose_imp_with_clk)
#         reward_t = reward_positive + reward_negative
#         # print(reward_positive, reward_negative)
#         # print(t, reward_t)
#     else:
#         reward_t = 0
#
#     n = 1 # 奖励函数type1的缩放因子
#
#     return reward_t / n

def adjust_reward(auc_len, e_no_adjust_bids, e_no_adjust_bids_clks, sum_market_price, market_prices, sum_to_market_prices, real_hour_clks, e_true_value, e_miss_true_value, bid_win_t, market_price_win_t, e_win_imp_with_clk_value, e_cost, e_win_imp_without_clk_cost, e_clks, real_clks,
                  e_lose_imp_with_clk_value,
                  e_clk_aucs,
                  e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs, e_no_clk_no_win_aucs, no_win_imps_market_prices_t, budget, total_clks, t):
    if auc_len > 0:
        rl_clks = np.sum(e_clks[:t+1]) / total_clks
        h_clks = np.sum(e_no_adjust_bids_clks[:t+1]) / total_clks

        rl_cost = np.sum(e_cost[:t + 1]) / budget
        h_cost = np.sum(e_no_adjust_bids[:t + 1]) / budget

        # rl_clks = e_clks[t] / total_clks
        # h_clks = e_no_adjust_bids_clks[t] / total_clks
        #
        # rl_cost = e_cost[t] / budget
        # h_cost = e_no_adjust_bids[t] / budget

        # reward_t = rl_cost / rl_clks - h_cost / h_clks if rl_clks > 0 and h_clks > 0 else -1
        # reward_t = (rl_clks - h_clks) / (rl_cost - h_cost)
        remain_rl = ((budget - np.sum(e_cost[:t + 1])) / budget) / ((total_clks - np.sum(e_clks[:t+1])) / total_clks)
        remain_h = ((budget - np.sum(e_no_adjust_bids[:t + 1])) / budget) / ((total_clks - np.sum(e_no_adjust_bids_clks[:t+1])) / total_clks)

        x = rl_cost / rl_clks if rl_clks > 0 else 0
        y = h_cost / h_clks if h_clks > 0 else 0

        reward_t = (rl_clks - h_clks) * (remain_rl - remain_h)

        # print(remain_rl - remain_h)
    else:
        reward_t = 0

    n = 1e3 # 奖励函数type1的缩放因子

    return reward_t / n

def run_env(budget_para):
    # 训练
    print('data loading')
    test_data = pd.read_csv(data_type['data_path'] + data_type['campaign_id'] + str(data_type['fraction_type'])
                            + '/test_' + data_type['type'] + '.csv', header=None).drop([0])
    test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = test_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    test_data.iloc[:, config['data_pctr_index']] \
        = test_data.iloc[:, config['data_pctr_index']].astype(
        float)
    test_data.iloc[:, config['data_fraction_index']] \
        = test_data.iloc[:, config['data_fraction_index']].astype(
        int)
    test_data = test_data.values

    train_data = pd.read_csv(data_type['data_path'] + data_type['campaign_id'] + str(data_type['fraction_type'])
                            + '/train_' + data_type['type'] + '.csv')
    train_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2] \
        = train_data.iloc[:, config['data_clk_index']:config['data_marketprice_index'] + 2].astype(
        int)
    train_data.iloc[:, config['data_pctr_index']] \
        = train_data.iloc[:, config['data_pctr_index']].astype(
        float)
    train_data.iloc[:, config['data_fraction_index']] \
        = train_data.iloc[:, config['data_fraction_index']].astype(
        int)
    train_data = train_data.values

    # config['train_budget'] = np.sum(train_data[:, config['data_marketprice_index']])
    config['train_budget'] = 32000000
    budget = config['train_budget'] * budget_para

    # config['test_budget'] = np.sum(test_data[:, config['data_marketprice_index']])
    config['test_budget'] = 32000000

    original_ctr = np.sum(train_data[:, config['data_clk_index']]) / len(train_data)
    total_clks = np.sum(train_data[:, config['data_clk_index']])
    real_hour_clks = []
    market_prices = []
    sum_market_price = np.sum(train_data[:, config['data_marketprice_index']])

    for i in range(data_type['fraction_type']):
        real_hour_clks.append(
            np.sum(train_data[train_data[:, config['data_fraction_index']] == i][:, config['data_clk_index']]))
        market_prices.append(train_data[train_data[:, config['data_fraction_index']] == i][:, config['data_marketprice_index']])

    sum_to_market_prices = [0 for i in range(data_type['fraction_type'])]
    for i in range(data_type['fraction_type']):
        if i == 0:
            sum_to_market_prices[i] = np.sum(market_prices[i])
        else:
            sum_to_market_prices[i] = np.sum(market_prices[i]) + sum_to_market_prices[i - 1]

    td_error, action_loss = 0, 0
    eCPC = choose_eCPC(data_type['campaign_id'], original_ctr)

    e_results = []
    e_actions = []
    test_records = []

    is_learn = False

    fraction_type = data_type['fraction_type']
    exploration_rate = config['exploration_rate']
    for episode in range(config['train_episodes']):
        e_clks = [0 for i in range(fraction_type)]  # episode各个时段所获得的点击数，以下类推
        e_profits = [0 for i in range(fraction_type)]
        e_reward = [0 for i in range(fraction_type)]
        e_cost = [0 for i in range(fraction_type)]
        e_no_adjust_bids = [0 for i in range(fraction_type)]
        e_no_adjust_bids_clks = [0 for i in range(fraction_type)]

        e_true_value = [0 for i in range(fraction_type)]
        e_miss_true_value = [0 for i in range(fraction_type)]
        e_win_imp_with_clk_value = [0 for i in range(fraction_type)]
        e_win_imp_without_clk_cost = [0 for i in range(fraction_type)] # 各个时段浪费在没有点击的曝光上的预算
        e_lose_imp_with_clk_value = [0 for i in range(fraction_type)]
        e_clk_aucs = [0 for i in range(fraction_type)]
        e_clk_no_win_aucs = [0 for i in range(fraction_type)]
        e_lose_imp_without_clk_cost = [0 for i in range(fraction_type)]
        e_no_clk_aucs = [0 for i in range(fraction_type)]
        e_no_clk_no_win_aucs = [0 for i in range(fraction_type)]

        e_dones = False

        actions = [0 for i in range(fraction_type)]
        init_action = 0
        next_action = 0

        state_ = np.array([])

        break_time_slot = 0
        real_clks = [0 for i in range(fraction_type)]
        bid_nums = [0 for i in range(fraction_type)]
        imps = [0 for i in range(fraction_type)]

        ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

        # 状态包括：当前CTR，
        for t in range(fraction_type):
            auc_datas = train_data[train_data[:, config['data_fraction_index']] == t]

            if t == 0:
                state = np.array([1, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, ctr_t, win_rate_t
                action = RL.choose_action(state)
                # action = np.clip(action + np.random.normal(0, 1) * 0.1, -0.99, 0.99)
                init_action = action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
                bids = np.where(bids >= 300, 300, bids)
            else:
                state = state_
                action = next_action
                bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
                bids = np.where(bids >= 300, 300, bids)

            actions[t] = action

            origin_bids = auc_datas[auc_datas[:, config['data_pctr_index']] * eCPC >= auc_datas[:, config['data_marketprice_index']]]
            e_no_adjust_bids[t] = np.sum(origin_bids[:, config['data_marketprice_index']])
            e_no_adjust_bids_clks[t] = np.sum(origin_bids[:, config['data_clk_index']])

            win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
            no_win_auctions = auc_datas[bids <= auc_datas[:, config['data_marketprice_index']]]
            e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
            e_profits[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC - win_auctions[:, config['data_marketprice_index']])

            e_true_value[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC)
            e_miss_true_value[t] = np.sum(no_win_auctions[:, config['data_pctr_index']] * eCPC)
            with_clk_win_auctions = win_auctions[win_auctions[:, config['data_clk_index']] == 1]
            e_win_imp_with_clk_value[t] = np.sum(with_clk_win_auctions[:, config['data_pctr_index']] * eCPC)
            e_win_imp_without_clk_cost[t] = np.sum(win_auctions[win_auctions[:, config['data_clk_index']] == 0][:, config['data_marketprice_index']])
            with_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 1]
            e_lose_imp_with_clk_value[t] = np.sum(with_clk_no_win_auctions[:, config['data_pctr_index']] * eCPC)

            e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)
            imps[t] = len(win_auctions)
            real_clks[t] = np.sum(auc_datas[:, config['data_clk_index']], dtype=int)
            bid_nums[t] = len(auc_datas)

            e_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 1])
            e_clk_no_win_aucs[t] = len(with_clk_no_win_auctions)

            e_no_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 0])

            without_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 0]
            e_lose_imp_without_clk_cost[t] = np.sum(without_clk_no_win_auctions[:, config['data_marketprice_index']])
            e_no_clk_no_win_aucs[t] = len(without_clk_no_win_auctions)

            market_prices_t = auc_datas[:, config['data_marketprice_index']]

            bid_win_t = bids[bids >= auc_datas[:, config['data_marketprice_index']]]
            market_price_win_t = market_prices_t[bids >= auc_datas[:, config['data_marketprice_index']]]

            no_win_imps_market_prices_t = np.sum(no_win_auctions[:, config['data_marketprice_index']])
            if np.sum(e_cost) >= budget:
                break_time_slot = t
                temp_cost = 0
                temp_lose_cost = 0
                temp_win_auctions = 0
                e_clks[t] = 0
                e_profits[t] = 0
                e_no_adjust_bids[t] = 0
                e_no_adjust_bids_clks[t] = 0

                e_true_value[t] = 0
                e_miss_true_value[t] = 0

                e_win_imp_without_clk_cost[t] = 0
                e_lose_imp_with_clk_value[t] = 0
                real_clks[t] = 0
                imps[t] = 0
                bid_nums[t] = 0

                e_win_imp_with_clk_value[t] = 0
                e_clk_aucs[t] = 0
                e_lose_imp_without_clk_cost[t] = 0
                e_no_clk_aucs[t] = 0
                e_no_clk_no_win_aucs[t] = 0

                bids_t = []
                market_prices_t = []

                bid_win_t = []
                market_price_win_t = []
                no_win_imps_market_prices_t = 0
                for i in range(len(auc_datas)):
                    if temp_cost >= (budget - np.sum(e_cost[:t])):
                        e_dones = True
                        break
                    current_data = auc_datas[i, :]

                    temp_clk = int(current_data[config['data_clk_index']])
                    temp_market_price = current_data[config['data_marketprice_index']]
                    if t == 0:
                        temp_action = init_action
                    else:
                        temp_action = next_action
                    bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                    bid = bid if bid <= 300 else 300
                    real_clks[t] += temp_clk
                    bid_nums[t] += 1

                    if temp_clk == 1:
                        e_clk_aucs[t] += temp_clk
                    else:
                        e_no_clk_aucs[t] += 1
                    bids_t.append(bid)
                    market_prices_t.append(temp_market_price)

                    origin_bid = current_data[config['data_pctr_index']] * eCPC
                    if origin_bid >= temp_market_price:
                        e_no_adjust_bids[t] += temp_market_price
                        e_no_adjust_bids_clks[t] += temp_clk

                    if bid >= temp_market_price:
                        if temp_clk == 0:
                            e_win_imp_without_clk_cost[t] += temp_market_price
                        else:
                            e_win_imp_with_clk_value[t] += current_data[config['data_pctr_index']] * eCPC
                        bid_win_t.append(bid)
                        market_price_win_t.append(temp_market_price)

                        e_profits[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                        e_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                        e_clks[t] += temp_clk
                        imps[t] += 1
                        temp_cost += temp_market_price
                        temp_win_auctions += 1
                    else:
                        e_miss_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                        temp_lose_cost += temp_market_price
                        no_win_imps_market_prices_t += temp_market_price
                        if temp_clk == 1:
                            e_clk_no_win_aucs[t] += 1
                            e_lose_imp_with_clk_value[t] += current_data[config['data_pctr_index']] * eCPC
                        else:
                            e_no_clk_no_win_aucs[t] += 1
                e_cost[t] = temp_cost
                ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
                win_rate_t = temp_win_auctions / bid_nums[t]
            else:
                ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(
                    win_auctions) > 0 else 0
                win_rate_t = len(win_auctions) / len(auc_datas) if len(auc_datas) > 0 else 0
            budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
            budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
            time_left_ratio = (fraction_type - 1 - t)/ fraction_type
            avg_time_spend = budget_left_ratio / time_left_ratio if time_left_ratio > 0 else 0
            cost_t_ratio = e_cost[t] / budget
            if t == 0:
                state_ = np.array([avg_time_spend, cost_t_ratio, ctr_t, win_rate_t])
            else:
                state_ = np.array(
                    [avg_time_spend, cost_t_ratio, ctr_t, win_rate_t])
            action_ = RL.choose_action(state_)
            # print(action_, exploration_rate, np.random.normal(action_, exploration_rate))
            # action_ = np.clip(action_ + np.random.normal(0, 1.0) * 0.1, -0.99, 0.99)
            next_action = action_

            reward_t = adjust_reward(len(auc_datas), e_no_adjust_bids, e_no_adjust_bids_clks, sum_market_price, market_prices, sum_to_market_prices, real_hour_clks, e_true_value, e_miss_true_value, bid_win_t, market_price_win_t, e_win_imp_with_clk_value, e_cost, e_win_imp_without_clk_cost, e_clks, real_clks,
                  e_lose_imp_with_clk_value,
                  e_clk_aucs,
                  e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs, e_no_clk_no_win_aucs, no_win_imps_market_prices_t, budget, total_clks, t)
            reward = reward_t
            e_reward[t] = reward

            if t == fraction_type - 1:
                e_dones = True

            transition = np.hstack((state.tolist(), action, reward, state_.tolist(), e_dones))

            RL.store_transition(transition)

            if np.sum(e_cost) >= budget:
                break

            if RL.memory_counter >= RL.batch_size:
                td_e, a_loss = RL.learn()
                td_error, action_loss = td_e, a_loss

        if (episode > 0) and ((episode + 1) % config['observation_episode'] == 0):
            e_result = [np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
                        int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                        np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error,
                        action_loss]
            e_results.append(e_result)

            # 在原始论文中，每感知一次环境就要对模型进行一次训练
            # 然而频繁地学习在未充分感知环境的情况下，会使模型陷入局部（当前）最优
            # 因此可以每感知N次再对模型训练n次，这样会使得模型更稳定，并加快学习速度
            # is_learn = True
            # exploration_rate *= 0.999
            # if is_learn:  # after observing config['observation_size'] times, for config['learn_iter'] learning time
            #     for m in range(config['learn_iter']):
            #         td_e, a_loss = RL.learn()
            #         td_error, action_loss = td_e, a_loss
            #         if m == config['learn_iter'] - 1:
            #             is_learn = False

            actions_df = pd.DataFrame(data=actions)
            actions_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/train_actions_' + str(budget_para) + '.csv')

            hour_clks = {'clks': e_clks, 'no_bid_clks': np.subtract(real_hour_clks, e_clks).tolist(),
                         'real_clks': real_hour_clks}
            hour_clks_df = pd.DataFrame(data=hour_clks)
            hour_clks_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/train_hour_clks_' + str(budget_para) + '.csv')
            print('episode {}, reward={}, profits={}, budget={}, cost={}, clks={}, real_clks={}, bids={}, '
                  'imps={}, cpm={}, break_time_slot={}, td_error={}, action_loss={}\n'.format(
                    episode + 1, np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
                    int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
                    np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, td_error, action_loss))
            test_result, test_actions, test_hour_clks = test_env(config['test_budget'] * budget_para, budget_para, test_data, eCPC)
            test_records.append(test_result)
            e_actions.append(test_actions)

    e_results_df = pd.DataFrame(data=e_results, columns=['reward', 'profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                                         'break_time_slot', 'td_error', 'action_loss'])
    e_results_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/train_episode_results_' + str(budget_para) + '.csv')

    e_actions_df = pd.DataFrame(data=e_actions)
    e_actions_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/test_episode_actions_' + str(budget_para) + '.csv')

    test_records_df = pd.DataFrame(data=test_records,
                                   columns=['rewards', 'profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                            'break_time_slot'])
    test_records_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/test_episode_results_' + str(budget_para) + '.csv')


def test_env(budget, budget_para, test_data, eCPC):
    results = []

    total_clks = np.sum(test_data[:, config['data_clk_index']])

    real_hour_clks = []
    market_prices = []
    sum_market_price = np.sum(test_data[:, config['data_marketprice_index']])
    for i in range(data_type['fraction_type']):
        real_hour_clks.append(
            np.sum(test_data[test_data[:, config['data_fraction_index']] == i][:, config['data_clk_index']]))
        market_prices.append(
            test_data[test_data[:, config['data_fraction_index']] == i][:, config['data_marketprice_index']])

    sum_to_market_prices = [0 for i in range(data_type['fraction_type'])]
    for i in range(data_type['fraction_type']):
        if i == 0:
            sum_to_market_prices[i] = np.sum(market_prices[i])
        else:
            sum_to_market_prices[i] = np.sum(market_prices[i]) + sum_to_market_prices[i - 1]

    fraction_type = data_type['fraction_type']
    exploration_rate = config['exploration_rate']

    e_clks = [0 for i in range(fraction_type)]  # episode各个时段所获得的点击数，以下类推
    e_profits = [0 for i in range(fraction_type)]
    e_reward = [0 for i in range(fraction_type)]
    e_cost = [0 for i in range(fraction_type)]
    e_no_adjust_bids = [0 for i in range(fraction_type)]
    e_no_adjust_bids_clks = [0 for i in range(fraction_type)]

    e_true_value = [0 for i in range(fraction_type)]
    e_miss_true_value = [0 for i in range(fraction_type)]
    e_win_imp_with_clk_value = [0 for i in range(fraction_type)]
    e_win_imp_without_clk_cost = [0 for i in range(fraction_type)]  # 各个时段浪费在没有点击的曝光上的预算
    e_lose_imp_with_clk_value = [0 for i in range(fraction_type)]
    e_clk_aucs = [0 for i in range(fraction_type)]
    e_clk_no_win_aucs = [0 for i in range(fraction_type)]
    e_lose_imp_without_clk_cost = [0 for i in range(fraction_type)]
    e_no_clk_aucs = [0 for i in range(fraction_type)]
    e_no_clk_no_win_aucs = [0 for i in range(fraction_type)]

    e_dones = False

    actions = [0 for i in range(fraction_type)]
    init_action = 0
    next_action = 0

    state_ = np.array([])

    break_time_slot = 0
    real_clks = [0 for i in range(fraction_type)]
    bid_nums = [0 for i in range(fraction_type)]
    imps = [0 for i in range(fraction_type)]


    # 状态包括：当前CTR，
    for t in range(fraction_type):
        auc_datas = test_data[test_data[:, config['data_fraction_index']] == t]

        if t == 0:
            state = np.array([1, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, ctr_t, win_rate_t
            action = RL.choose_best_action(state)
            # action = np.clip(action + np.random.normal(0, 1) * 0.1, -0.99, 0.99)
            init_action = action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + init_action)
            bids = np.where(bids >= 300, 300, bids)
        else:
            state = state_
            action = next_action
            bids = auc_datas[:, config['data_pctr_index']] * eCPC / (1 + action)
            bids = np.where(bids >= 300, 300, bids)

        actions[t] = action

        origin_bids = auc_datas[auc_datas[:, config['data_pctr_index']] * eCPC >= auc_datas[:, config['data_marketprice_index']]]
        # print(bids, origin_bids)
        e_no_adjust_bids[t] = np.sum(origin_bids[:, config['data_marketprice_index']])
        e_no_adjust_bids_clks[t] = np.sum(origin_bids[:, config['data_clk_index']])

        win_auctions = auc_datas[bids >= auc_datas[:, config['data_marketprice_index']]]
        no_win_auctions = auc_datas[bids <= auc_datas[:, config['data_marketprice_index']]]
        e_cost[t] = np.sum(win_auctions[:, config['data_marketprice_index']])
        e_profits[t] = np.sum(
            win_auctions[:, config['data_pctr_index']] * eCPC - win_auctions[:, config['data_marketprice_index']])

        e_true_value[t] = np.sum(win_auctions[:, config['data_pctr_index']] * eCPC)
        e_miss_true_value[t] = np.sum(no_win_auctions[:, config['data_pctr_index']] * eCPC)
        with_clk_win_auctions = win_auctions[win_auctions[:, config['data_clk_index']] == 1]
        e_win_imp_with_clk_value[t] = np.sum(with_clk_win_auctions[:, config['data_pctr_index']] * eCPC)
        e_win_imp_without_clk_cost[t] = np.sum(
            win_auctions[win_auctions[:, config['data_clk_index']] == 0][:, config['data_marketprice_index']])
        with_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 1]
        e_lose_imp_with_clk_value[t] = np.sum(with_clk_no_win_auctions[:, config['data_pctr_index']] * eCPC)

        e_clks[t] = np.sum(win_auctions[:, config['data_clk_index']], dtype=int)
        imps[t] = len(win_auctions)
        real_clks[t] = np.sum(auc_datas[:, config['data_clk_index']], dtype=int)
        bid_nums[t] = len(auc_datas)

        e_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 1])
        e_clk_no_win_aucs[t] = len(with_clk_no_win_auctions)

        e_no_clk_aucs[t] = len(auc_datas[auc_datas[:, config['data_clk_index']] == 0])

        without_clk_no_win_auctions = no_win_auctions[no_win_auctions[:, config['data_clk_index']] == 0]
        e_lose_imp_without_clk_cost[t] = np.sum(without_clk_no_win_auctions[:, config['data_marketprice_index']])
        e_no_clk_no_win_aucs[t] = len(without_clk_no_win_auctions)

        market_prices_t = auc_datas[:, config['data_marketprice_index']]

        bid_win_t = bids[bids >= auc_datas[:, config['data_marketprice_index']]]
        market_price_win_t = market_prices_t[bids >= auc_datas[:, config['data_marketprice_index']]]

        no_win_imps_market_prices_t = np.sum(no_win_auctions[:, config['data_marketprice_index']])
        if np.sum(e_cost) >= budget:
            break_time_slot = t
            temp_cost = 0
            temp_lose_cost = 0
            temp_win_auctions = 0
            e_clks[t] = 0
            e_profits[t] = 0
            e_no_adjust_bids[t] = 0
            e_no_adjust_bids_clks[t] = 0

            e_true_value[t] = 0
            e_miss_true_value[t] = 0

            e_win_imp_without_clk_cost[t] = 0
            e_lose_imp_with_clk_value[t] = 0
            real_clks[t] = 0
            imps[t] = 0
            bid_nums[t] = 0

            e_win_imp_with_clk_value[t] = 0
            e_clk_aucs[t] = 0
            e_lose_imp_without_clk_cost[t] = 0
            e_no_clk_aucs[t] = 0
            e_no_clk_no_win_aucs[t] = 0

            bids_t = []
            market_prices_t = []

            bid_win_t = []
            market_price_win_t = []
            no_win_imps_market_prices_t = 0
            for i in range(len(auc_datas)):
                if temp_cost >= (budget - np.sum(e_cost[:t])):
                    e_dones = True
                    break
                current_data = auc_datas[i, :]

                temp_clk = int(current_data[config['data_clk_index']])
                temp_market_price = current_data[config['data_marketprice_index']]
                if t == 0:
                    temp_action = init_action
                else:
                    temp_action = next_action
                bid = current_data[config['data_pctr_index']] * eCPC / (1 + temp_action)
                bid = bid if bid <= 300 else 300
                real_clks[t] += temp_clk
                bid_nums[t] += 1

                if temp_clk == 1:
                    e_clk_aucs[t] += temp_clk
                else:
                    e_no_clk_aucs[t] += 1
                bids_t.append(bid)
                market_prices_t.append(temp_market_price)

                origin_bid = current_data[config['data_pctr_index']] * eCPC
                if origin_bid >= temp_market_price:
                    e_no_adjust_bids[t] += temp_market_price
                    e_no_adjust_bids_clks[t] += temp_clk

                if bid >= temp_market_price:
                    if temp_clk == 0:
                        e_win_imp_without_clk_cost[t] += temp_market_price
                    else:
                        e_win_imp_with_clk_value[t] += current_data[config['data_pctr_index']] * eCPC
                    bid_win_t.append(bid)
                    market_price_win_t.append(temp_market_price)

                    e_profits[t] += (current_data[config['data_pctr_index']] * eCPC - temp_market_price)
                    e_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                    e_clks[t] += temp_clk
                    imps[t] += 1
                    temp_cost += temp_market_price
                    temp_win_auctions += 1
                else:
                    e_miss_true_value[t] += current_data[config['data_pctr_index']] * eCPC
                    temp_lose_cost += temp_market_price
                    no_win_imps_market_prices_t += temp_market_price
                    if temp_clk == 1:
                        e_clk_no_win_aucs[t] += 1
                        e_lose_imp_with_clk_value[t] += current_data[config['data_pctr_index']] * eCPC
                    else:
                        e_no_clk_no_win_aucs[t] += 1
            e_cost[t] = temp_cost
            ctr_t = e_clks[t] / temp_win_auctions if temp_win_auctions > 0 else 0
            win_rate_t = temp_win_auctions / bid_nums[t]
        else:
            ctr_t = np.sum(win_auctions[:, config['data_clk_index']]) / len(win_auctions) if len(
                win_auctions) > 0 else 0
            win_rate_t = len(win_auctions) / len(auc_datas) if len(auc_datas) > 0 else 0
        budget_left_ratio = (budget - np.sum(e_cost[:t + 1])) / budget
        budget_left_ratio = budget_left_ratio if budget_left_ratio >= 0 else 0
        time_left_ratio = (fraction_type - 1 - t) / fraction_type
        avg_time_spend = budget_left_ratio / time_left_ratio if time_left_ratio > 0 else 0
        cost_t_ratio = e_cost[t] / budget
        if t == 0:
            state_ = np.array([avg_time_spend, cost_t_ratio, ctr_t, win_rate_t])
        else:
            state_ = np.array(
                [avg_time_spend, cost_t_ratio, ctr_t, win_rate_t])
        action_ = RL.choose_best_action(state_)
        # print(action_, exploration_rate, np.random.normal(action_, exploration_rate))
        # action_ = np.clip(action_ + np.random.normal(0, 1.0) * 0.1, -0.99, 0.99)
        next_action = action_

        reward_t = adjust_reward(len(auc_datas), e_no_adjust_bids, e_no_adjust_bids_clks, sum_market_price, market_prices, sum_to_market_prices, real_hour_clks, e_true_value,
                                 e_miss_true_value, bid_win_t, market_price_win_t, e_win_imp_with_clk_value, e_cost,
                                 e_win_imp_without_clk_cost, e_clks, real_clks,
                                 e_lose_imp_with_clk_value,
                                 e_clk_aucs,
                                 e_clk_no_win_aucs, e_lose_imp_without_clk_cost, e_no_clk_aucs,
                                 e_no_clk_no_win_aucs, no_win_imps_market_prices_t, budget, total_clks, t)
        reward = reward_t
        e_reward[t] = reward

        if t == fraction_type - 1:
            e_dones = True

        if np.sum(e_cost) >= budget:
            break
    print('-----------测试结果-----------\n')
    result = [np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)), int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
              np.sum(e_cost) / np.sum(imps), break_time_slot]
    hour_clks = {'clks': e_clks, 'no_bid_clks': np.subtract(real_hour_clks, e_clks).tolist(),
                 'real_clks': real_hour_clks}

    results.append(result)
    result_df = pd.DataFrame(data=results,
                             columns=['rewards', 'profits', 'budget', 'cost', 'clks', 'real_clks', 'bids', 'imps', 'cpm',
                                      'break_time_slot'])
    result_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/test_result_' + str(budget_para) + '.csv')

    test_actions_df = pd.DataFrame(data=actions)
    test_actions_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/test_action_' + str(budget_para) + '.csv')

    test_hour_clks_df = pd.DataFrame(data=hour_clks)
    test_hour_clks_df.to_csv(log_path + '/result_reward_1/' + str(fraction_type) + '/test_hour_clks_' + str(budget_para) + '.csv')

    print('rewards={}, profits={}, budget={}, cost={}, clks={}, real_clks={}, bids={}, imps={}, cpm={}, break_time_slot={}, {}\n'.format(
        np.sum(e_reward), np.sum(e_profits), budget, np.sum(e_cost), int(np.sum(e_clks)),
        int(np.sum(real_clks)), np.sum(bid_nums), np.sum(imps),
        np.sum(e_cost) / np.sum(imps) if np.sum(imps) > 0 else 0, break_time_slot, datetime.datetime.now()))

    return result, actions, hour_clks


if __name__ == '__main__':
    log_path = data_type['campaign_id'] + data_type['type']
    if not os.path.exists(log_path + '/result_reward_1/' + str(data_type['fraction_type'])):
        os.mkdir(log_path + '/result_reward_1/' + str(data_type['fraction_type']))

    RL = DDPG(
        feature_nums=config['feature_num'],
        action_nums=1,
        lr_A=config['learning_rate_a'],
        lr_C=config['learning_rate_c'],
        reward_decay=config['reward_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        tau=config['tau'],  # for target network soft update
        device=config['device'],
    )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        run_env(budget_para[i])