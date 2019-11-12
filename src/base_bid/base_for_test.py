import sys
import random
import math
import pandas as pd
import numpy as np
from src.base_bid.fit_c import fit_c
from src.data_type import config as data_type
import os
random.seed(10)

def bidding_lin(pctr, base_ctr, avg_market_price, base_bid):
    return int(pctr * avg_market_price / base_ctr)

def bidding_opt(c, pCTR, lamda=5.5e-6):  # 出价策略函数
    bid_price = math.sqrt(c*pCTR/lamda + c**2) -c
    return bid_price

def win_auction(case, bid):
    return bid >= case[1] # bid > winning price

# budgetProportion clk cnv bid imp budget spend para
def simulate_one_bidding_strategy_with_parameter(bidding_opt_c, cases, ctrs, tcost, proportion, algo, para):
    budget = int(tcost / proportion) # intialise the budget
    cpc = 30000 # cost per click

    cost = 0
    clks = 0
    bids = 0
    imps = 0
    profits = 0

    real_imps = 0
    real_clks = 0

    no_win_hour_clks = [0 for i in range(24)]
    win_hour_clks = [0 for i in range(24)]
    hour_clks = [0 for i in range(24)]

    for idx in range(0, len(cases)):
        pctr = ctrs[idx]
        if algo == "lin":
            bid = bidding_lin(pctr, original_pctr, avg_market_price, para)
        elif algo == "bidding_opt":
            bid = bidding_opt(bidding_opt_c, pctr)
        else:
            print('wrong bidding strategy name')
            sys.exit(-1)
        bids += 1
        case = cases[idx]
        real_imps += 1
        real_clks += case[0]
        hour_clks[case[2]] += case[0]
        if win_auction(case, bid):
            imps += 1
            clks += case[0]
            win_hour_clks[case[2]] += case[0]
            cost += case[1]
            profits += (cpc*pctr - case[1])
        else:
            no_win_hour_clks[case[2]] += case[0]

        if cost > budget:
            print('早停时刻', case[2])
            break
    cpm = (cost / imps) if imps > 0 else 0
    return str(proportion) + '\t' + str(profits) + '\t' + str(clks) + '\t' + str(real_clks) + '\t' + str(bids) + '\t' + \
        str(imps) + '\t' + str(real_imps) + '\t' + str(budget) + '\t' + str(cost) + '\t' + str(cpm) + '\t'+ algo + '\t' + str(para)\
        , no_win_hour_clks, win_hour_clks, hour_clks

def simulate_one_bidding_strategy(bidding_opt_c, cases, ctrs, tcost, proportion, algo, writer, campaign):
    paras = algo_paras[algo]
    for para in paras:
        res, no_win_hour_clks, win_hour_clks, hour_clks = simulate_one_bidding_strategy_with_parameter(bidding_opt_c, cases, ctrs, tcost, proportion, algo, para)
        print(res)
        hour_clks_data = {'win_hour_clks' : win_hour_clks, 'no_win_hour_clks' : no_win_hour_clks, 'hour_clks' : hour_clks}
        hour_clks_data_df = pd.DataFrame(data=hour_clks_data)
        hour_clks_data_df.to_csv('result/' + campaign + '/test_hour_clks_' + str(proportion) + '.csv')
        writer.write(res + '\n')

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')

    # 从训练数据中读取到初始ecpc和初始ctr
    train_data = pd.read_csv(data_type['data_path'] + data_type['campaign_id'] + '/train_' + data_type['type'] + '.csv', header=None).drop(0, axis=0)
    train_data.iloc[:, 1: 4] \
            = train_data.iloc[:, 1 : 4].astype(
            int)
    train_data.iloc[:, 4] \
        = train_data.iloc[:, 4].astype(
        float)
    imp_num = len(train_data.values)
    original_ctr = np.sum(train_data.values[:, 1]) / imp_num
    original_ecpc = np.sum(train_data.values[:, 2]) / np.sum(train_data.values[:, 1])

    avg_market_price = np.sum(train_data.values[:, 2]) / imp_num
    original_pctr = np.sum(train_data.values[:, 4]) / imp_num

    bidding_opt_c = fit_c(train_data)

    clicks_prices = [] # clk and price
    total_cost = 0 # total original cost during the test data
    # 从测试数据中读取测试数据
    test_data = pd.read_csv(data_type['data_path'] + data_type['campaign_id'] + '/test_' + data_type['type'] + '.csv', header=None).drop(0, axis=0)
    test_data.iloc[:, 1: 4] \
            = test_data.iloc[:, 1 : 4].astype(
            int)
    test_data.iloc[:, 4] \
        = test_data.iloc[:, 4].astype(
        float)
    data = test_data.values
    for i in range(len(data)):
        click = int(data[i][1])
        winning_price = int(data[i][2])
        clicks_prices.append((click, winning_price, int(data[i][3])))

    total_cost += test_data.iloc[:, 2].sum()

    print('总预算{}'.format(total_cost))
    pctrs = test_data.values[:, 4].flatten().tolist()

    # parameters setting for each bidding strategy
    budget_proportions = [2, 4, 8, 16]

    fo = open('result/' + data_type['campaign_id'] + 'results_test.txt', 'w') # rtb.results.txt
    header = "prop\tprofits\tclks\treal_clks\tbids\timps\treal_imps\tbudget\tspend\tcpm\talgo\tpara"
    fo.write(header + '\n')
    print(header)
    for k, proportion in enumerate(budget_proportions):
        algo_paras = {"lin": [0], "bidding_opt": [0]}
        for algo in algo_paras:
            simulate_one_bidding_strategy(bidding_opt_c, clicks_prices, pctrs, total_cost, proportion, algo, fo, data_type['campaign_id'])

