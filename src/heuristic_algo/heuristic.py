import sys
import random
import math
import pandas as pd
import numpy as np
import os
from src.data_type import config as data_type
random.seed(10)

def bidding_lin(pctr, base_ctr, base_bid): # 启发式算法
    return int(pctr * base_bid / base_ctr)

def win_auction(case, bid):
    return bid >= case[1] # bid > winning price

# budgetProportion clk cnv bid imp budget spend para
def simulate_one_bidding_strategy_with_parameter(original_ctr, cases, ctrs, tcost, proportion, algo, para):
    budget = int(tcost / proportion) # intialise the budget
    cpc = 30000 # cost per click

    cost = 0
    clks = 0
    bids = 0
    imps = 0
    profits = 0

    real_imps = 0
    real_clks = 0

    for idx in range(0, len(cases)):
        pctr = ctrs[idx]
        if algo == "lin":
            bid = bidding_lin(pctr, original_ctr, para)
        else:
            print('wrong bidding strategy name')
            sys.exit(-1)
        bids += 1
        case = cases[idx]
        real_imps += 1
        real_clks += case[0]
        if win_auction(case, bid):
            imps += 1
            clks += case[0]
            cost += case[1]
            profits += (cpc*pctr - case[1])
        if cost > budget:
            print('早停时刻', case[2])
            break
    cpm = (cost / imps) if imps > 0 else 0
    return str(proportion) + '\t' + str(profits) + '\t' + str(clks) + '\t' + str(real_clks) + '\t' + str(bids) + '\t' + \
        str(imps) + '\t' + str(real_imps) + '\t' + str(budget) + '\t' + str(cost) + '\t' + str(cpm) + '\t'+ algo + '\t' + str(para)

def simulate_one_bidding_strategy(original_ctr, cases, ctrs, tcost, proportion, algo, algo_paras, writer):
    paras = algo_paras[algo]
    for para in paras:
        res = simulate_one_bidding_strategy_with_parameter(original_ctr, cases, ctrs, tcost, proportion, algo, para)
        print(res)
        writer.write(res + '\n')

def to_train_results(campaign):
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

    clicks_prices = [] # clk and price
    total_cost = 0 # total original cost during the train data
    data = train_data.values
    for i in range(len(data)):
        click = int(data[i][1])
        winning_price = int(data[i][2])
        clicks_prices.append((click, winning_price, int(data[i][3])))
    total_cost += train_data.iloc[:, 2].sum()

    print('总预算{}'.format(total_cost))

    pctrs = train_data.values[:, 4].flatten().tolist()

    # parameters setting for each bidding strategy
    budget_proportions = [2, 4, 8, 16]
    lin_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))

    algo_paras = {"lin": lin_paras}

    fo = open('result/' + campaign + '/results_train.txt', 'w') # rtb.results.txt
    header = "prop\tprofits\tclks\treal_clks\tbids\timps\treal_imps\tbudget\tspend\tcpm\talgo\tpara"
    fo.write(header + '\n')
    print(header)
    for proportion in budget_proportions:
        for algo in algo_paras:
            simulate_one_bidding_strategy(original_ctr, clicks_prices, pctrs, total_cost, proportion, algo, algo_paras, fo)

if not os.path.exists('result'):
    os.mkdir('result')

if __name__ == '__main__':
    campaign = data_type['campaign_id']
    
    to_train_results(campaign)  # 生成训练结果

    fi = open('result/' + campaign + '/results_train.txt', 'r') # rtb.result.1458.txt
    fo = open('result/' + campaign + '/results_train.txt'.replace('.txt', '.best.perf.txt'), 'w')
    first = True

    setting_row = {}
    setting_perf = {}

    for line in fi:
        line = line.strip()
        s = line.split('\t')
        if first:
            first = False
            fo.write(line + '\n')
            continue
        algo = s[10]
        prop = s[0]
        perf = float(s[2]) # 选择点击2排序，利润1
        setting = (prop, algo)
        if setting in setting_perf and perf > setting_perf[setting] or setting not in setting_perf:
            setting_perf[setting] = perf
            setting_row[setting] = line
    fi.close()

    best_bid = []
    for setting in sorted(setting_perf):
        best_bid.append([int(setting_row[setting].split('\t')[0]), int(setting_row[setting].split('\t')[-1])])
        fo.write(setting_row[setting] + '\n')
    fo.close()

