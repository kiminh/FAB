import random
import pandas as pd
import numpy as np

random.seed(999)

'''
  对原始数据进行负采样，以对比FAB是否对环境具有动态适应性
'''
campaign_id = '3386'

train_data = pd.read_csv(campaign_id + '/train_data.csv')
test_data = pd.read_csv(campaign_id + '/test_data.csv')
train_data.iloc[:, 1] = train_data.iloc[:, 1].astype(int)
test_data.iloc[:, 1] = test_data.iloc[:, 1].astype(int)

# 负采样后达到的点击率
CLICK_RATE = 0.001  # 1:1000

train_clks = int(np.sum(train_data.iloc[:, 1]))
train_auc_nums = len(train_data)
# '+config['train_date']+'一天
def getSampleRate():
    click = train_clks  #'+config['train_date']+' 1天
    total = train_auc_nums  # '+config['train_date']+' 1天
    rate = click / (CLICK_RATE * (total - click))
    # 原始数据中的点击和曝光总数
    print('clicks: {0} impressions: {1}\n'.format(click, total))
    # 一个负例被选中的概率，每多少个负例被选中一次
    # print('sample rate: {0} sample num: {1}'.format(rate, 1 / rate))
    print('sample_rate is:',rate)
    return rate

# 获取训练样本
sample_rate = getSampleRate()

with open( campaign_id + '/train_sample.csv', 'w') as fo:
    fi = open(campaign_id + '/train_data.csv')
    p = 0 # 原始正样本
    n = 0 # 原始负样本
    nn = 0 # 剩余的负样本
    c = 0 # 总数
    for t, line in enumerate(fi, start=1):
        if t == 1:
            fo.write(line)
        else:
            c += 1
            label = line.split(',')[1] # 是否点击标签
            if int(label) == 0:
                n += 1
                if random.randint(0, train_auc_nums) <= train_auc_nums * sample_rate:  # down sample, 选择对应数据量的负样本
                    fo.write(line)
                    nn += 1
            else:
                p += 1
                fo.write(line)

        if t % 1000000 == 0:
            print(t)
    fi.close()
# print(c, n, p+nn, p, nn, (p+nn)/c, nn / n, p / nn)
print('训练数据负采样完成')

train_sample = pd.read_csv(campaign_id + '/train_sample.csv')
train_sample.iloc[:, 2] = train_sample.iloc[:, 2].astype(int)
print(np.sum(train_sample.iloc[:, 2]))

test_clks = int(np.sum(test_data.iloc[:, 1]))
test_auc_nums = len(test_data)

def getTestSampleRate():
    click = test_clks
    total = test_auc_nums
    rate = click / (CLICK_RATE * (total - click))
    # 原始数据中的点击和曝光总数
    print('clicks: {0} impressions: {1}\n'.format(click, total))
    print('sample_rate is:',rate)
    return rate

# 获取训练样本
test_sample_rate = getTestSampleRate()

# 获取测试样本,20130609一天
with open( campaign_id + '/test_sample.csv', 'w') as fo:
    fi = open(campaign_id + '/test_data.csv')
    p = 0 # 原始正样本
    n = 0 # 原始负样本
    nn = 0 # 剩余的负样本
    c = 0 # 总数
    for t, line in enumerate(fi, start=1):
        if t==1:
            fo.write(line)
        else:
            c += 1
            label = line.split(',')[1] # 是否点击标签
            if int(label) == 0:
                n += 1
                if random.randint(0, test_auc_nums) <= test_auc_nums * test_sample_rate:  # down sample, 选择对应数据量的负样本
                    fo.write(line)
                    nn += 1
            else:
                p += 1
                fo.write(line)

        if t % 10000 == 0:
            print(t)
    fi.close()
print('测试数据负采样完成')

test_sample = pd.read_csv(campaign_id + '/test_sample.csv')
test_sample.iloc[:, 2] = test_sample.iloc[:, 2].astype(int)
print(np.sum(test_sample.iloc[:, 2]))

