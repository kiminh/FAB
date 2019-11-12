import pandas as pd
import numpy as np

# 生成FAB使用的原始数据

campaign_id = '1458'
train_log = pd.read_csv(campaign_id + '/11_log.csv')
test_log = pd.read_csv(campaign_id + '/12_log.csv')

train_ctr = pd.read_csv(campaign_id + '/11_test_submission.csv')
test_ctr = pd.read_csv(campaign_id + '/12_test_submission.csv')

train_log_values = train_log.values
train_data = {'ctr': train_ctr.values[:, 1] * 1000, 'clk': train_log_values[:, 0],
              'market_price': train_log_values[:, 23], 'hour': train_log_values[:, 2], 'pctr': train_ctr.values[:, 1]}
test_log_values = test_log.values
test_data = {'ctr': test_ctr.values[:, 1] * 1000, 'clk': test_log_values[:, 0],
              'market_price': test_log_values[:, 23], 'hour': test_log_values[:, 2], 'pctr': test_ctr.values[:, 1]}

train_data_df = pd.DataFrame(data=train_data)
train_data_df.to_csv('train_data.csv', index=None)
test_data_df = pd.DataFrame(data=test_data)
test_data_df.to_csv('test_data.csv', index=None)

print(len(train_data_df), np.sum(train_data_df.iloc[:, 1]), np.sum(train_data_df.iloc[:, 2]))
print(len(test_data_df), np.sum(test_data_df.iloc[:, 1]), np.sum(test_data_df.iloc[:, 2]))


