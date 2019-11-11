import pandas as pd

campaign_id ='3386'
type = 'sample'

origin_train_data = pd.read_csv(campaign_id + '/train_' + type + '.csv').values
origin_test_data = pd.read_csv(campaign_id + '/test_' + type + '.csv').values


rlb_train_data = {'clk': origin_train_data[:, 1].astype(int),'market_price': origin_train_data[:, 2].astype(int), 'pctr': origin_train_data[:, 4].astype(float), 'hour': origin_train_data[:, 3].astype(int)}
rlb_test_data = {'clk': origin_test_data[:, 1].astype(int), 'market_price': origin_test_data[:, 2].astype(int), 'pctr': origin_test_data[:, 4].astype(float), 'hour': origin_test_data[:, 3].astype(int)}

rlb_train_data_df = pd.DataFrame(data=rlb_train_data)
rlb_train_data_df.to_csv('../src/RLB/data/ipinyou-data/' + campaign_id + '/train.theta.txt', index=None, header=None)
rlb_test_data_df = pd.DataFrame(data=rlb_test_data)
rlb_test_data_df.to_csv('../src/RLB/data/ipinyou-data/' + campaign_id + '/test.theta.txt', index=None, header=None)