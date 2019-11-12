import pandas as pd
import numpy as np

campaign_id = '3386'

# train_data
train_data = pd.read_csv(campaign_id + '/11_log.csv', header=None).drop([0])
train_ctrs = pd.read_csv(campaign_id + '/11_test_submission.csv', header=None).drop([0])
train_data.iloc[:, [4]] = train_data.iloc[:, [4]].astype(str) # 类型强制转换

clk_arrays = train_data.iloc[:, 0].values
pay_price_arrays = train_data.iloc[:, 23].values
ctr_arrays = train_ctrs.iloc[:, 1].values

'''
    15,30,45,60
    115,130,145,160
    ....
    2315,2330,2345,2360
'''
time_faction = [15,30,45,60]
time_factions = []
for i in range(24):
    temp_time_fraction = np.add(100 * i, time_faction)
    for i in range(4):
        time_factions.append(temp_time_fraction[i])

origin_time_arrays = train_data.iloc[:, 4].values
now_time_arrays = []
for k, time_item in enumerate(origin_time_arrays):
    minute_item = int(origin_time_arrays[k][8: 12])
    now_time_arrays.append(minute_item)
now_time_np_array = np.array(now_time_arrays)
train_data.iloc[:, 4] = now_time_np_array
origin_time_arrays = train_data.iloc[:, 4].values

for i, fraction_item in enumerate(time_factions):
    up_time = fraction_item
    down_time = fraction_item - 15
    for k, time_item in enumerate(origin_time_arrays):
        if time_item <= up_time and time_item >= down_time:
            origin_time_arrays[k] = i+1

train_to_data = {'clk': clk_arrays, 'pCTR': ctr_arrays, 'pay_price': pay_price_arrays, 'time_fraction': origin_time_arrays}
train_to_data_df = pd.DataFrame(data=train_to_data)
train_to_data_df.to_csv('../src/DRLB/data/' + campaign_id + '/train_DRLB.csv', index=None)

# test_data
test_data = pd.read_csv(campaign_id + '/12_log.csv', header=None).drop([0])
test_ctrs = pd.read_csv(campaign_id + '/12_test_submission.csv', header=None).drop([0])
test_data.iloc[:, [4]] = test_data.iloc[:, [4]].astype(str) # 类型强制转换

clk_arrays = test_data.iloc[:, 0].values
pay_price_arrays = test_data.iloc[:, 23].values
ctr_arrays = test_ctrs.iloc[:, 1].values

'''
    15,30,45,60
    115,130,145,160
    ....
    2315,2330,2345,2360
'''
time_faction = [15,30,45,60]
time_factions = []
for i in range(24):
    temp_time_fraction = np.add(100 * i, time_faction)
    for i in range(4):
        time_factions.append(temp_time_fraction[i])

origin_time_arrays = test_data.iloc[:, 4].values
now_time_arrays = []
for k, time_item in enumerate(origin_time_arrays):
    minute_item = int(origin_time_arrays[k][8: 12])
    now_time_arrays.append(minute_item)
now_time_np_array = np.array(now_time_arrays)
test_data.iloc[:, 4] = now_time_np_array
origin_time_arrays = test_data.iloc[:, 4].values

for i, fraction_item in enumerate(time_factions):
    up_time = fraction_item
    down_time = fraction_item - 15
    for k, time_item in enumerate(origin_time_arrays):
        if time_item <= up_time and time_item >= down_time:
            origin_time_arrays[k] = i+1

test_to_data = {'clk': clk_arrays, 'pCTR': ctr_arrays, 'pay_price': pay_price_arrays, 'time_fraction': origin_time_arrays}
test_to_data_df = pd.DataFrame(data=test_to_data)
test_to_data_df.to_csv('../src/DRLB/data/' + campaign_id + '/test_DRLB.csv', index=None)