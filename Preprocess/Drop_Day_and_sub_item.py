import pandas as pd
import numpy as np
import datetime
import sys
import time



if __name__ == '__main__':
	user_table = pd.read_csv('../DataSet/tianchi_fresh_comp_train_user.csv')
	item_table = pd.read_csv('../DataSet/tianchi_fresh_comp_train_item.csv')
	user_table = user_table[user_table.item_id.isin(list(item_table.item_id))]
	user_table['days'] = user_table['time'].map(lambda x:x.split(' ')[0])
	user_table['hours'] = user_table['time'].map(lambda x:x.split(' ')[1])

	user_buy = user_table[(user_table['days'] == '2014-12-12') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1212.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-11') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1211.csv', index=None)

	user_table = user_table[user_table['days'] != '2014-12-12']
	user_table = user_table[user_table['days'] != '2014-12-11']
	user_table.to_csv('../DataSet/drop1112_sub_item.csv',index=None)

	user_buy = user_table[(user_table['days'] == '2014-12-18') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1218.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-17') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1217.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-16') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1216.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-15') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1215.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-14') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1214.csv', index=None)
	user_buy = user_table[(user_table['days'] == '2014-12-13') & (user_table['behavior_type'] == 4)]
	user_buy.to_csv('../DataSet/user_buy1213.csv', index=None)


