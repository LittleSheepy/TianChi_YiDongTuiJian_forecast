import pandas as pd
import numpy as np
import datetime
import sys
import time
import xgboost as xgb
from add_feture import *
FEATURE_EXTRACTION_SLOT = 10
LabelDay = datetime.datetime(2014,12,17,0,0,0)
Data = pd.read_csv("../DataSet/drop1112_sub_item.csv")
Data['daystime'] = Data['days'].map(lambda x: time.strptime(x, "%Y-%m-%d")).map(lambda x: datetime.datetime(*x[:6]))
szTime = ""

def get_train(train_user,end_time):
    # 取出label day 前一天的记录作为打标记录
    data_train = train_user[(train_user['daystime'] == (end_time-datetime.timedelta(days=1)))]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))

    filePath = "../mid/data_train/"
    # 训练样本中，删除重复的样本
    data_train.to_csv(filePath + szTime + "data_train1.csv", index = False)
    data_train = data_train.drop_duplicates(['user_id', 'item_id'])
    data_train.to_csv(filePath + szTime + "data_train2.csv", index = False)
    data_train_ui = data_train['user_id'] / data_train['item_id']
    data_train_ui.to_csv(filePath + szTime + "data_train_ui.csv", index = False)

    # print(len(data_train))

    # 使用label day 的实际购买情况进行打标
    data_label = train_user[train_user['daystime'] == end_time]
    data_label.to_csv(filePath + szTime + "data_label.csv", index = False)
    data_label_buy = data_label[data_label['behavior_type'] == 4]
    print("data_label_buy ",len(data_label_buy))
    data_label_buy.to_csv(filePath + szTime + "data_label_buy.csv", index = False)
    data_label_buy_ui = data_label_buy['user_id'] / data_label_buy['item_id']
    data_label_buy_ui.to_csv(filePath + szTime + "data_label_buy_ui.csv", index = False)

    # 对前一天的交互记录进行打标
    data_train_labeled = data_train_ui.isin(data_label_buy_ui)
    dict = {True: 1, False: 0}
    data_train_labeled = data_train_labeled.map(dict)
    data_train_labeled.to_csv(filePath + szTime + "data_train_labeled.csv", index = False)

    data_train['label'] = data_train_labeled
    return data_train[['user_id', 'item_id','item_category', 'label']]

def get_label_testset(train_user,LabelDay):
    # 测试集选为上一天所有的交互数据
    data_test = train_user[(train_user['daystime'] == LabelDay)]#&((train_user.behavior_type==3)|(train_user.behavior_type==2))
    data_test = data_test.drop_duplicates(['user_id', 'item_id'])
    return data_test[['user_id', 'item_id','item_category']]



def item_category_feture(data,end_time,beforeoneday):
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_category,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_category,beforefiveday.behavior_type)
    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_category,beforethreeday.behavior_type)
        
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_category,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

    buyRate_2 = pd.DataFrame()
    buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
    buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
    buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
    buyRate_2.index = item_count_before5.index

    buyRate_3 = pd.DataFrame()
    buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
    buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
    buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
    buyRate_3.index = item_count_before_3.index


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_category_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,countAverage,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,buyRate,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before5,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before_3,how='left',right_index=True,left_index=True)
    item_category_feture = pd.merge(item_category_feture,item_count_before_2,how='left',right_index=True,left_index=True)
#    item_category_feture = pd.merge(item_category_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_category_feture = pd.merge(item_category_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_category_feture.fillna(0,inplace=True)
    item_category_feture = item_category_feture.reset_index()
    return item_category_feture

def item_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    item_count = pd.crosstab(data.item_id,data.behavior_type)
    item_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        item_count_before5 = pd.crosstab(beforefiveday.item_id,beforefiveday.behavior_type)

    item_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        item_count_before_3 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)

    item_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        item_count_before_2 = pd.crosstab(beforethreeday.item_id,beforethreeday.behavior_type)
        
    item_count_unq = data.groupby(by = ['item_id','behavior_type']).agg({"user_id":lambda x:x.nunique()});item_count_unq = item_count_unq.unstack()
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayitem_count = pd.crosstab(beforeoneday.item_id,beforeoneday.behavior_type)
    countAverage = item_count/FEATURE_EXTRACTION_SLOT
    buyRate = pd.DataFrame()
    buyRate['click'] = item_count[1]/item_count[4]
    buyRate['skim'] = item_count[2]/item_count[4]
    buyRate['collect'] = item_count[3]/item_count[4]
    buyRate.index = item_count.index

    buyRate_2 = pd.DataFrame()
    buyRate_2['click'] = item_count_before5[1]/item_count_before5[4]
    buyRate_2['skim'] = item_count_before5[2]/item_count_before5[4]
    buyRate_2['collect'] = item_count_before5[3]/item_count_before5[4]
    buyRate_2.index = item_count_before5.index

    buyRate_3 = pd.DataFrame()
    buyRate_3['click'] = item_count_before_3[1]/item_count_before_3[4]
    buyRate_3['skim'] = item_count_before_3[2]/item_count_before_3[4]
    buyRate_3['collect'] = item_count_before_3[3]/item_count_before_3[4]
    buyRate_3.index = item_count_before_3.index

    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    item_id_feture = pd.merge(item_count,beforeonedayitem_count,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,countAverage,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,buyRate,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_unq,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before5,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before_3,how='left',right_index=True,left_index=True)
    item_id_feture = pd.merge(item_id_feture,item_count_before_2,how='left',right_index=True,left_index=True)
#    item_id_feture = pd.merge(item_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
#    item_id_feture = pd.merge(item_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    item_id_feture.fillna(0,inplace=True)
    item_id_feture = item_id_feture.reset_index()
    return item_id_feture


def user_id_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_count = pd.crosstab(data.user_id,data.behavior_type)
    user_count_before5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id, beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_count_before5 = pd.crosstab(beforefiveday.user_id,beforefiveday.behavior_type)

    user_count_before_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_count_before_3 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)

    user_count_before_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_count_before_2 = pd.crosstab(beforethreeday.user_id,beforethreeday.behavior_type)

    filePath = "../mid/user_id_feture/"
    # 训练样本中，删除重复的样本
    user_count_before5.to_csv(filePath + szTime + "user_count_before5.csv", index = False)
    user_count.to_csv(filePath + szTime + "user_count.csv", index = False)

    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    beforeonedayuser_count = pd.crosstab(beforeoneday.user_id,beforeoneday.behavior_type)
    beforeonedayuser_count.to_csv(filePath + szTime + "beforeonedayuser_count.csv", index = False)

    countAverage = user_count/FEATURE_EXTRACTION_SLOT
    countAverage.columns = ["1_" + "typeMean","2_" + "typeMean","3_" + "typeMean","4_" + "typeMean"]

    countAverage.to_csv(filePath + szTime + "countAverage.csv", index = False)
    buyRate = pd.DataFrame()
    buyRate['clickAll'] = user_count[1]/user_count[4]
    buyRate['skimAll'] = user_count[2]/user_count[4]
    buyRate['collectAll'] = user_count[3]/user_count[4]
    buyRate.index = user_count.index
    buyRate.to_csv(filePath + szTime + "buyRate.csv", index = False)

    buyRate_2 = pd.DataFrame()
    buyRate_2['click5d'] = user_count_before5[1]/user_count_before5[4]
    buyRate_2['skim5d'] = user_count_before5[2]/user_count_before5[4]
    buyRate_2['collect5d'] = user_count_before5[3]/user_count_before5[4]
    buyRate_2.index = user_count_before5.index
    buyRate_2.to_csv(filePath + szTime + "buyRate_2.csv", index = False)

    buyRate_3 = pd.DataFrame()
    buyRate_3['click3d'] = user_count_before_3[1]/user_count_before_3[4]
    buyRate_3['skim3d'] = user_count_before_3[2]/user_count_before_3[4]
    buyRate_3['collect3d'] = user_count_before_3[3]/user_count_before_3[4]
    buyRate_3.index = user_count_before_3.index
    buyRate_3.to_csv(filePath + szTime + "buyRate_3.csv", index = False)

    buyRate_4 = pd.DataFrame()
    buyRate_4['click2d'] = user_count_before_2[1]/user_count_before_2[4]
    buyRate_4['skim2d'] = user_count_before_2[2]/user_count_before_2[4]
    buyRate_4['collect2d'] = user_count_before_2[3]/user_count_before_2[4]
    buyRate_4.index = user_count_before_2.index
    buyRate_4.to_csv(filePath + szTime + "buyRate_4.csv", index = False)


    buyRate = buyRate.replace([np.inf, -np.inf], 0)
    buyRate_2 = buyRate_2.replace([np.inf, -np.inf], 0)
    buyRate_3 = buyRate_3.replace([np.inf, -np.inf], 0)
    buyRate_4 = buyRate_4.replace([np.inf, -np.inf], 0)

    long_online = pd.pivot_table(beforeoneday,index=['user_id'],values=['hours'],aggfunc=[np.min,np.max,np.ptp])
    long_online.columns = ["minOnline","maxOnline","ptpOnline"]
    user_count.columns = ["1_" + "typeAll", "2_" + "typeAll", "3_" + "typeAll","4_" + "typeAll"]
    user_count_before5.columns = ["1_" + "typeN5","2_" + "typeN5","3_" + "typeN5","4_" + "typeN5"]
    user_count_before_3.columns = ["1_" + "typeN3","2_" + "typeN3","3_" + "typeN3","4_" + "typeN3"]
    user_count_before_2.columns = ["1_" + "typeN2","2_" + "typeN2","3_" + "typeN2","4_" + "typeN2"]
    beforeonedayuser_count.columns = ["1_" + "typeN1","2_" + "typeN1","3_" + "typeN1","4_" + "typeN1"]

    user_id_feture = pd.merge(user_count,beforeonedayuser_count,how='left',right_index=True,left_index=True)
    user_id_feture.to_csv(filePath + szTime + "user_id_feture_user_count.csv", index = False)

    user_id_feture = pd.merge(user_id_feture,countAverage,how='left',right_index=True,left_index=True)
    user_id_feture.to_csv(filePath + szTime + "user_id_feture_countAverage.csv", index = False)

    user_id_feture = pd.merge(user_id_feture,user_count_before5,suffixes=("","_typeN5"),how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,user_count_before_3,suffixes=("","_typeN3"),how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,user_count_before_2,suffixes=("","_typeN2"),how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,long_online,how='left',right_index=True,left_index=True)
    long_online.to_csv(filePath + szTime + "long_online.csv", index = False)
    user_id_feture = pd.merge(user_id_feture,buyRate,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,buyRate_2,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,buyRate_3,how='left',right_index=True,left_index=True)
    user_id_feture = pd.merge(user_id_feture,buyRate_4,how='left',right_index=True,left_index=True)
    user_id_feture.fillna(0,inplace=True)
    user_id_feture = user_id_feture.reset_index()
    return user_id_feture



def user_item_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_id],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    user_item_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5+2)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=end_time-datetime.timedelta(days=5)]
        user_item_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_id],beforefiveday.behavior_type)
    user_item_count_3=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3+2)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=3)]
        user_item_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)

    user_item_count_2=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7+2)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=end_time-datetime.timedelta(days=7)]
        user_item_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_id],beforethreeday.behavior_type)
        
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_id],beforeoneday.behavior_type)
    
    #    _live = user_item_long_touch(data)
    
    
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_id'],values=['behavior_type'],aggfunc=np.max)
    user_item_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_item_feture = pd.merge(user_item_feture,_live,how='left',right_index=True,left_index=True)

    user_item_feture = pd.merge(user_item_feture,user_item_count_5,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,user_item_count_3,how='left',right_index=True,left_index=True)
    user_item_feture = pd.merge(user_item_feture,user_item_count_2,how='left',right_index=True,left_index=True)
    user_item_feture.fillna(0,inplace=True)
    user_item_feture = user_item_feture.reset_index()
    return user_item_feture

def user_cate_feture(data,end_time,beforeoneday):   
    # data = Data[(Data['daystime']<LabelDay) & (Data['daystime']>LabelDay-datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))]
    user_item_count = pd.crosstab([data.user_id,data.item_category],data.behavior_type)
    # beforeoneday = Data[Data['daystime'] == LabelDay-datetime.timedelta(days=1)]
    
    user_cate_count_5=None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5+2))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    else:
        beforefiveday = data[data['daystime']>=(end_time-datetime.timedelta(days=5))]
        user_cate_count_5 = pd.crosstab([beforefiveday.user_id,beforefiveday.item_category],beforefiveday.behavior_type)
    user_cate_count_3 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3+2))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=3))]
        user_cate_count_3 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)


    user_cate_count_2 = None
    if (((end_time-datetime.timedelta(days=5))<datetime.datetime(2014,12,13,0,0,0))&((end_time-datetime.timedelta(days=5))>datetime.datetime(2014,12,10,0,0,0))):
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7+2))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
    else:
        beforethreeday = data[data['daystime']>=(end_time-datetime.timedelta(days=7))]
        user_cate_count_2 = pd.crosstab([beforethreeday.user_id,beforethreeday.item_category],beforethreeday.behavior_type)
        
#    _live = user_cate_long_touch(data)
    beforeonedayuser_item_count = pd.crosstab([beforeoneday.user_id,beforeoneday.item_category],beforeoneday.behavior_type)
    max_touchtime = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['hours'],aggfunc=[np.min,np.max])
    max_touchtype = pd.pivot_table(beforeoneday,index=['user_id','item_category'],values=['behavior_type'],aggfunc=np.max)
    user_cate_feture = pd.merge(user_item_count,beforeonedayuser_item_count,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtime,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,max_touchtype,how='left',right_index=True,left_index=True)
#    user_cate_feture = pd.merge(user_cate_feture,_live,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_5,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_3,how='left',right_index=True,left_index=True)
    user_cate_feture = pd.merge(user_cate_feture,user_cate_count_2,how='left',right_index=True,left_index=True)
    user_cate_feture.fillna(0,inplace=True)
    user_cate_feture = user_cate_feture.reset_index()
    return user_cate_feture

def GetTrainFeaturesEx(trainDays, endDay):
    result = []
    LabelDay = endDay
    for i in range(trainDays):
        train_user_window1 = None
        if (LabelDay >= datetime.datetime(2014, 12, 12, 0, 0, 0)):
            print(i, "LabelDay = ", LabelDay, LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT + 2))
            train_user_window1 = Data[
                (Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT + 2))) & (
                            Data['daystime'] < LabelDay)]
        else:
            print(i, "LabelDay = ", LabelDay, LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))
            train_user_window1 = Data[
                (Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (
                            Data['daystime'] < LabelDay)]
        #        train_user_window1 = Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT))) & (Data['daystime'] < LabelDay)]
        beforeoneday = Data[Data['daystime'] == (LabelDay - datetime.timedelta(days=1))]
        beforeoneday.to_csv(filePath + szTime + 'beforeoneday.csv', index=None)
        train_user_window1.to_csv(filePath + szTime + 'train_user_window1.csv', index=None)
        beforetwoday = Data[
            (Data['daystime'] >= (LabelDay - datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
        beforefiveday = Data[
            (Data['daystime'] >= (LabelDay - datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
        x = get_train(Data, LabelDay)
        add_user_click_1 = user_click(beforeoneday)
        add_user_item_click_1 = user_item_click(beforeoneday)
        add_user_cate_click_1 = user_cate_click(beforeoneday)

        add_user_click_1.to_csv(filePath + szTime + 'add_user_click_1.csv', index=None)
        add_user_item_click_1.to_csv(filePath + szTime + 'add_user_item_click_1.csv', index=None)
        add_user_cate_click_1.to_csv(filePath + szTime + 'add_user_cate_click_1.csv', index=None)

        add_user_click_2 = user_click(beforetwoday)
        add_user_click_5 = user_click(beforefiveday)
        liveday = user_liveday(train_user_window1)
        # sys.exit()
        a = user_id_feture(train_user_window1, LabelDay, beforeoneday)

        b = item_id_feture(train_user_window1, LabelDay, beforeoneday)

        c = item_category_feture(train_user_window1, LabelDay, beforeoneday)

        d = user_cate_feture(train_user_window1, LabelDay, beforeoneday)

        e = user_item_feture(train_user_window1, LabelDay, beforeoneday)

        x.to_csv(filePath + szTime + 'x.csv', index=None)
        x = pd.merge(x, a, on=['user_id'], how='left')
        a.to_csv(filePath + szTime + 'a.csv', index=None)
        x.to_csv(filePath + szTime + 'x_a.csv', index=None)
        x = pd.merge(x, b, on=['item_id'], how='left')
        x = pd.merge(x, c, on=['item_category'], how='left')
        x = pd.merge(x, d, on=['user_id', 'item_category'], how='left')
        x = pd.merge(x, e, on=['user_id', 'item_id'], how='left')
        x = pd.merge(x, add_user_click_1, left_on=['user_id'], right_index=True, how='left')
        x = pd.merge(x, add_user_click_2, left_on=['user_id'], right_index=True, how='left')
        x = pd.merge(x, add_user_click_5, left_on=['user_id'], right_index=True, how='left')
        x = pd.merge(x, add_user_item_click_1, left_on=['user_id', 'item_id'], right_index=True, how='left')
        x = pd.merge(x, add_user_cate_click_1, left_on=['user_id', 'item_category'], right_index=True, how='left')
        x = pd.merge(x, liveday, left_on=['user_id'], right_index=True, how='left')
        x = x.fillna(0)
        print(i, LabelDay, len(x))
        LabelDay = LabelDay - datetime.timedelta(days=1)
        if (LabelDay == datetime.datetime(2014, 12, 13, 0, 0, 0)):
            LabelDay = datetime.datetime(2014, 12, 10, 0, 0, 0)
        result.append(x)

    train_set = pd.concat(result, axis=0, ignore_index=True)
    return train_set

if __name__ == '__main__':
#    pass
    filePath = "../mid/main/"
    szTime = str(LabelDay.month) + str(LabelDay.day)

    TrainEndDay = datetime.datetime(2014,12,17,0,0,0)
    train_set = GetTrainFeaturesEx(2,TrainEndDay)
    train_set.to_csv(filePath + szTime + 'train_set.csv', index=None)
    #train_set.to_csv('train_train_no_jiagou.csv',index=None)
    ###############################################
    
    LabelDay=datetime.datetime(2014,12,17,0,0,0)
    test = get_label_testset(Data,LabelDay)

    train_user_window1 =  Data[(Data['daystime'] > (LabelDay - datetime.timedelta(days=FEATURE_EXTRACTION_SLOT-1))) & (Data['daystime'] <= LabelDay)]
    beforeoneday = Data[Data['daystime'] == LabelDay]
    beforetwoday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=2))) & (Data['daystime'] < LabelDay)]
    beforefiveday = Data[(Data['daystime'] >= (LabelDay-datetime.timedelta(days=5))) & (Data['daystime'] < LabelDay)]
    add_user_click = user_click(beforeoneday)
    add_user_item_click = user_item_click(beforeoneday)
    add_user_cate_click = user_cate_click(beforeoneday)
    add_user_click_2 = user_click(beforetwoday)
    add_user_click_5 = user_click(beforefiveday)
    liveday = user_liveday(train_user_window1)
    a = user_id_feture(train_user_window1, LabelDay,beforeoneday)

    b = item_id_feture(train_user_window1, LabelDay,beforeoneday)

    c = item_category_feture(train_user_window1, LabelDay,beforeoneday)

    d = user_cate_feture(train_user_window1, LabelDay,beforeoneday)

    e = user_item_feture(train_user_window1, LabelDay,beforeoneday)

    test = pd.merge(test,a,on=['user_id'],how='left')
    test = pd.merge(test,b,on=['item_id'],how='left')
    test = pd.merge(test,c,on=['item_category'],how='left')
    test = pd.merge(test,d,on=['user_id','item_category'],how='left')
    test = pd.merge(test,e,on=['user_id','item_id'],how='left')
    test = pd.merge(test,add_user_click,left_on = ['user_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_click_2,left_on = ['user_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_click_5,left_on = ['user_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_item_click,left_on = ['user_id','item_id'],right_index=True,how = 'left' )
    test = pd.merge(test,add_user_cate_click,left_on = ['user_id','item_category'],right_index=True,how = 'left' )
    test = pd.merge(test,liveday,left_on = ['user_id'],right_index=True,how = 'left' )
    test = test.fillna(0)
    #test.to_csv('../result/test_test_no_jiagou.csv',index=None)
    test.to_csv(filePath + szTime + 'test_set.csv', index=None)
#
#    sys.exit()

    ###############采样
    train_set_1 = train_set[train_set['label']==1]
    train_set_0 = train_set[train_set['label']==0]
    new_train_set_0 = train_set_0.sample(len(train_set_1)*90)
    train_set = pd.concat([train_set_1,new_train_set_0],axis=0)
    ###############
    train_y = train_set['label'].values
    train_x = train_set.drop(['user_id', 'item_id','item_category', 'label'], axis=1).values
    test_x = test.drop(['user_id', 'item_id','item_category'], axis=1).values   
    num_round = 900
    params = {
        'silent': 1,        #当这个参数值为1时，静默模式开启，不会输出任何信息。
        'max_depth': 5,     #树的最大深度3-10。
        'colsample_bytree': 0.8,        #用来控制每棵随机采样的列数的占比(每一列是一个特征)。
        'subsample': 0.8,               #这个参数控制对于每棵树，随机采样的比例。
        'eta': 0.02,
        'objective': 'binary:logistic',
        'eval_metric ':'error',
        'min_child_weight': 1,        #决定最小叶子节点样本权重和
        'max_delta_step':0,           #
        'gamma':0,
        'scale_pos_weight':1,
        'seed': 10}  #
    plst = list(params.items())
    #train_x.to_csv("../result/train_x.csv", index=False)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    bst = xgb.train(plst, dtrain, num_round)
    predicted_proba = bst.predict(dtest)
    print("predicted_proba" , predicted_proba)

    predicted_proba = pd.DataFrame(predicted_proba)
    predicted_proba.to_csv(filePath + szTime + 'predicted_proba.csv', index=None)

    predicted = pd.concat([test[['user_id', 'item_id']], predicted_proba], axis=1)
    predicted.columns = ['user_id','item_id','prob']
    #print(predicted)
    predicted = predicted.sort_values('prob',  axis=0,ascending=False)
    #print(predicted)
#    predict1 = predicted.iloc[:650, [0, 1]]
#    # 保存到文件
#    predict1.to_csv("../result/10_30_2/650_1B80minchildweight1.8.csv", index=False)
    
    predict2 = predicted.iloc[:800, [0, 1]]
    # 保存到文件
    predicted.to_csv("../result/predict.csv", index=False)
    predict2.to_csv("../result/result.csv", index=False)
    
#    predict3 = predicted.iloc[:750, [0, 1]]
#    # 保存到文件
#    predict3.to_csv("../result/10_30_2/750_1B80minchildweight1.8.csv", index=False)
    #sys.exit()
#    evaluate(predicted)




    #####################################################################线下验证部分
    predicted = predict2
    reference = Data[Data['daystime'] == (LabelDay+datetime.timedelta(days=1))]
    reference = reference[reference['behavior_type'] == 4]  # 购买的记录
    reference = reference[['user_id', 'item_id']]  # 获取ui对
    reference = reference.drop_duplicates(['user_id', 'item_id'])  # 去重
    ui = predicted['user_id'] / predicted['item_id']

    predicted=predicted[ui.duplicated() == False]

    predicted_ui = predicted['user_id'] / predicted['item_id']
    reference_ui = reference['user_id'] / reference['item_id']

    is_in = predicted_ui.isin(reference_ui)
    true_positive = predicted[is_in]

    tp = len(true_positive)
    predictedSetCount = len(predicted)
    referenceSetCount = len(reference)

    precision = tp / predictedSetCount
    recall = tp / referenceSetCount

    f_score = 2 * precision * recall / (precision + recall)

    tp = recall * referenceSetCount
    predictedSetCount = tp / precision

    print('%.8f%%   %.8f   %.8f   %.0f   %.0f' %
          (f_score * 100, precision, recall, tp, predictedSetCount))