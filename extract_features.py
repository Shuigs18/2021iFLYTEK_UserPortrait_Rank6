import pandas as pd
import numpy as np
from scipy import stats

from gensim.models import Word2Vec
from tensorflow.keras.preprocessing import text, sequence
import tensorflow as tf
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from pandarallel import pandarallel

import time
import gc
import os
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

pandarallel.initialize()

SEED = 2021

# 原始特征预处理
def preprocess(data):
    # 空缺值填充
    data['gender'] = data['gender'].fillna(-1).astype(int)
    data['age'] = data['age'].fillna(-1).astype(int)
    data[["province", "city","model", "make"]] = data[["province", "city","model", "make"]].fillna(str(-1))
    
    # province 按照区域进行划分
    ## 省份区域划分
    area_temp = {
        '东北': [u'黑龙江',u'吉林',u'辽宁'],
        '华东': [u'上海' ,u'江苏',u'浙江',u'安徽' ,u'江西',u'山东',u'福建'],
        '华北': [u'北京' ,u'天津',u'山西',u'河北' ,u'内蒙古'],
        '华中': [u'河南',u'湖南',u'湖北'],
        '华南': [u'广东',u'广西',u'海南'],
        '西南': [u'四川',u'贵州',u'云南' ,u'重庆',u'西藏'],
        '西北': [u'陕西',u'甘肃',u'青海' ,u'宁夏',u'新疆'],
        '港澳台': [u'香港',u'澳门',u'台湾'],
        '缺失值': ['-1']
    }
    area_dict = dict()
    for k, v in area_temp.items():
        for province in v:
            area_dict[province] = k
    data['province_area'] = data['province'].map(lambda x: area_dict[x] if x in area_dict else None)
    
    # city
    ## 按照1234线城市划分
    city_level_temp = {
        0 : [u'北京',u'上海',u'广州' ,u'深圳'],
        1 : [u'成都',	u'重庆',	u'杭州',	u'武汉',	u'西安',	u'郑州',	u'青岛',
                    u'长沙',	u'天津',	u'苏州',	u'南京',	u'东莞',	u'沈阳',	u'合肥',	u'佛山'],
        2 : [u'宁波',	u'昆明',	u'福州',	u'无锡',	u'厦门',	u'哈尔滨',	u'长春',
                u'南昌',	u'济南',	u'大连',	u'贵阳',	u'温州',	u'石家庄',	u'泉州',
                u'南宁',	u'金华',	u'常州',	u'珠海',	u'惠州',	u'嘉兴',	u'南通',
                u'中山',	u'保定',	u'兰州',	u'台州',	u'徐州',	u'太原',	u'绍兴',
                u'烟台',	u'廊坊'],
        3: [u'海口',	u'汕头',	u'潍坊',	u'扬州',	u'洛阳',	u'乌鲁木齐',
                u'临沂',	u'唐山',	u'镇江',	u'盐城',	u'湖州',	u'赣州',	u'漳州',
                u'揭阳',	u'江门',	u'桂林',	u'邯郸',	u'泰州',	u'济宁',	u'呼和浩特',
                u'咸阳',	u'芜湖',	u'三亚',	u'阜阳',	u'淮安',	u'遵义',	u'银川',
                u'衡阳',	u'上饶',	u'柳州',	u'淄博',	u'莆田',	u'绵阳',	u'湛江',
                u'商丘',	u'宜昌',	u'沧州',	u'连云港',	u'南阳',	u'蚌埠',	u'驻马店',
                u'滁州',	u'邢台',	u'潮州',	u'秦皇岛',	u'肇庆',	u'荆州',	u'周口',
                u'马鞍山',	u'清远',	u'宿州',	u'威海',	u'九江',	u'新乡',	u'信阳',
                u'襄阳',	u'岳阳',	u'安庆',	u'菏泽',	u'宜春',	u'黄冈',	u'泰安',
                u'宿迁',	u'株洲',	u'宁德',	u'鞍山',	u'南充',	u'六安',	u'大庆',	u'舟山'],
        4: [u'常德',	u'渭南',	u'孝感',	u'丽水',	u'运城',	u'德州',	u'张家口',
                u'鄂尔多斯',	u'阳江',	u'泸州',	u'丹东',	u'曲靖',	u'乐山',	u'许昌',
                u'湘潭',	u'晋中',	u'安阳',	u'齐齐哈尔',	u'北海',	u'宝鸡',	u'抚州',
                u'景德镇',	u'延安',	u'三明',	u'抚顺',	u'亳州',	u'日照',	u'西宁',
                u'衢州',	u'拉萨',	u'淮北',	u'焦作',	u'平顶山',	u'滨州',	u'吉安',
                u'濮阳',	u'眉山',	u'池州',	u'荆门',	u'铜仁',	u'长治',	u'衡水',
                u'铜陵',	u'承德',	u'达州',	u'邵阳',	u'德阳',	u'龙岩',	u'南平',
                u'淮南',	u'黄石',	u'营口',	u'东营',	u'吉林',	u'韶关',	u'枣庄',
                u'包头',	u'怀化',	u'宣城',	u'临汾',	u'聊城',	u'梅州',	u'盘锦',
                u'锦州',	u'榆林',	u'玉林',	u'十堰',	u'汕尾',	u'咸宁',	u'宜宾',
                u'永州',	u'益阳',	u'黔南州',	u'黔东南',	u'恩施',	u'红河',	u'大理',
                u'大同',	u'鄂州',	u'忻州',	u'吕梁',	u'黄山',	u'开封',	u'郴州',
                u'茂名',	u'漯河',	u'葫芦岛',	u'河源',	u'娄底',	u'延边'],
        5: ["-1"]
    }
    city_level_dict = dict()
    for k, v in city_level_temp.items():
        for province in v:
            city_level_dict[province] = k		
    data['city_level'] = data['city'].apply(lambda x: city_level_dict[x] if x in city_level_dict else 6)

    # 根据训练集中city数量分类
    # 北京>20000 成都>8000 石家庄>8000  >5000  >1000  >100
    city_dict = data.city.value_counts().to_dict()
    def city_encode(x):
        if x == u'-1':
            return 0
        elif x == u'北京':
            return 1
        elif x == u'重庆':
            return 2
        elif x == u'成都':
            return 3
        elif x == u'石家庄':
            return 4
        elif city_dict[x] > 5000:
            return 5
        elif city_dict[x] > 1000:
            return 6
        elif city_dict[x] > 100:
            return 7
        else:
            return 8
    data['city_cluster'] =data['city'].apply(lambda x: city_encode(x))

    # 处理model make
    ## 调整model
    data['model'] = data['model'].str.lower()
    ## 调整make型号
    data['make'] = data['make'].str.lower()
    data['make'] = data['make'].str.replace(' ', '')
    for model in data['model'].unique():
        data['make'] = data['make'].map(lambda x: x.replace(model, ''))
    data['model_make'] = data['model'].astype(str).values + '_' + data['make'].astype(str).values
    model_encode_dict = {
        "oppo": 0,
        "vivo": 1,
        "华为": 2,
        "荣耀": 3,
        "小米": 4,
        "三星": 5,
        "魅族": 6,
        "苹果": 7,
        "-1": 8
    }
    data['model_cluster'] =data['model'].apply(lambda x: model_encode_dict[x] if x in model_encode_dict else 9)

    # 根据训练集中model_make数量分类
    ## >20000  >10000  >5000  >1000  >100
    model_make_dict = data.model_make.value_counts().to_dict()
    def model_make_encode(x):
        if x == u'-1':
            return 0
        elif x == u'oppo_a5':
            return 1
        elif model_make_dict[x] > 10000:
            return 2
        elif model_make_dict[x] > 5000:
            return 3
        elif model_make_dict[x] > 1000:
            return 4
        elif model_make_dict[x] > 100:
            return 5
        else:
            return 6
    data['make_cluster'] =data['model_make'].apply(lambda x: model_make_encode(x))

    # tagid
    data['tagid'] = data['tagid'].parallel_apply(lambda x:eval(x))
    data['tagid'] = data['tagid'].parallel_apply(lambda x:[str(i) for i in x])

    # time
    data['timestamp'] = data.timestamp.parallel_apply(lambda x:eval(x))
    # ms转为s
    data['timestamp'] = data.timestamp.parallel_apply(lambda x:[float(i)/1000 for i in x])

    return data

def timeTargetEncoder(data, columns):
    time_path = "./feats/time_feats"
    if not os.path.exists(time_path):
        os.mkdir(time_path)
    # 处理tag标签
    train = data[data.label.notnull().T]
    test = data[data.label.isnull().T]
    train[columns] = train[columns].astype(str).str.replace('[', '').str.replace(']', '')
    test[columns] = test[columns].astype(str).str.replace('[', '').str.replace(']', '')

    del test['label']
    temp_train = train.set_index(["userid"])[columns].astype(str).str.split(",", expand=True).stack().reset_index(drop=True,level=-1).reset_index().rename(columns={0: columns})
    temp_train[f'{columns}_count'] = temp_train.groupby(columns)[columns].transform("count")
    temp_train = temp_train[temp_train[f'{columns}_count']>=500]

    temp_test = test.set_index(["userid"])[columns].str.split(",", expand=True).stack().reset_index(drop=True,level=-1).reset_index().rename(columns={0: columns})    
    del train[columns]
    del test[columns]
    gc.collect()

    train = train.merge(temp_train, on="userid", how="left")
    train = train.dropna(subset=[columns])

    test = test.merge(temp_test, on="userid", how="left")
    train[f'{columns}_buy_count'] = train.groupby([columns])['label'].transform("sum")
    train[f'{columns}_label'] = train.groupby([columns])['label'].transform("mean")
    trn_tmp = train.groupby(['userid', columns])[f'{columns}_count',f'{columns}_buy_count',f'{columns}_label'].mean().reset_index()
    tst_tmp = test[['userid', columns]].merge(trn_tmp[[columns, f'{columns}_count',f'{columns}_buy_count',f'{columns}_label']].drop_duplicates(subset=[columns]),
                                        on=columns, how='left')
    tagid_label_mean = trn_tmp[f'{columns}_label'].mean()
    tagid_count_mean = trn_tmp[f'{columns}_count'].mean()
    tagid_buy_count_mean = trn_tmp[f'{columns}_buy_count'].mean()
    values = {f'{columns}_label': tagid_label_mean, 
                f'{columns}_count': tagid_count_mean, 
                f'{columns}_buy_count': tagid_buy_count_mean}
    tst_tmp = tst_tmp.fillna(value=values)

    trn_tmp = trn_tmp.groupby(['userid'])[f'{columns}_label', f'{columns}_count', f'{columns}_buy_count'].agg(["mean", "std"]).reset_index()
    trn_tmp.columns = [x+"_"+y if y != '' else x for x, y in trn_tmp.columns.values]
    trn_tmp.to_feather(f'./feats/time_feats/复赛_500_train_time_{columns}.feather')

    tst_tmp = tst_tmp.groupby(['userid'])[f'{columns}_label', f'{columns}_count', f'{columns}_buy_count'].agg(["mean", "std"]).reset_index()
    tst_tmp.columns = [x+"_"+y if y != '' else x for x, y in tst_tmp.columns.values]
    tst_tmp.to_feather(f'./feats/time_feats/复赛_500_test_time_{columns}.feather')

    del tst_tmp
    del trn_tmp


def time_preprocess(data):
    ## 删去非2019 2020 2021 的数据
    data.loc[data.pid == 1424548, 'timestamp'] = data.loc[data.pid == 1424548, 'timestamp'].map(lambda x:[i + 365*24*60*60 for i in x])

    pid = []
    tagid = []
    timestamp = []

    for sub_data in data.values:
        s_tagid = sub_data[4]
        s_time = sub_data[5]
        for x, y in zip(s_tagid, s_time):
            pid.append(sub_data[0])
            tagid.append(x)
            timestamp.append(y)

    new_data = pd.DataFrame()

    new_data['pid'] = pid
    new_data['tagid'] = tagid
    new_data['timestamp'] = timestamp

    new_data['timestamp'] = new_data['timestamp'].astype(float)
    new_data['timestamp'] = new_data['timestamp'].astype('int')

    new_data['date'] = pd.to_datetime(list(new_data['timestamp']), unit='s', utc=True).tz_convert('Asia/Shanghai').strftime("%Y-%m-%d %H:%M:%S")
    new_data['date'] = pd.to_datetime(new_data['date'], format = "%Y-%m-%d %H:%M:%S")

    new_data['year'] = new_data['date'].parallel_apply(lambda x: x.year).astype('int16')

    # 仅保留2019-2021年的数据
    new_data = new_data[(new_data['year']>=2019)&(new_data['year']<=2021)]

    # 按照时间信息，抽取对应的tagid和time信息
    new_tagid_list = new_data.groupby(['pid'])['tagid'].apply(list).reset_index()
    new_tagid_list.columns = ['pid','new_tagid']
    # 按照时间信息，抽取对应的tagid和time信息
    new_time_list = new_data.groupby(['pid'])['timestamp'].apply(list).reset_index()
    new_time_list.columns = ['pid','new_timestamp']

    data = pd.merge(data,new_tagid_list,on=['pid'],how='left')
    data = pd.merge(data,new_time_list,on=['pid'],how='left')

    del pid
    del tagid
    del timestamp
    del new_data
    del new_tagid_list
    del new_time_list
    gc.collect()

    train_temp = data.loc[data.pid <= 1400000]
    test_temp = data.loc[data.pid >= 1400001]
    train_temp = train_temp.dropna(subset=["new_tagid", "new_timestamp"])
    data = pd.concat([train_temp, test_temp])

    return data


def gen_time_feats(data):
    # 提取年 月 小时 信息 季节
    data['time'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(i+28800)) for i in x])
    data['year'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%Y", time.localtime(i+28800)) for i in x])
    data['month'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%m", time.localtime(i+28800)) for i in x])
    data['day'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%d", time.localtime(i+28800)) for i in x])
    data['hour'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%H", time.localtime(i+28800)) for i in x])

    #  0 - 6 星期天-星期六
    data['DayOfWeek'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%w", time.localtime(i+28800)) for i in x])

    for f in ['year', 'month', 'day', 'hour', 'DayOfWeek']:
        data[f] = data[f].map(lambda x: [int(i) for i in x])
        
    def quater(x):
        for i, v in enumerate(x):
            if v in [3, 4, 5]:
                x[i] = 0
            elif v in [6, 7, 8]:
                x[i] = 1
            elif v in [9, 10, 11]:
                x[i] = 2
            else:
                x[i] = 3
        return x

    data['Quarter'] = data.loc[:, 'month'].copy(deep=True)
    data['Quarter'] = data['Quarter'].parallel_apply(quater)
    data['month'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%m", time.localtime(i+28800)) for i in x])
    data['month'] = data['month'].parallel_apply(lambda x: [int(i) for i in x])

    # 距离周末的时间差    
    data['day_to_weekend'] = data['DayOfWeek'].parallel_apply(lambda x: [6 - i if i >= 1 and i <= 5 else 0 for i in x])
    data['day_to_weekend_mean'] = data['DayOfWeek'].parallel_apply(lambda x: np.mean(x))

    # 是否是一年的开始
    data['month_day'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%m-%d", time.localtime(i+28800)) for i in x])
    data['is_year_start_mean'] = data['month_day'].parallel_apply(lambda x: np.mean([1 if i=="01-01" else 0 for i in x]))

    # 统计周末数量 时段数量 季节数量 年份数量 月初数量 月末数
    def weekend_mean(x):
        nums = 0
        for i in x:
            if i in [0,6]:
                nums += 1
        return nums/len(x)

    def friday_mean(x):
        nums = 0
        for i in x:
            if i == 5:
                nums += 1
        return nums/len(x)

    data['weekend_mean'] = data['DayOfWeek'].parallel_apply(weekend_mean)
    data['friday_mean'] = data['DayOfWeek'].parallel_apply(friday_mean)

    # 统计周末数量 时段数量 季节数量 年份数量 月初数量 月末数
    def month_start_end_mean(x, seq):
        nums = 0
        if seq == 'start':
            for i in x:
                if i == 1:
                    nums += 1
        else:
            for i in x:
                if (i == 31) or (i == 30):
                    nums += 1
        return nums/len(x)


    def month_5_6_mean(x, seq):
        nums = 0
        for i in x:
            if i == seq:
                nums += 1
        return nums/len(x)


    data['month_start_mean'] = data['day'].parallel_apply(lambda x: month_start_end_mean(x, 'start'))
    data['month_end_mean'] = data['day'].parallel_apply(lambda x: month_start_end_mean(x, 'end'))
    data['month_5_mean'] = data['month'].parallel_apply(lambda x: month_5_6_mean(x, 5))
    data['month_6_mean'] = data['month'].parallel_apply(lambda x: month_5_6_mean(x, 6))

    # 统计时段数量
    def hour_seq_mean(x, seq):
        if seq == 0:
            nums = 0
            for i in x:
                if i >= 0 and i < 7:
                    nums += 1
        elif seq == 1:
            nums = 0
            for i in x:
                if i>=7 and i < 13:
                    nums += 1
        elif seq == 2:
            nums = 0
            for i in x:
                if i>=13 and i < 19:
                    nums += 1
        else:
            nums = 0
            for i in x:
                if  i>=19 and i < 24:
                    nums += 1
        return nums / len(x)

    # 计算不同时段数量
    data['hour_seg_0_mean'] = data['hour'].parallel_apply(lambda x: hour_seq_mean(x, 0))
    data['hour_seg_1_mean'] = data['hour'].parallel_apply(lambda x: hour_seq_mean(x, 1))
    data['hour_seg_2_mean'] = data['hour'].parallel_apply(lambda x: hour_seq_mean(x, 2))
    data['hour_seg_3_mean'] = data['hour'].parallel_apply(lambda x: hour_seq_mean(x, 3))

    def quater_mean(x, seq):
        nums = 0
        for i in x:
            if i == seq:
                nums += 1
        return nums / len(x)

    # 计算不同季节数量
    data['Quarter_0_mean'] = data['Quarter'].parallel_apply(lambda x: quater_mean(x, 0))
    data['Quarter_1_mean'] = data['Quarter'].parallel_apply(lambda x: quater_mean(x, 1))
    data['Quarter_2_mean'] = data['Quarter'].parallel_apply(lambda x: quater_mean(x, 2))
    data['Quarter_3_mean'] = data['Quarter'].parallel_apply(lambda x: quater_mean(x, 3))

    def year_mean(x, year):
        nums = 0
        for i in x:
            if i == year:
                nums += 1
        return nums / len(x)

    # 计算不同年份数量
    data['year_2021_mean'] = data['year'].parallel_apply(lambda x: year_mean(x, 2021))
    data['year_2020_mean'] = data['year'].parallel_apply(lambda x: year_mean(x, 2020))
    data['year_2019_mean'] = data['year'].parallel_apply(lambda x: year_mean(x, 2019))

    # 计算time month day DayOfWeek 的mean max min median
    for f in ['new_timestamp','month']:
        data['{}_mean'.format(f)] = data[f].parallel_apply(lambda x: np.mean(x))
        data['{}_max'.format(f)] = data[f].parallel_apply(lambda x: np.max(x))
        data['{}_min'.format(f)] = data[f].parallel_apply(lambda x: np.min(x))
        data['{}_median'.format(f)] = data[f].parallel_apply(lambda x: np.median(x))
    
    data['pid_tagid_change_time'] = data['new_timestamp'].parallel_apply(lambda x: len(list(set(x))))
    data['date_ymd'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%Y-%m-%d", time.localtime(i+28800)) for i in x])
    data['date_ymd'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%Y-%m-%d", time.localtime(i+28800)) for i in x])
    data['pid_tagid_change_time_ymd'] = data['date_ymd'].parallel_apply(lambda x: len(list(set(x))))

    def shopping_festival_mean(x):
        shopping_festival = [
        '2019-06-18', '2019-11-11', 
        '2020-06-18', '2020-11-11',
        '2021-06-18', '2021-11-11',
    ]
        nums = 0
        for i in x:
            if i in shopping_festival:
                nums += 1
        return nums / len(x)

    data['shopping_festival_mean'] = data['date_ymd'].parallel_apply(shopping_festival_mean)
    # 计算两次关键词时间差均值 最大值 最小值
    def diff(x):
        if len(x) == 1:
            return 0
        diff_list = []
        x = sorted(x)
        for i in range(len(x) - 1):
            diff_list.append(x[i + 1] - x[i])
        return diff_list

    data['time_diff_mean'] = data['new_timestamp'].parallel_apply(lambda x: np.mean(diff(x)))
    data['time_diff_max'] = data['new_timestamp'].parallel_apply(lambda x: np.max(diff(x)))
    data['time_diff_min'] = data['new_timestamp'].parallel_apply(lambda x: np.min(diff(x)))
    data['time_diff_var'] = data['new_timestamp'].parallel_apply(lambda x: np.var(diff(x)))

    # 计算时间的峰度和偏度
    data['time_skew'] = data['new_timestamp'].parallel_apply(lambda x: stats.skew(np.array(x)))
    data['time_kurtosis'] = data['new_timestamp'].parallel_apply(lambda x: stats.kurtosis(np.array(x)))

    data['time_count'] = data.new_timestamp.parallel_apply(lambda x: len(x))
    data['time_count'] = data['time_count'].astype(int)

    # 对用户点击/关键词数量进行划分
    def timecountSeg(x):  
        if x >=0 and x<10:  
            return 1  
        elif x>=10 and x<30:  
            return 2
        elif x>=30 and x<50:  
            return 3 
        elif x>=50 and x<100:  
            return 4 
        else:  
            return 5 
        
    data['time_count_seg'] = data['time_count'].parallel_apply(lambda x: timecountSeg(x)).astype('int8')

    data['time_long'] = data['new_timestamp_max'] - data['new_timestamp_min']
    # 对用户留存时长进行划分
    def timelongSeg(x):  
        if x >=0 and x<86400:  
            return 1  
        elif x>=86400 and x<2592000:  
            return 2
        elif x>=2592000 and x<15768000:  
            return 3
        elif x>=15768000 and x<31536000:  
            return 4
        elif x>=31536000 and x<47304000:  
            return 5
        else:  
            return 6
        
    data['time_long_seg'] = data['time_long'].parallel_apply(lambda x: timelongSeg(x)).astype('int8')
    
    time_features = [
        'userid', 'timestamp', 'new_timestamp', 'time',
        'weekend_mean', 'friday_mean', 'month_start_mean', 'month_end_mean', 
        'month_5_mean', 'month_6_mean', 
        'hour_seg_0_mean', 'hour_seg_1_mean', 'hour_seg_2_mean','hour_seg_3_mean', 
        'Quarter_0_mean', 'Quarter_1_mean', 'Quarter_2_mean','Quarter_3_mean',
        'year_2021_mean', 'year_2020_mean', 'year_2019_mean',
        'new_timestamp_mean', 'new_timestamp_max', 'new_timestamp_min','new_timestamp_median',
        'month_mean', 'month_max', 'month_min', 'month_median',
        'pid_tagid_change_time', 'pid_tagid_change_time_ymd',
        'shopping_festival_mean', 
        'time_diff_mean', 'time_diff_max', 'time_diff_min', 'time_diff_var',
        'time_skew', 'time_kurtosis',
        'time_count', 'time_count_seg', 'time_long', 'time_long_seg',
        'day_to_weekend_mean', 'is_year_start_mean' 
        ]
    data = data.reset_index(drop=True)
    data[time_features].to_feather('./feats/复赛_time_feats.feather')

    
    data['Hour'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%H", time.localtime(i+28800)) for i in x])
    data['Y-M-D'] = data['new_timestamp'].parallel_apply(lambda x:[time.strftime("%Y-%m-%d", time.localtime(i+28800)) for i in x])
    for column in ['Y-M-D', 'Hour']:
        timeTargetEncoder(data[['userid', 'label', 'Hour', 'Y-M-D']], column)

def tagTargetEncoder(data):
    tagid_path = './feats/tagid_feats'
    if not os.path.exists(tagid_path):
        os.mkdir(tagid_path)
    # 处理tag标签
    train = data.loc[data.pid <= 1400000,:]
    test = data.loc[data.pid >= 1400001, :]
    train['tagid'] = train['tagid'].astype(str).str.replace('[', '').str.replace(']', '')
    test['tagid'] = test['tagid'].astype(str).str.replace('[', '').str.replace(']', '')
    
    del test['label']
    temp_train = train.set_index(["userid"])['tagid'].astype(str).str.split(",", expand=True).stack().reset_index(drop=True,level=-1).reset_index().rename(columns={0: 'tagid'})
    temp_train['tagid_count'] = temp_train.groupby("tagid")['tagid'].transform("count")
    temp_train = temp_train[temp_train['tagid_count']>=500]
    
    temp_test = test.set_index(["userid"])['tagid'].str.split(",", expand=True).stack().reset_index(drop=True,level=-1).reset_index().rename(columns={0: 'tagid'})    
    del train['tagid']
    del test['tagid']
    gc.collect()
    
    train = train.merge(temp_train, on="userid", how="left")
    train = train.dropna(subset=['tagid'])
    
    test = test.merge(temp_test, on="userid", how="left")
    train['tagid_buy_count'] = train.groupby(['tagid'])['label'].transform("sum")
    train['tagid_label'] = train.groupby(['tagid'])['label'].transform("mean")
    trn_tmp = train.groupby(['userid', 'tagid'])['tagid_count','tagid_buy_count','tagid_label'].mean().reset_index()
    tst_tmp = test[['userid', 'tagid']].merge(trn_tmp[['tagid', 'tagid_count','tagid_buy_count','tagid_label']].drop_duplicates(subset=['tagid']),
                                     on='tagid', how='left')
    tagid_label_mean = trn_tmp[f'tagid_label'].mean()
    tagid_count_mean = trn_tmp[f'tagid_count'].mean()
    tagid_buy_count_mean = trn_tmp[f'tagid_buy_count'].mean()
    values = {f'tagid_label': tagid_label_mean, 
              f'tagid_count': tagid_count_mean, 
              f'tagid_buy_count': tagid_buy_count_mean}
    tst_tmp = tst_tmp.fillna(value=values)
    
    trn_tmp = trn_tmp.groupby(['userid'])['tagid_label', 'tagid_count', 'tagid_buy_count'].agg(["sum", "max", "min", "mean", "std"]).reset_index()
    trn_tmp.columns = [x+"_"+y if y != '' else x for x, y in trn_tmp.columns.values]
    trn_tmp.to_feather(f'./feats/tagid_feats/复赛_500_train_tagid.feather')
    
    tst_tmp = tst_tmp.groupby(['userid'])['tagid_label', 'tagid_count', 'tagid_buy_count'].agg(["sum", "max", "min", "mean", "std"]).reset_index()
    tst_tmp.columns = [x+"_"+y if y != '' else x for x, y in tst_tmp.columns.values]
    tst_tmp.to_feather(f'./feats/tagid_feats/复赛_500_test_tagid.feather')
    
    del tst_tmp
    del trn_tmp

    for f in ["age", "province", "province_area", "city", 
              "city_level","city_cluster", "model_cluster", "make_cluster"]:
        train[f'{f}_tagid_count'] = train.groupby([f, 'tagid'])['tagid'].transform("count")
        train[f'{f}_tagid_buy_count'] = train.groupby([f, 'tagid'])['label'].transform("sum")
        train[f'{f}_tagid_label'] = train.groupby([f, 'tagid'])['label'].transform("mean")
        train = train[train[f'{f}_tagid_count']>=500]
        
    for f in ["age", "province", "province_area", "city", 
              "city_level","city_cluster", "model_cluster","make_cluster"]:
        tmp = train.groupby([f, 'tagid'])[f'{f}_tagid_label', f'{f}_tagid_buy_count', f'{f}_tagid_count'].mean().reset_index()
        
        ts_re = test.merge(tmp, on=[f, 'tagid'], how='left')
        f_tagid_label_mean = tmp[f'{f}_tagid_label'].mean()
        f_tagid_count_mean = tmp[f'{f}_tagid_count'].mean()
        f_tagid_buy_count_mean = tmp[f'{f}_tagid_buy_count'].mean()
        values = {f'{f}_tagid_label': f_tagid_label_mean, 
                  f'{f}_tagid_count': f_tagid_count_mean, 
                  f'{f}_tagid_buy_count': f_tagid_buy_count_mean}
        ts_re = ts_re.fillna(value=values)
        ts_re_feats = ts_re.groupby(['userid', f])[f'{f}_tagid_label', f'{f}_tagid_buy_count', f'{f}_tagid_count'].agg(["sum", "max", "min", "mean", "std"]).reset_index()
        ts_re_feats.columns = [x+"_"+y if y != '' else x for x, y in ts_re_feats.columns.values]
        ts_re_feats.to_feather(f'./feats/tagid_feats/复赛_500_test_{f}_tagid.feather')
        
        tr_re_feats = train.groupby(['userid', f])[f'{f}_tagid_label', f'{f}_tagid_buy_count', f'{f}_tagid_count'].agg(["sum", "max", "min", "mean", "std"]).reset_index()
        tr_re_feats.columns = [x+"_"+y if y != '' else x for x, y in tr_re_feats.columns.values]
        tr_re_feats.to_feather(f'./feats/tagid_feats/复赛_500_train_{f}_tagid.feather')

def gen_base_feats(data):
    train_temp = data.loc[data.pid <= 1400000]
    test_temp = data.loc[data.pid >= 1400001]
    for f in ['age','province','province_area', 'city', 'city_level',
            'city_cluster', 'model_cluster', 'make_cluster']:
        train_temp[f'{f}_count'] = train_temp.groupby(f)[f].transform('count')
        train_temp[f'{f}_sale_count'] = train_temp.groupby(f)['label'].transform('sum')
        train_temp[f'{f}_sale_rato'] = train_temp.groupby(f)['label'].transform('mean')
        temp = train_temp.groupby(f)[[f'{f}_count', f'{f}_sale_count', f'{f}_sale_rato']].mean().reset_index()
        test_temp = test_temp.merge(temp, on=f, how='left')
    test_model = test_temp.model.unique()
    test_model_make = test_temp.model_make.unique()
    data = pd.concat([train_temp, test_temp], axis=0)
    data['model_in'] = data['model'].parallel_apply(lambda x: 1 if x in test_model else 0)
    data['model_make_in'] = data['model_make'].parallel_apply(lambda x: 1 if x in test_model_make else 0)
    del train_temp
    del test_temp
    columns = ['userid', 'pid', 'label', 'gender', 'age', 'tagid', 'province',
           'province_area', 'city', 'city_level', "city_cluster", 'model', 'make', 'model_make',
           'model_cluster', 'make_cluster', 
           'age_count', 'age_sale_count', 'age_sale_rato',
           'province_count', 'province_sale_count', 'province_sale_rato',
           'province_area_count', 'province_area_sale_count', 'province_area_sale_rato',
           'city_count', 'city_sale_count', 'city_sale_rato', 
            'city_level_count', 'city_level_sale_count', 'city_level_sale_rato',
           'city_cluster_count', 'city_cluster_sale_count', 'city_cluster_sale_rato',
           'model_cluster_count', 'model_cluster_sale_count', 'model_cluster_sale_rato',
           'make_cluster_count', 'make_cluster_sale_count', 'make_cluster_sale_rato', 
           'model_in', 'model_make_in'
          ]
    data = data.reset_index(drop=True)
    data[columns].to_feather("./feats/复赛_base_feats.feather")

def gen_svd_feats(data):
    # tagid
    data['tagid_temp'] = data['tagid'].parallel_apply(lambda x:['tagid'+str(i) for i in x])
    data['tagid_temp'] = data['tagid_temp'].parallel_apply(lambda x:' '.join(x))

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
    from sklearn.decomposition import TruncatedSVD

    tfidf   = TfidfVectorizer(max_df=0.95, min_df=3, ngram_range=(1, 2), sublinear_tf=True)
    res     = tfidf.fit_transform(data['tagid_temp'])

    n_components = 100
    svd     = TruncatedSVD(n_components=n_components, random_state=2021)
    svd_res = svd.fit_transform(res)
    tfidf_svd_feature = []
    for i in (range(n_components)):
        data['tfidf_svd_{}'.format(i)] = svd_res[:, i]
        data['tfidf_svd_{}'.format(i)] = data['tfidf_svd_{}'.format(i)].astype(np.float32)
        tfidf_svd_feature.append('tfidf_svd_{}'.format(i))

    del data['tagid_temp']
    del tfidf
    del res
    del svd
    del svd_res
    gc.collect()
    save_columns = ['userid'] + tfidf_svd_feature
    data = data.reset_index(drop=True)
    data[save_columns].to_feather(f'./feats/复赛_tagid_svd{n_components}.feather')

def w2v_preprocess(data):

    save_path = './预训练模型'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    from gensim.models.fasttext import FastText
    embed_size = 200
    MAX_SEQUENCE_LENGTH = 128
    LEARNING_RATE = 0.001
    fast_WINDOW = 1
    w2v_WINDOW = 1

    data['tagid'] = data['tagid'].apply(lambda x: eval(str(x)))
    data['tagid'] = data['tagid'].apply(lambda x:[str(i) for i in x])

    FastText_model = FastText(sentences=data['tagid'].tolist(), vector_size=embed_size, window=fast_WINDOW, min_count=3)
    FastText_model.save(f'./预训练模型/复赛_tagid_{embed_size}维_window{fast_WINDOW}_FastText_model.model')

    w2v_model = Word2Vec(sentences=data['tagid'].tolist(), vector_size=embed_size, window=w2v_WINDOW, min_count=3, negative=5, hs=0)
    w2v_model.save(f'./预训练模型/复赛_tagid_{embed_size}维_window{w2v_WINDOW}_word2vec_model.model')

def gen_sim_feats(data):
    # 加载训练好的词向量模型
    w2v_model = Word2Vec.load(f'./预训练模型/复赛_tagid_200维_window1_word2vec_model.model')
    vocab = w2v_model.wv.key_to_index

    # 得到任意text的vector
    def get_vector(word_list):
        # 建立一个全是0的array
        res =np.zeros([200])
        count = 0
        for word in word_list:
            if word in vocab:
                res += w2v_model.wv[word]
                count += 1
        if count == 0:
            return res
        else:
            return res/count
    
    data['tagid_vec'] = [get_vector(x) for x in data['tagid']]
    data['tagid_vec'] = data['tagid_vec'].parallel_apply(lambda x:[float(i) for i in x])
    tagid_vec_model_make = data.groupby('model_make')['tagid_vec'].agg(list)
    dict_grouped = {'model_make':tagid_vec_model_make.index,'numbers':tagid_vec_model_make.values}
    df_grouped = pd.DataFrame(dict_grouped)
    df_grouped['vec'] = df_grouped['numbers'].apply(lambda x: np.array(x).mean(axis = 0))
    del df_grouped['numbers']
    gc.collect()

    df_grouped.rename(columns={'vec': 'model_make_vec'}, inplace=True)
    data = pd.merge(data, df_grouped, on=['model_make'], how='left')

    def w2v_cos_sim(x, y):
        try:
            sim = 1 - cosine(x, y)
            return float(sim)
        except:
            return float(0)

    data['w2v_cos_sim_in_tagid_model_make'] = data.apply(lambda x: w2v_cos_sim(x['tagid_vec'], x['model_make_vec']), axis=1)
    save_columns = [
        'userid','w2v_cos_sim_in_tagid_model_make' 
    ]
    data = data.reset_index(drop=True)
    data[save_columns].to_feather('./feats/复赛_相似度特征.feather')

def gen_feats():
    # 读取数据，简单处理list数据
    train_data = pd.read_csv('./data/复赛/train.txt',header=None)
    test_data = pd.read_csv('./data/复赛/test.txt',header=None)

    train_data.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'timestamp', 'province', 'city', 'model', 'make']
    test_data.columns = ['pid', 'gender', 'age', 'tagid', 'timestamp', 'province', 'city', 'model', 'make']

    train_data['userid'] = train_data.pid.astype(str).values + '_train2'
    test_data['userid'] = test_data.pid.astype(str).values + '_test'
    test_data_notnull = test_data[test_data.tagid.notnull().T]
    train_data_notnull = train_data[train_data.tagid.notnull().T]
    data = pd.concat([train_data_notnull,test_data_notnull])

    data = preprocess(data)
    data = time_preprocess(data)
    gen_time_feats(data)
    tagTargetEncoder(data)
    gen_base_feats(data)
    gen_svd_feats(data)
    w2v_preprocess(data)
    gen_sim_feats(data)


if __name__ == "__main__":
    feat_path = "./feats"
    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
    gen_feats()