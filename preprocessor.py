#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

__author__ = "Bo-Syun Cheng"
__email__ = "k12s35h813g@gmail.com"


# In[2]:


class Data_preprocessor():
    def __init__(self,data,filter_user=1,filter_item=5):
        self.data = data
        self.filter_user = filter_user
        self.filter_item = filter_item

    def preprocess(self):
        self.filter_()
        return self.train_test_split()
    def filter_(self):
        """
        過濾掉session長度過短的user和評分數過少的item

        :param filter_user: 少於這個session長度的user要被過濾掉 default=1
        :param filter_item: 少於這個評分數的item要被過濾掉 default=5
        :return: dataframe
        """
        session_lengths = self.data.groupby('userId').size()
        self.data = self.data[np.in1d(self.data['userId'], session_lengths[session_lengths>1].index)] #將長度不足2的session過濾掉
        print("剩餘data : %d"%(len(self.data)))
        item_supports = self.data.groupby('movieId').size() #統計每個item被幾個使用者評過分
        self.data = self.data[np.in1d(self.data['movieId'], item_supports[item_supports>5].index)] #將被評分次數低於5的item過濾掉
        print("剩餘data : %d"%(len(self.data)))
        """再把只有一個click的user過濾掉 因為過濾掉商品可能會導致新的單一click的user出現"""
        session_lengths = self.data.groupby('userId').size()
        self.data = self.data[np.in1d(self.data['userId'], session_lengths[session_lengths>1].index)]
        print("剩餘data : %d"%(len(self.data)))
    def train_test_split(self,time_range=86400):
        """
        切割訓練和測試資料集

        :param time_range:session若在這個區間內，將被分為test_data default=86400(1day)
        :retrun: a tuple of two dataframe
        """
        tmax = self.data['timestamp'].max()
        session_tmax = self.data.groupby('userId')['timestamp'].max()
        train = self.data[np.in1d(self.data['userId'] , session_tmax[session_tmax<=tmax -86400].index)]
        test = self.data[np.in1d(self.data['userId'] , session_tmax[session_tmax>tmax -86400].index)]
        print("訓練資料集統計:  session個數:%d , item個數:%d , event數:%d"%(train['userId'].nunique(),train['movieId'].nunique(),len(train)))
        """
        基於協同式過濾的特性，若test data中含有train data沒出現過的item，將該item過濾掉
        """
        test = test[np.in1d(test['movieId'], train['movieId'])]
        tslength = test.groupby('userId').size()
        test = test[np.in1d(test['userId'], tslength[tslength>=2].index)]
        print("測試資料集統計:  session個數:%d , item個數:%d , event數:%d"%(test['userId'].nunique(),test['movieId'].nunique(),len(test)))

        return train
