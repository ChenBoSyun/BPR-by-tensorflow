#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import time
import tensorflow as tf
import random
import math

__author__ = "Bo-Syun Cheng"
__email__ = "k12s35h813g@gmail.com"


class BPR():
    '''
    parameter
    train_sample_size : 訓練時，每個正樣本，我sample多少負樣本
    test_sample_size : 測試時，每個正樣本，我sample多少負樣本
    num_k : item embedding的維度大小
    evaluation_at : recall@多少，及正樣本要排前幾名，我們才視為推薦正確
    '''
    def __init__(self,data,n_epochs=10,batch_size=32,train_sample_size=10,test_sample_size=50,num_k=100,evaluation_at=10):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.num_k = num_k
        self.evaluation_at = evaluation_at

        self.data = data
        self.num_user = len(self.data['userId'].unique())
        self.num_item = len(self.data['movieId'].unique())
        self.num_event = len(self.data)

        self.all_item = set(self.data['movieId'].unique())
        self.experiment = []

        #Because the id is not always continuous , we build a map to normalize id . For example:[1,3,5,156]->[0,1,2,3]
        user_id = self.data['userId'].unique()
        self.user_id_map = {user_id[i] : i for i in range(self.num_user)}
        item_id = self.data['movieId'].unique()
        self.item_id_map = {item_id[i] : i for i in range(self.num_item)}
        training_data = self.data.loc[:,['userId','movieId']].values
        self.training_data = [[self.user_id_map[training_data[i][0]],self.item_id_map[training_data[i][1]]] for i in range(self.num_event)]



        #data preprocess
        self.split_data() #split data into training_data and testing
        self.sample_dict = self.negative_sample() #for each trainging data (user,item+) , we sample 10 negative item for bpr training

        self.build_model() #build TF graph
        self.sess = tf.Session() #create session
        self.sess.run(tf.global_variables_initializer())


    def split_data(self):
        user_session = self.data.groupby('userId')['movieId'].apply(set).reset_index().loc[:,['movieId']].values.reshape(-1)
        self.testing_data =[]
        for index,session in enumerate(user_session):
            random_pick = self.item_id_map[random.sample(session,1)[0]]
            self.training_data.remove([index,random_pick])
            self.testing_data.append([index,random_pick])


    def negative_sample(self):
        user_session = self.data.groupby('userId')['movieId'].apply(set).reset_index().loc[:,['movieId']].values.reshape(-1)
        sample_dict = {}

        for td in self.training_data:
            sample_dict[tuple(td)] = [self.item_id_map[s] for s in random.sample(self.all_item.difference(user_session[td[0]]) , self.train_sample_size)]

        return sample_dict

    def build_model(self):
        self.X_user = tf.placeholder(tf.int32,shape=(None , 1))
        self.X_pos_item = tf.placeholder(tf.int32,shape=(None , 1))
        self.X_neg_item = tf.placeholder(tf.int32,shape=(None , 1))
        self.X_predict = tf.placeholder(tf.int32,shape=(1))

        user_embedding = tf.Variable(tf.truncated_normal(shape=[self.num_user,self.num_k],mean=0.0,stddev=0.5))
        item_embedding = tf.Variable(tf.truncated_normal(shape=[self.num_item,self.num_k],mean=0.0,stddev=0.5))

        embed_user = tf.nn.embedding_lookup(user_embedding , self.X_user)
        embed_pos_item = tf.nn.embedding_lookup(item_embedding , self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(item_embedding , self.X_neg_item)

        pos_score = tf.matmul(embed_user , embed_pos_item , transpose_b=True)
        neg_score = tf.matmul(embed_user , embed_neg_item , transpose_b=True)

        self.loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(pos_score-neg_score)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        predict_user_embed = tf.nn.embedding_lookup(user_embedding , self.X_predict)
        self.predict = tf.matmul(predict_user_embed , item_embedding , transpose_b=True)

    def fit(self):
        self.experiment = []
        for epoch in range(self.n_epochs):
            np.random.shuffle(self.training_data)
            total_loss = 0
            for i in range(0 , len(self.training_data) , self.batch_size):
                training_batch = self.training_data[i:i+self.batch_size]
                user_id = []
                pos_item_id = []
                neg_item_id = []
                for single_training in training_batch:
                    for neg_sample in list(self.sample_dict[tuple(single_training)]):
                        user_id.append(single_training[0])
                        pos_item_id.append(single_training[1])
                        neg_item_id.append(neg_sample)

                user_id = np.array(user_id).reshape(-1,1)
                pos_item_id = np.array(pos_item_id).reshape(-1,1)
                neg_item_id = np.array(neg_item_id).reshape(-1,1)

                _ , loss = self.sess.run([self.optimizer , self.loss] ,
                           feed_dict = {self.X_user : user_id , self.X_pos_item : pos_item_id , self.X_neg_item : neg_item_id}
                           )
                total_loss += loss

            num_true = 0
            for test in self.testing_data:
                result = self.sess.run(self.predict , feed_dict = {self.X_predict : [test[0]]})
                result = result.reshape(-1)
                if (result[[self.item_id_map[s] for s in random.sample(self.all_item , self.test_sample_size)]] > result[test[1]]).sum()+1 <= self.evaluation_at:
                    num_true += 1

            print("epoch:%d , loss:%.2f , recall:%.2f"%(epoch , total_loss , num_true/len(self.testing_data)))
            self.experiment.append([epoch , total_loss , num_true/len(self.testing_data)])
