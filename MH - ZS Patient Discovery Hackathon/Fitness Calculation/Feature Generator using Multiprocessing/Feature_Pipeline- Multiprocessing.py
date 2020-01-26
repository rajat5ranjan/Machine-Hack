# -*- coding: utf-8 -*-

""" This is the expected File Format for Feature_pipeline.py script """

## import required libraries

from multiprocessing import Pool
from feature_generators import calc_recency,calc_freq,calc_norm
import time 
from itertools import repeat
import numpy as np # linear algebra
import time
import pandas as pd
import os


def fitness_calculation(data):
    if ((data['sd_0'] == 0 ) and (data['sd_1'] == 0)) and (((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):
        return 9999999999
    elif (((data['sd_0'] == 0 ) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (data['avg_0'] == data['avg_1']):
        return 1
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):
        return ((data['avg_1']/data['sd_1'])/(data['avg_0']/data['sd_0']))
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):
        return 9999999999
    else:
        return 1

# define functions for recency calculation
def create_recency_features(pool):
    column = ['event_name','plan_type','specialty']
    res_recency = pool.apply_async(calc_recency,(column,train_data)).get()
    res_recency['fitness_value'] = res_recency.apply(fitness_calculation, axis=1)
    return res_recency

# define functions for frequency calculation.
def create_frequency_features(pool):
    column = ['event_name','plan_type','specialty']
    res_freq = pool.apply_async(calc_freq,(column,train_data)).get()
    res_freq['fitness_value'] = res_freq.apply(fitness_calculation, axis=1)
    return res_freq


def create_normchange_features(pool):
    column = ['event_name','plan_type','specialty']
    res_normChange = pool.apply_async(calc_norm,(column,train_data)).get()
    res_normChange['fitness_value'] = res_normChange.apply(fitness_calculation, axis=1)
    return res_normChange


if __name__ == '__main__':
    try:
        start=time.time()
        # load the training set - Do not edit this line
        train_transaction_df = pd.read_csv("train_data.csv",nrows=1000,usecols=['patient_id','event_name','specialty','plan_type','event_time']) #The train dataset name an path should remail unchanged train_data.csv- Do not edit this line
        # append the train label.

        #load the test set- Do not edit this line
        # test_transaction_df = pd.read_csv("test_data.csv") #The test dataset name an path should remail unchanged train_data.csv- Do not edit this line

        time_var = 'event_time'
        id_var = 'patient_id'
        y_var = 'outcome_flag'
        labels = pd.read_csv('train_labels.csv')

        train_data = pd.merge(train_transaction_df, labels, on='patient_id', how='left')
        print('DataFrame Loaded...')
        pool = Pool(processes=4)
        res=pd.DataFrame()
        rec = create_recency_features(pool)
        freq = create_frequency_features(pool)
        normCh = create_normchange_features(pool)
        res=rec.append(freq,ignore_index=True)
        res=res.append(normCh,ignore_index=True)
        # print(res.shape)
        res.to_csv('final_fit__pool_v5.csv',index=False)
        print('Done',time.time()-start)
    except:
        print('Except : Exception Occurred',time.time()-start)
        pool.close()
    finally:
        print('Finally : Done',time.time()-start)
        pool.close()