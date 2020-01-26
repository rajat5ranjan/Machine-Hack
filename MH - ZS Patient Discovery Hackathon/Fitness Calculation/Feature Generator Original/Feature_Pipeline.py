# -*- coding: utf-8 -*-

""" This is the expected File Format for Feature_pipeline.py script """

## import required libraries

from feature_generators import calc_recency,calc_freq,calc_norm
import time 
from itertools import repeat
import numpy as np # linear algebra
import time
import pandas as pd
import os


# load the training set - Do not edit this line
train_transaction_df = pd.read_csv("train_data.csv",usecols=['patient_id','event_name','specialty','plan_type','event_time']) #The train dataset name an path should remail unchanged train_data.csv- Do not edit this line
# append the train label.

#load the test set- Do not edit this line
# test_transaction_df = pd.read_csv("test_data.csv") #The test dataset name an path should remail unchanged train_data.csv- Do not edit this line

time_var = 'event_time'
id_var = 'patient_id'
y_var = 'outcome_flag'
labels = pd.read_csv('train_labels.csv')

train_data = pd.merge(train_transaction_df, labels, on='patient_id', how='left')
print('DataFrame Loaded...')


'''Auto Feature Engineering: Once the set of mandatory features using the training data,
evaluate fitness of these features using the following methodology
Fitness Value =
Considerations:
– For recency features, remove all 999999999 values while calculating the fitness value
– For Frequency and Normchange features, If a patient does not have a particular event /specialty /plan –
feature value will be 0 and will be used for calculating the fitness value
'''
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
def create_recency_features():
    column = ['event_name','plan_type','specialty']
    res_recency = calc_recency(column,train_data)
    res_recency['fitness_value'] = res_recency.apply(fitness_calculation, axis=1)
    return res_recency

# define functions for frequency calculation.
def create_frequency_features():
    column = ['event_name','plan_type','specialty']
    res_freq = calc_freq(column,train_data)
    res_freq['fitness_value'] = res_freq.apply(fitness_calculation, axis=1)
    return res_freq

# define functions for normchange calculation.
def create_normchange_features():
    column = ['event_name','plan_type','specialty']
    # column = ['specialty']
    res_normChange = calc_norm(column,train_data)
    res_normChange['fitness_value'] = res_normChange.apply(fitness_calculation, axis=1)
    return res_normChange


if __name__ == '__main__':
    try:
        start=time.time()
        res=pd.DataFrame()
        rec = create_recency_features()
        freq = create_frequency_features()
        normCh = create_normchange_features()
        res=rec.append(freq,ignore_index=True)
        res=res.append(normCh,ignore_index=True)
        res.to_csv('final_fit_v5.csv',index=False)
        print('Done',time.time()-start)
    except:
        print('Except : Exception Occurred',time.time()-start)
    finally:
        print('Finally : Done',time.time()-start)