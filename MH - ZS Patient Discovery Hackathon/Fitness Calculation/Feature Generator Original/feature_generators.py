import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from itertools import product

'''
Recency features
How recently did an event happen prior to the anchor date?
'''
def calc_recency(cols,train_data):
    print('in calc_recency')
    st=time.time()
    lavg1=[]
    lsd1=[]
    lavg0=[]
    lsd0=[]
    fn=[]

    for k in cols:
        _data=train_data[[k,'patient_id', 'outcome_flag', 'event_time']].groupby(['patient_id',k, 'outcome_flag'])['event_time'].min().reset_index()
        for j in tqdm(train_data[k].unique()):
            patient_level_feature=_data[_data[k]==j].drop([k],axis=1)
            patient_level_feature.columns = ['patient_id', 'outcome_flag', 'feature_value']
            x1=patient_level_feature[(patient_level_feature['outcome_flag']==1) & (patient_level_feature['feature_value']!=9999999999)]['feature_value']
            x0=patient_level_feature[(patient_level_feature['outcome_flag']==0) & (patient_level_feature['feature_value']!=9999999999)]['feature_value']
            lavg1.append(x1.mean())
            lsd1.append(x1.std())
            lavg0.append(x0.mean())
            lsd0.append( x0.std())
            fn.append('recency_'+str(k)+'__'+str(j))
    fitness = pd.DataFrame([fn, lavg1, lavg0, lsd1, lsd0]).transpose().fillna(0)
    fitness.columns = ['feature_name', 'avg_1', 'avg_0', 'sd_1', 'sd_0']
    # print(fitness.shape)
    print('exiting cal recency',time.time()-st,fitness.shape)
    return fitness

'''
Frequency features
How many times did an event happen in a specific time frame?
Data has 3 years of patient history i.e. 36 months resulting in 1 frequency feature per event per month, total of
36 features per event. Hence #frequency features will be equal to 36 times #events
'''
def calc_freq(cols,train_data):
    st=time.time()
    print('in calc_fitness')
    _df3=train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)
    ft=pd.DataFrame()
    for kk in tqdm(cols):
        for t in tqdm(([x for x in range(1080,0,-30)][::-1])):
            _freq = train_data[(train_data[time_var]<=int(t))][[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()
            _freq.columns = [id_var, 'feature_name', 'feature_value']
            _freq['feature_name'] = 'frequency_' + str(t) + '_' +str(kk)+'__'+ _freq['feature_name'].astype(str)
            _freq = _freq.reset_index(drop=True)
            freqTotal=pd.DataFrame(product(train_data[id_var].unique(),_freq['feature_name'].unique()),columns=['patient_id','feature_name'])
            freqTotal = pd.merge(freqTotal, _freq, on=[id_var, 'feature_name'], how='left')
            freqTotal.fillna(0, inplace=True)
            freqTotal = freqTotal.merge(_df3, on=id_var, how='left')
            xx1=freqTotal.loc[freqTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()
            xx1.columns=['feature_name','avg_1','sd_1']
            xx0=freqTotal.loc[freqTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()
            xx0.columns=['feature_name','avg_0','sd_0']
            _fitness_value = pd.merge(xx1, xx0, on='feature_name', how='left')
            ft=ft.append(_fitness_value,ignore_index=True)

    # ft=ft.append(ftns[ftns.feature_name.str.startswith('frequency')][~ftns[ftns.feature_name.str.startswith('frequency')].feature_name.isin(ft.feature_name)])
    for k,j in list(product(train_data['specialty'].unique(),[x for x in range(1080,0,-30)])):
        if 'frequency_' + str(j) + '_specialty' +'__'+ k not in ft.feature_name.values:
            ft=ft.append({'feature_name':'frequency_' + str(j) + '_specialty' +'__'+ k,
                    'avg_1':0,'avg_0':0,'sd_0':0,'sd_1':0},ignore_index=True)
    print('exiting freq in ',time.time()-st,ft.shape)
    return ft

'''
NormChange features
Has the frequency of an event increased or decreased in a recent time
frame (not more than 1.5 years) as compared to the previous time frame?
▪ Data has 36 months of patient history and can be split into two time-periods using 18 split-points i.e. total of 18
features per event (split points 1 months to 18 months)
▪ For e.g. frequency/day of an event within last 2 months vs frequency/day of an event in 2-36 months (34 months)
▪ In the below split example:Comparison will be between frequency / day in 1 year vs frequency / day in earliest 2
years from anchor
▪ Frequency / day in time period x (days: 30,60,90….) = total events / days in the time period
▪ Change in frequency = Frequency per day in time period 1 – Frequency per day in time period (1080 – x)
▪ If change in frequency >0 then 1 else 0
'''
def calc_norm(cols,train_data):
    st=time.time()
    _df3=train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)
    print('in calc_norm')
    ftnorm=pd.DataFrame()
    for kk in tqdm(cols):
        for t in tqdm(([x for x in range(540,0,-30)][::-1])):
            _data_post = train_data[train_data[time_var]<=int(t)].reset_index(drop=True)
            _data_pre = train_data[train_data[time_var]>int(t)].reset_index(drop=True)
            _freq_post = _data_post[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()
            _freq_pre = _data_pre[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()
            _freq_post.columns = [id_var, 'feature_name', 'feature_value_post']
            _freq_pre.columns = [id_var, 'feature_name', 'feature_value_pre']
            _freq_post['feature_value_post'] = _freq_post['feature_value_post']/int(t)
            _freq_pre['feature_value_pre'] = _freq_pre['feature_value_pre']/((train_data[time_var].max()) - int(t))
            _normChange = pd.merge(_freq_post, _freq_pre, on=[id_var, 'feature_name'], how='outer')
            _normChange.fillna(0, inplace=True)
            _normChange['feature_value'] = np.where(_normChange['feature_value_post']>_normChange['feature_value_pre'], 1, 0)
            _normChange.drop(['feature_value_post', 'feature_value_pre'], axis=1, inplace=True)
            _normChange['feature_name'] = 'normChange_' + str(t)+'_'+ kk + '__' + _normChange['feature_name'].astype(str)
            _normChange = _normChange.reset_index(drop=True)

            # _df1 = pd.DataFrame(_normChange['feature_name'].unique().tolist(), columns=['feature_name'])
            # _df2 = pd.DataFrame(_normChange[id_var].unique().tolist(), columns=[id_var])
            # _df1['key'] = 1
            # _df2['key'] = 1
            # _normTotal = pd.merge(_df2, _df1, on='key')
            # _normTotal.drop(['key'], axis=1, inplace=True)
            _normTotal=pd.DataFrame(data=product(_normChange[id_var].unique().tolist(),_normChange['feature_name'].unique().tolist()),columns=['patient_id','feature_name'])
            _normTotal = pd.merge(_normTotal, _normChange, on=[id_var, 'feature_name'], how='left')
            _normTotal.fillna(0, inplace=True)
            
            _normTotal = _normTotal.merge(_df3, on=id_var, how='left')
            normTotal = _normTotal.copy()
            xx1=normTotal.loc[normTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()
            xx1.columns=['feature_name','avg_1','sd_1']
            xx0=normTotal.loc[normTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()
            xx0.columns=['feature_name','avg_0','sd_0']
            _fitness_value = pd.merge(xx1, xx0, on='feature_name', how='left')
            ftnorm=ftnorm.append(_fitness_value)
    print('exiting norm in ',time.time()-st,ftnorm.shape)
    return ftnorm

time_var = 'event_time'
id_var = 'patient_id'
y_var = 'outcome_flag'