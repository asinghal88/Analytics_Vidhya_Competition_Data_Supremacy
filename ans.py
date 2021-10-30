# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 23:24:12 2018

@author: abc
"""
import gc
gc.collect()
import pandas as pd
import numpy as np
from scipy.stats import norm, skew
import seaborn as sns
import matplotlib.pyplot as plt
train=pd.read_csv('train_jqd04QH.csv')
test=pd.read_csv('test_GYi4Gz5.csv')
#t=pd.concat([train,test],axis=0) 
from sklearn.model_selection import train_test_split 
for i in range (18359):
    if (train.education_level[i]=='Primary School'):
        train.major_discipline[i]='No Major'
for i in range (15021):
    if (test.education_level[i]=='Primary School'):
        test.major_discipline[i]='No Major'
#
#for i in t.columns:
#    if t[i].dtype == object:
#        t[i]=t[i].astype('category')
#
#t.gender = t.gender.cat.codes 
#t.city=t.city.cat.codes
#t.relevent_experience=t.relevent_experience.cat.codes
#t.enrolled_university=t.enrolled_university.cat.codes
#t.education_level=t.education_level.cat.codes
#t.major_discipline =t.major_discipline.cat.codes
#t.experience=t.experience.cat.codes
#t.company_size=t.company_size.cat.codes
#t.company_type=t.company_type.cat.codes
#t.last_new_job =t.last_new_job.cat.codes
#t= t.replace(-1,np.NaN)
#fin1=t.copy()
#fin1.drop('target',axis=1,inplace=True)
#from fancyimpute import  KNN
#fin2 = KNN(k=5).complete(fin1)
#fin2=pd.DataFrame(fin2)
#fin2.columns=fin1.columns
#for i in fin1.columns:
#    if (fin1[i].isnull().values.any()==True):
#        fin2[i]=fin2[i].astype('int64')

for col in list(train.columns.values):
    if col != 'target':
        train[col].fillna('NA',inplace=True)
        test[col].fillna('NA',inplace=True)

from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
     return self.fit(X,y).transform(X)
fin=pd.concat([train,test],axis=0) 
fin.columns
fin.dtypes
#fin.training_hours=np.log1p(fin.training_hours)
ytrain=train.iloc[:,13]
#tra=fin2.iloc[0:18359,:]  
#tra=pd.concat([tra,ytrain],axis=1)
cat_columns = ['city','gender','relevent_experience','enrolled_university','education_level','major_discipline','experience','company_size','company_type','last_new_job']

for col in cat_columns:
    if col != 'target':
        gp_mean = train.groupby(col)['target'].mean().reset_index().rename(columns={'target':col+'_mean'})
        fin=fin.merge(gp_mean,on=col,how='left')
        

fin1=MultiColumnLabelEncoder(columns = ['city', 'company_size', 'company_type',
       'education_level', 'enrolled_university', 'experience',
       'gender', 'last_new_job', 'major_discipline', 'relevent_experience']).fit_transform(fin) 
train1=fin1.iloc[0:18359,:]    
test1=fin1.iloc[18359:,:]
predictors = list(train1.columns.values)
predictors.remove('enrollee_id')
predictors.remove('target')
from sklearn.model_selection import KFold
import lightgbm as lgb
rounds = 1000
params = {'boosting_type': 'gbdt',
#         'max_depth' : -1,
          'objective': 'binary',
          'nthread': 4, # Updated from nthread
          'num_leaves': 31,
          'learning_rate': 0.01,
          'max_bin': 25,
#          'subsample': 1,
#          'subsample_freq': 1,
          'colsample_bytree': 0.7,
          'reg_alpha': 1,
          'reg_lambda': 1,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 2,
          'metric' : 'auc'}

#ytrain=pd.DataFrame(ytrain)

from catboost import Pool, CatBoostClassifier
#X_train, X_test, y_train, y_test = train_test_split(train1, ytrain, test_size=0.3, random_state=42)
#imbalance_weight=y_train.value_counts(normalize=True)[0]/ytrain.value_counts(normalize=True)[1]

X_train = train1.loc[train_index,predictors]
y_train = train1.loc[train_index,'target']
X_test = train1.loc[test_index,predictors]
y_test = train1.loc[test_index,'target']
#X_train=X_train.loc[:,predictors]
#X_test=X_test.loc[:,predictors]
model = CatBoostClassifier(depth=6,
                           od_wait = 100,
                           scale_pos_weight=2,
                           random_seed=42,
                           logging_level = "Verbose",loss_function= 'Logloss',
                           metric_period = 100,iterations=650,learning_rate=0.06,eval_metric='AUC' #fold_len_multiplier = 1.1, l2_leaf_reg = 5
                           )

model1=model.fit(X_train, y_train,cat_features=[0,2,3,4,5,6,7,8,9,10],eval_set=(X_test,y_test))
test1.target=model1.predict_proba(test1[predictors])[:,1]
#test1.target=test1.rank(pct=True)
test1[['enrollee_id','target']].to_csv('sub4.csv',index=False) 
kf = KFold(n_splits=10,random_state=37,shuffle=True)    
for i, (train_index, test_index) in zip(range(1,6),kf.split(train1)):
    X_train = train1.loc[train_index,predictors].values
    y_train = train1.loc[train_index,'target'].values
    X_test = train1.loc[test_index,predictors].values
    y_test = train1.loc[test_index,'target'].values
    X_train = lgb.Dataset(X_train,y_train,feature_name=predictors,categorical_feature=cat_columns)
    X_test = lgb.Dataset(X_test,y_test,feature_name=predictors,categorical_feature=cat_columns)
    model = lgb.train(params,train_set=X_train,valid_sets=X_test,num_boost_round=1000,early_stopping_rounds=200,verbose_eval=100)
   # model1=model.fit(X_train, y_train,cat_features=[0,2,3,4,5,6,7,8,9,10],eval_set=(X_test,y_test),use_best_model=True)
    test1['target_'+str(i)] = model.predict(test1[predictors])  
 
for i in range(1,5):
    test1['target_'+str(i)] = test1['target_'+str(i)].rank(pct=True)
test2=test1.copy() 
test2['target'] = (test2.target_1 + test2.target_2 + test2.target_3 + test2.target_4 + test2.target_5)/5.
test2['target']=test2.iloc[:,24:].sum(axis=1)/5.
test2[['enrollee_id','target']].to_csv('sub3.csv',index=False)    

