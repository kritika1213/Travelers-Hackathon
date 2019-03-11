# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:47:48 2018

@author: kkhat
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset = pd.read_csv("uconn_comp_2018_train_modified.csv")
dataset.describe()
dataset.isnull().sum()
dataset.dtypes
categorical = [var for var in dataset.columns if dataset[var].dtype == 'O']
categorical
dataset[categorical].head()
Numerical = [var for var in dataset.columns if dataset[var].dtype != 'O']
Numerical
dataset[Numerical].head()


categorical
for var in categorical:
    if len(dataset[var].unique()) <= 20:
        print(var, dataset[var].unique())

dataset.boxplot(column = 'age_of_driver' )


##################################dummy variable########
dataset['gender'] = pd.get_dummies(dataset['gender'], drop_first=True)
dataset['living_status'] = pd.get_dummies(dataset['living_status'], drop_first=True)
dataset[['accident_site_local','accident_site_parking']]  = pd.get_dummies(dataset['accident_site'], drop_first=True)
dataset[['channel_online','channel_phone']] = pd.get_dummies(dataset['channel'], drop_first=True)
dataset[['vehicle_category_large','vehicle_category_medium']] = pd.get_dummies(dataset['vehicle_category'], drop_first=True)

#######
dataset_2 = pd.read_csv("uconn_comp_2018_test_modified.csv")
##################################dummy variable########
dataset_2['gender'] = pd.get_dummies(dataset['gender'], drop_first=True)
dataset_2['living_status'] = pd.get_dummies(dataset['living_status'], drop_first=True)
dataset_2[['accident_site_local','accident_site_parking']]  = pd.get_dummies(dataset['accident_site'], drop_first=True)
dataset_2[['channel_online','channel_phone']] = pd.get_dummies(dataset['channel'], drop_first=True)
dataset_2[['vehicle_category_large','vehicle_category_medium']] = pd.get_dummies(dataset['vehicle_category'], drop_first=True)
#######
dataset.columns


training_vars = [var for var in dataset.columns if var not in ['claim_number','fraud','vehicle_color', 'claim_day_of_week','claim_date', 'zip_code','Living_status_numeric']]
X = dataset[training_vars]
Y = dataset.iloc[:,27]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)





########### fit scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train) #  fit  the scaler to the train set and then transform it
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

##############################################
from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(scaler.transform(X_train), Y_train)

from sklearn.metrics import roc_auc_score
pred = logit_model.predict_proba(scaler.transform(X_train))
print('Logit train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
pred = logit_model.predict_proba(scaler.transform(X_test))
print('Logit test roc-auc: {}'.format(roc_auc_score(Y_test, pred[:,1])))

########################################

import xgboost as xgb

xgb_model = xgb.XGBClassifier()

eval_set = [(X_test, Y_test)]
xgb_model.fit(X_train, Y_train, eval_metric="auc", eval_set=eval_set, verbose=False)

pred = xgb_model.predict_proba(X_train)
print('xgb train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
pred = xgb_model.predict_proba(X_test)
print('xgb test roc-auc: {}'.format(roc_auc_score(Y_test, pred[:,1])))

#########################

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

pred = rf_model.predict_proba(X_train)
print('RF train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
pred = rf_model.predict_proba(X_test)
print('RF test roc-auc: {}'.format(roc_auc_score(Y_test, pred[:,1])))








###############################
##predict new values:

X_pred = dataset_2[training_vars]

pred = xgb_model.predict_proba(X_pred)
pred_dataset = pd.DataFrame(pred)
pred_dataset['claim_number'] = dataset_2['claim_number']
pred_dataset.to_csv('pred_dataset_xg.csv')


######################################################
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier()
ada_model.fit(X_train, Y_train)

pred = ada_model.predict_proba(X_train)
print('Adaboost train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
pred = ada_model.predict_proba(X_test)
print('Adaboost test roc-auc: {}'.format(roc_auc_score(Y_test, pred[:,1])))



############ using original train dataset and test dataset############### 

X_train = dataset[training_vars]
X_test =  dataset_2[training_vars]
Y_train = dataset.iloc[:,27]
Y_test
 ####run xgboot code
xgb_model.fit(X_train, Y_train , eval_metric="auc")
pred_y = xgb_model.predict_proba(X_test)
pred_dataset = pd.DataFrame(pred_y)
pred_dataset['claim_number'] = dataset_2['claim_number']
pred_dataset.to_csv('pred_dataset_xg_full.csv')

#######accuracy test
pred = xgb_model.predict_proba(X_train)
print('xgb train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))

logit_model = LogisticRegression()
logit_model.fit(scaler.transform(X_train), Y_train)
pred_y = logit_model.predict_proba(scaler.transform(X_test))
pred_dataset = pd.DataFrame(pred_y)
pred_dataset['claim_number'] = dataset_2['claim_number']
pred_dataset.to_csv('pred_dataset_logit_full.csv')

#####adaboost
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, Y_train)

pred_y = ada_model.predict_proba(X_test)
pred_dataset = pd.DataFrame(pred_y)
pred_dataset['claim_number'] = dataset_2['claim_number']
pred_dataset.to_csv('pred_dataset_adaboost_full.csv')


pred = ada_model.predict_proba(X_train)
print('Adaboost train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
########randomforest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

pred = rf_model.predict_proba(X_train)
print('RF train roc-auc: {}'.format(roc_auc_score(Y_train, pred[:,1])))
pred_y = rf_model.predict_proba(X_test)
pred_dataset = pd.DataFrame(pred_y)
pred_dataset['claim_number'] = dataset_2['claim_number']
pred_dataset.to_csv('pred_dataset_randomForest_full.csv')
