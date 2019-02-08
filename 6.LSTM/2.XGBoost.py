# https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import randint
import scipy.stats as stats
from random import *
from sklearn.model_selection import train_test_split
import time


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# coin_data = pd.read_pickle('./data/RNN_coin/X_10_10_1_0.1.pickle')
# coin_data_x_np = np.array(coin_data[0])
# coin_data_y_np = np.array(coin_data[1])
# coin_data_x_np_reshaped = coin_data_x_np.reshape(52550*8, -1)
# coin_data_y_np_reshaped = coin_data_y_np.reshape(-1)
# print("[INFO] coin_data_x_np_reshaped.shape : {}".format(coin_data_x_np_reshaped.shape))
# print("[INFO] coin_data_y_np_reshaped.shape : {}".format(coin_data_y_np_reshaped.shape))
# coin_data_x_pd = pd.DataFrame(coin_data_x_np_reshaped)
# coin_data_y_pd = pd.DataFrame(coin_data_y_np_reshaped)
filename = 'X_10_25_1_0.1'
data = pd.read_pickle('./data/RNN_coin/'+filename+'.pickle')
X = np.array(data[0])[:,1,:,:]
# print(X)
# print(type(X))
# print(X.shape)
# print(type(X[:,1,:,:]))
# print(X[:,1,:,:].shape)
# y = np.array(data[1])
y = np.array(data[1])[:,1]
# print(X)
# print(y)
# print(type(X[:,1,:,:]))
# print(X[:,1,:,:].shape)
X = np.reshape(X, [list(X.shape)[0], 4, 25])
print("X.shape-", y.shape)
X_reshaped = X[:,1] # closing price
# X_reshaped = X.reshape(len(y) * 8, -1)
# y_reshaped = y.reshape(-1)
X_pd = pd.DataFrame(X_reshaped)
y_pd = pd.DataFrame(y)

X_train, X_test, y_train, y_test = train_test_split(X_pd,
                                                    y_pd,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
print("[DONE] - train_test_split")

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1)

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(X_train)
print("[DONE] -StandardScaler")

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# A parameter grid for XGBoost
params = {'learning_rate': [0.005, 0.05, 0.5, 1.0],
          'n_estimators': [10, 50, 100, 500],
          'max_depth': [1, 3, 5, 10],
          'min_child_weight': [1, 3, 5, 10],
          'gamma': [0, 0.1, 0.5, 1.0],
          'subsample': [0.8],
          'colsample_bytree': [0.8],
          'objective': ['multi:softmax', 'binary:logistic'],
          'nthread': [1, 3, 5, 10],
          'scale_pos_weight': [1],
          'seed': [42]
          }



xgb = XGBClassifier(silent=False)
print("[DONE] - XGBClassifier")

folds = 5 # 5이상 높여서 실험
param_comb = 1 # 5이상 높여서 실험

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
print("[DONE] - StratifiedKFold")

random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='f1',
                                   n_jobs=1,
                                   cv=skf.split(X_train,y_train),
                                   verbose=1,
                                   random_state=1001)


print("[DONE] - RandomizedSearchCV")

# Here we go
print("[START] - random_search fitting")
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train,y_train, verbose=10)
timer(start_time) # timing ends here for "start_time" variable
print("[DONE] - random_search fitting")

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
date = str(time.time())
results.to_csv('./evaluate_result/eXtream_Gradient_Boosting/' + date + '-' + filename + '.csv', index=False)
print("[INFO]andom_search.predict - ", random_search.predict(X_test))
print("[INFO]random_search.score - ", random_search.score(X_test, y_test))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
print("[INFO]means - ", means)
print("[INFO]stds - ", stds)
print("[INFO]params - ", params)