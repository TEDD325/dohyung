# https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost/notebook

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint
import scipy.stats as stats
from random import *
from sklearn.model_selection import train_test_split
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

data = pd.read_pickle('./data/RNN_coin/X_10_10_1_0.1.pickle')
X = np.array(data[0])
y = np.array(data[1])
X_reshaped = X.reshape(len(y) * 8, -1)
y_reshaped = y.reshape(-1)
X_pd = pd.DataFrame(X_reshaped)
y_pd = pd.DataFrame(y_reshaped)

X_train, X_test, y_train, y_test = train_test_split(X_pd,
                                                    y_pd,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    shuffle=True)

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# A parameter grid for XGBoost
params = {
        # 'min_child_weight': [2], # 2
        #  'gamma': [1.1], # 1.1
        # 'subsample': [0.1], # 0.1
        # 'colsample_bytree': [1.0], # 1.0  ,. 0 <= colsample <= 1
        # 'max_depth': [7], # 7
        # 'learning_rate': [0.21], # 0.21
        # 'n_estimators' : [5], # 5
        # 'objective' :['binary:hinge'], # 'binary:hinge'
        # 'nthread': [1, 2, 3] # 1
        'min_child_weight': [2, 3],
         'gamma': [1.1, 1.2, 1.3, 1.4, 1.5],
        'subsample': [0.3, 0.4, 0.5],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'max_depth': [6, 7, 8],
        'learning_rate': [0.17, 0.19, 0.21],
        'n_estimators' : [4, 5, 6],
        'objective' :['binary:hinge'],
        'nthread': [1, 2, 3]
}

xgb = XGBClassifier(silent=False)

folds = 5 # 5이상 높여서 실험
param_comb = 1 # 5이상 높여서 실험

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='f1',
                                   n_jobs=1,
                                   cv=skf.split(X_train,y_train),
                                   verbose=1,
                                   random_state=1001)

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train,y_train)
timer(start_time) # timing ends here for "start_time" variable

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
results.to_csv('./XGBoost_result/' + date + '.csv', index=False)

