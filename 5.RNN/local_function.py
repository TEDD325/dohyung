# import library
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import SimpleRNN, LSTM, GRU
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import metrics
from keras import losses
from keras import __version__
print('Using Keras version:', __version__, 'backend:', K.backend())
from keras.backend.tensorflow_backend import set_session

from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report # https://stackoverflow.com/questions/50065484/getting-precision-recall-and-f1-score-per-class-in-keras

from IPython.display import Javascript
import numpy as np
from distutils.version import LooseVersion as LV
from datetime import datetime
import csv
import sys
from os import listdir
from os.path import isfile, join
import pprint
import boto3
import pickle
import time
import os.path
import pickle
sys.path.append(os.getcwd())
from matplotlib import pyplot

import smtplib
from email.mime.text import MIMEText # simple MSG
from email.mime.multipart import MIMEMultipart # complex MSG
        
from link_aws_key import *
pp = pprint.PrettyPrinter(indent=4)

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


coins = {
    0: 'KRW',
    1: 'BTC',
    2: 'ETH',
    3: 'XRP',
    4: 'BCH',
    5: 'LTC',
    6: 'DASH',
    7: 'ETC'
}

# aws_client = boto3.client(
#     's3',
#     aws_access_key_id=LINK_AWSAccessKeyId,
#     aws_secret_access_key=LINK_AWSSecretKey
# )

bucket = "bithumb10"
cleanup_file_name = "coin_{0}_{1}_cleanup.csv"


#######################################################

def Load_Dataset_X(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_X = "X_" + \
                    str(time_unit) + "_" + \
                    str(window_size) + "_" + \
                    str(gap) + "_" + \
                    str(margin_rate)
                    

    with open(dir_path + key_name_X + ".pickle", 'rb') as handle:
        b_x = pickle.load(handle)
    return b_x

def Load_Dataset_y(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_y = "y_" + \
                    str(time_unit) + "_" + \
                    str(window_size) + "_" + \
                    str(gap) + "_" + \
                    str(margin_rate)
                    

    with open(dir_path + key_name_y + ".pickle", 'rb') as handle:
        b_y = pickle.load(handle)
    return b_y

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def input_reshape(X_train_data, X_test_data, n_steps, n_coins, n_price):
    X_train_reshape = X_train_data.reshape(
        -1, 
        n_steps, 
        n_coins * n_price
    )
    X_test_reshape = X_test_data.reshape(
        -1, 
        n_steps, 
        n_coins * n_price
    )
    return X_train_reshape, X_test_reshape

def onehottify(x, n=None, dtype=np.int):
    """1-hot encode x with the max value n (computed from data if n is None)."""
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))