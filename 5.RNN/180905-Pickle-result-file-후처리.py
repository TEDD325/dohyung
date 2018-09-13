
# coding: utf-8

# In[1]:


# import library
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import SimpleRNN, LSTM, GRU
from keras import layers, models
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import metrics
from keras import losses
from keras import __version__
print('Using Keras version:', __version__, 'backend:', K.backend())
# assert(LV(__version__) >= LV("2.0.0"))
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)
# cfg = K.tf.ConfigProto()
# cfg.gpu_options.allow_growth = True
# K.set_session(K.tf.Session(config=cfg))

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 

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
#import boto3
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


# In[2]:


def Load_Dataset_X(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_X = "X_" +                     str(time_unit) + "_" +                     str(window_size) + "_" +                     str(gap) + "_" +                     str(margin_rate)
                    

    with open(dir_path + key_name_X + ".pickle", 'rb') as handle:
        b_x = pickle.load(handle)
    return b_x
    
def Load_Dataset_y(dir_path, time_unit, window_size, gap, margin_rate):
    key_name_y = "y_" +                     str(time_unit) + "_" +                     str(window_size) + "_" +                     str(gap) + "_" +                     str(margin_rate)
                    

    with open(dir_path + key_name_y + ".pickle", 'rb') as handle:
        b_y = pickle.load(handle)
    return b_y


# In[3]:


def save_to_Pickle_file(dir_path, dictionary):
    pickle_result_dir_path = "./evaluate_result/"
    new_file_name = str(dictionary.keys())[12:-3] + '_result.pickle'
    with open(pickle_result_dir_path +               new_file_name, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[24]:


# load pickle file 
def Load_result_pickle_file(dir_path, MODEL, coin, time_unit, window_size, gap, margin_rate):
    filename =  MODEL + "_" +                 coin + "_" +                 str(time_unit) + "_" +                 str(window_size) + "_" +                 str(gap) + "_" +                 str(margin_rate) + "_" +                 "result.pickle"

        
    b_x = pickle.load(open("./evaluate_result/"+filename, "rb"))

    # _key = str(b_x.keys())[12:-3]
    # b_x[_key].keys()
    b_x[str(b_x.keys())[12:-3]].keys()
    # 'MODEL'
    # 'grid_result.cv_results_['mean_test_score']'
    # 'grid_result.cv_results_['params']", 'grid_result.test_score']'

    dict_keys = tuple(b_x[str(b_x.keys())[12:-3]].keys())

    # key name 검사 및 수정
    if dict_keys[0] == 'MODEL: ':
        b_x[str(b_x.keys())[12:-3]]['MODEL'] = b_x[str(b_x.keys())[12:-3]]['MODEL: ']
        del b_x[str(b_x.keys())[12:-3]]['MODEL: ']
        b_x[str(b_x.keys())[12:-3]]['Cryptocurrency'] = b_x[str(b_x.keys())[12:-3]]['Cryptocurrency: ']
        del b_x[str(b_x.keys())[12:-3]]['Cryptocurrency: ']
        b_x[str(b_x.keys())[12:-3]]['Score'] = b_x[str(b_x.keys())[12:-3]]["grid_result.cv_results_['mean_test_score']"]
        del b_x[str(b_x.keys())[12:-3]]["grid_result.cv_results_['mean_test_score']"]
        b_x[str(b_x.keys())[12:-3]]['Params'] = b_x[str(b_x.keys())[12:-3]]["grid_result.cv_results_['params']"]
        del b_x[str(b_x.keys())[12:-3]]["grid_result.cv_results_['params']"]
        b_x[str(b_x.keys())[12:-3]]['test_score'] = b_x[str(b_x.keys())[12:-3]]['grid_result.test_score']
        del b_x[str(b_x.keys())[12:-3]]['grid_result.test_score']

#         pickle_result_dir_path = "./evaluate_result/"
#         save_to_Pickle_file(pickle_result_dir_path, b_x)
#         evaluate_result = pickle.load(open(dir_path + "SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
    return b_x


# In[25]:


# key = str(evaluate_result.keys())[12:-3]
# #evaluate_result[key]['Score']


# In[26]:


dir_path = './evaluate_result/'
MODEL = "SimpleRNN"
coin = "BTC"
time_unit = 10
window_size = 10
gap = 1
margin_rate = 0.1
evaluate_result = Load_result_pickle_file(dir_path, MODEL, coin, time_unit, window_size, gap, margin_rate)


# In[27]:


evaluate_result


# In[6]:


def Find_Best_Params(dictionary):
    key = str(dictionary.keys())[12:-3]
    Max_Score = max(dictionary[key]['Score'])
    for idx, item in enumerate(dictionary[key]['Score']):
        if item == Max_Score:
            Max_Score_Idx = idx
            Best_Params = dictionary[key]['Params'][idx]
            dictionary[key]['train_score'] = Max_Score
            dictionary[key]['Best_Params'] = Best_Params
            
            pickle_result_dir_path = "./evaluate_result/"
            save_to_Pickle_file(pickle_result_dir_path, dictionary)
        
            break
    
    
    return dictionary


# In[7]:


# tmp = Find_Best_Params(evaluate_result)
# tmp


# In[8]:


# evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
# evaluate_result


# In[30]:


def Value_to_List(evaluate_result):
    for i in evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']:
    #     print(evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params'][i])
        if type(evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params'][i]) != list:
            evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params'][i] =                     [evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params'][i]]
    print(evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params'])
    
#     pickle_result_dir_path = "./evaluate_result/"
#     save_to_Pickle_file(pickle_result_dir_path, evaluate_result)


# In[31]:


Value_to_List(evaluate_result)


# In[35]:


# evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'] = \
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'][0] = ['relu']


# In[36]:


pickle_result_dir_path = "./evaluate_result/"
save_to_Pickle_file(pickle_result_dir_path, evaluate_result)
evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
evaluate_result


# In[38]:


evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'] =     evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'][0] 


# In[39]:


pickle_result_dir_path = "./evaluate_result/"
save_to_Pickle_file(pickle_result_dir_path, evaluate_result)
evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
evaluate_result


# In[32]:


evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
evaluate_result


# In[28]:


evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['activation'][0][0]
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['batch_size'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['batch_size'][0][0]
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['init'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['init'][0][0]
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['n_state_units'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['n_state_units'][0][0]
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['optimizer'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['optimizer'][0][0]
evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['window_size'] =         evaluate_result[str(evaluate_result.keys())[12:-3]]['Best_Params']['window_size'][0][0]


# In[16]:


evaluate_result


# In[17]:


pickle_result_dir_path = "./evaluate_result/"
save_to_Pickle_file(pickle_result_dir_path, evaluate_result)


# In[18]:


pickle_result_dir_path = "./evaluate_result/"
save_to_Pickle_file(pickle_result_dir_path, evaluate_result)
evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
evaluate_result

