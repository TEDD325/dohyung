
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


def get_all_raw_data_from_aws(coin_name_list, start_date, end_date):
    start_ms_time = datetime.strptime(start_date + " +0900", "%Y-%m-%d %H:%M:%S %z").timestamp() * 1000
    end_ms_time = datetime.strptime(end_date + " +0900", "%Y-%m-%d %H:%M:%S %z").timestamp() * 1000
    
    year_temp = start_date[:4]
    years = [year_temp]
    while year_temp < end_date[:4]:
        year_temp = str(int(start_date[:4]) + 1)
        years.append(year_temp)
    raw_data = {}  # 전체 CSV Raw 데이터
    for coin_name in coin_name_list:
        raw_data[coin_name] = []

    # KRW 제외한 나머지 CSV Raw 데이터 수집
    for coin_name in coin_name_list:
        if coin_name == 'KRW':
            continue
        lines = []
        for year in years:
            obj = aws_client.get_object(
                Bucket=bucket,
                Key='cleanup/' + year + '/' + cleanup_file_name.format(coin_name, year)
            )
            if lines != []:
                lines += obj.get('Body')._raw_stream.readlines()
            else:
                lines = obj.get('Body')._raw_stream.readlines()

        for line in lines:
            line = str(line.strip())[2:-1]
            line = line.split(',')
            if start_ms_time <= int(line[0]) and int(line[0]) <= end_ms_time:
                raw_data[coin_name].append(line)

    raw_data['KRW'] = list()
    for line in raw_data['BTC']:
        raw_data['KRW'].append([line[0], line[1], 1, 1, 1, 1, 1.0, 'normal'])

    return raw_data

def get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir):
    trading_files = []
    for coin_name in coin_name_list:
        for data_file_name in [f for f in listdir(data_files_dir) if isfile(join(data_files_dir, f))]:
            if coin_name in data_file_name:
                trading_files.append(data_file_name)

    start_ms_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    end_ms_time = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    
    raw_data = {} #전체 CSV Raw 데이터
    for coin_name in coin_name_list:
        raw_data[coin_name] = []
    
    #KRW 제외한 나머지 CSV Raw 데이터 수집
    for coin_name in coin_name_list:
        for data_file_name in trading_files:
            if coin_name in data_file_name:
                file = open(data_files_dir + data_file_name, 'r', encoding='utf-8')
                rdr = csv.reader(file)
                for line in rdr:
                    if start_ms_time <= int(line[0]) and int(line[0]) <= end_ms_time:
                        raw_data[coin_name].append(line)
                file.close()
    
    for line in raw_data['BTC']:
        raw_data['KRW'].append([line[0], line[1], 1, 1, 1, 1, 1.0, 'normal'])
#     print("test")
    return raw_data

def Make_Dataset(dir_path, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate):
    print("Make_Dataset is Started.")
    a = time.time()

    X = {}
    y = {}
    idx = []

    key_name_X = "X_"
    key_name_y = "y_"
    for idx_time_unit in time_unit:
        for idx_window_size in window_size:
            for idx_gap in gap:
                for idx_margin_rate in margin_rate:
                    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    if (os.path.isfile(dir_path+key_name_X+".pickle")) is not True:
                        print(key_name_X)
                        X[key_name_X] =                         make_cryptocurrency_dataset(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margin_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_X+".pickle", 'wb') as handle:
                            pickle.dump(X[key_name_X], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        X = {}
                        print("X_success.")
                    if (os.path.isfile(dir_path + key_name_y+".pickle")) is not True:
                        print(key_name_y)
                        y[key_name_y] =                         make_cryptocurrency_dataset(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margin_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_y+".pickle", 'wb') as handle:
                            pickle.dump(y[key_name_y], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        y = {}
                        print("y_success.")
                    key_name_X = "X_"
                    key_name_y = "y_"
    b = time.time()
    print("Make_Dataset is Done.")
    #print("time: ", b-a)

def make_cryptocurrency_dataset(coin_name_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    y_trv = []
    y_btv = []
    num_coins = len(coin_name_list)
    #raw_data = get_all_raw_data_from_aws(coin_name_list, start_date, end_date)
    raw_data = get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir)    
    num_sticks = len(raw_data['BTC'])
    
    if time_unit % 10 != 0 or num_sticks < (time_unit / 10) * window_size + gap:
        return None, None
    
    num = int(num_sticks - ((time_unit / 10) * window_size + gap) + 1)
        
    X = []
    y = []
    # (윈도우 개수, 코인 개수, 윈도우 사이즈, 3)
    for idx in range(num):
        X.append([])
        y.append([])
        idx_coin = 0
        for coin_name in coin_name_list:
            X[idx].append([])
            
            last_idx_in_window = int(idx + time_unit / 10 * window_size - 1)
            close_price_in_last_idx_in_window = float(raw_data[coin_name][last_idx_in_window][3])
            
            for idx_in_window in range(window_size):
                X[idx][idx_coin].append([])
                idx_stick = int(idx + time_unit / 10 * (idx_in_window + 1) - 1)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][3]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][4]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][5]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][6]))
                
            target_idx_for_window = int(idx + time_unit / 10 * window_size - 1 + gap)
            target_price = float(raw_data[coin_name][target_idx_for_window][3])
            
            target = 0
            if target_price >= close_price_in_last_idx_in_window * (1.0 + float(margin_rate) / 100.0):
                target = 1
            y[idx].append(target)
            
            idx_coin += 1
           
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y
    
def make_cryptocurrency_dataset_X(coin_name_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    num_coins = len(coin_name_list)
    #raw_data = get_all_raw_data_from_aws(coin_name_list, start_date, end_date)
    raw_data = get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir)    
    num_sticks = len(raw_data['BTC'])
    
    if time_unit % 10 != 0 or num_sticks < (time_unit / 10) * window_size + gap:
        return None, None
    
    num = int(num_sticks - ((time_unit / 10) * window_size + gap) + 1)
        
    X = []
    # (윈도우 개수, 코인 개수, 윈도우 사이즈, 3)
    for idx in range(num):
        X.append([])
        idx_coin = 0
        for coin_name in coin_name_list:
            X[idx].append([])
            
            last_idx_in_window = int(idx + time_unit / 10 * window_size - 1)
            close_price_in_last_idx_in_window = float(raw_data[coin_name][last_idx_in_window][3])
            
            for idx_in_window in range(window_size):
                X[idx][idx_coin].append([])
                idx_stick = int(idx + time_unit / 10 * (idx_in_window + 1) - 1)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][3]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][4]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][5]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][6]))
                
    X = np.array(X)
    
    return X

def make_cryptocurrency_dataset_y(coin_name_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    y_trv = []
    y_btv = []
    num_coins = len(coin_name_list)
    #raw_data = get_all_raw_data_from_aws(coin_name_list, start_date, end_date)
    raw_data = get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir)    
    num_sticks = len(raw_data['BTC'])
    
    if time_unit % 10 != 0 or num_sticks < (time_unit / 10) * window_size + gap:
        return None, None
    
    num = int(num_sticks - ((time_unit / 10) * window_size + gap) + 1)
        
    
    y = []
    # (윈도우 개수, 코인 개수, 윈도우 사이즈, 3)
    for idx in range(num):
        
        y.append([])
        idx_coin = 0
        for coin_name in coin_name_list:            
            last_idx_in_window = int(idx + time_unit / 10 * window_size - 1)
            close_price_in_last_idx_in_window = float(raw_data[coin_name][last_idx_in_window][3])
            target_idx_for_window = int(idx + time_unit / 10 * window_size - 1 + gap)
            target_price = float(raw_data[coin_name][target_idx_for_window][3])
            target = 0
            
            if target_price >= close_price_in_last_idx_in_window * (1.0 + float(margin_rate) / 100.0):
                target = 1
            y[idx].append(target)
            
            idx_coin += 1
           
    y = np.asarray(y)
    
    return y

#

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
    
def Make_Dataset_numpy(dir_path, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    print("Make_Dataset_numpy is Started.")
    a = time.time()

    X = {}
    y = {}
    idx = []

    key_name_X = "X_"
    key_name_y = "y_"
    for idx_time_unit in time_unit:
        for idx_window_size in window_size:
            for idx_gap in gap:
                for idx_margin_rate in margin_rate:
                    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    if (os.path.isfile(dir_path + key_name_X+".pickle")) is not True:
                        print(key_name_X)
                        X[key_name_X] =                         make_cryptocurrency_dataset_X(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margin_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_X+".pickle", 'wb') as handle:
                            pickle.dump(X[key_name_X], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        X = {}
                        print("X_success.")
                    if (os.path.isfile(dir_path+key_name_y+".pickle")) is not True:
                        print(key_name_y)
                        y[key_name_y] =                         make_cryptocurrency_dataset_y(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margin_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_y+".pickle", 'wb') as handle:
                            pickle.dump(y[key_name_y], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        y = {}
                        print("y_success.")
                    key_name_X = "X_"
                    key_name_y = "y_"
    b = time.time()
    print("Make_Dataset_numpy is Done.")
    print()
    #print("time: ", b-a)
    
    
def Make_Dataset_tuple(dir_path, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    print("Make_Dataset_tuple is Started.")
    a = time.time()

    X = {}
    y = {}
    idx = []

    key_name_X = "X_"
    key_name_y = "y_"
    for idx_time_unit in time_unit:
        for idx_window_size in window_size:
            for idx_gap in gap:
                for idx_margin_rate in margin_rate:
                    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                    if (os.path.isfile(dir_path + key_name_X+".pickle")) is not True:
                        print(key_name_X)
                        X[key_name_X], _ =                         make_cryptocurrency_dataset(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margix_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_X+".pickle", 'wb') as handle:
                            pickle.dump(X[key_name_X], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        X = {}
                        print("X_success.")
                    if (os.path.isfile(dir_path+key_name_y+".pickle")) is not True:
                        print(key_name_y)
                        _, y[key_name_y] =                         make_cryptocurrency_dataset(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margix_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_y+".pickle", 'wb') as handle:
                            pickle.dump(y[key_name_y], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        y = {}
                        print("y_success.")
                    key_name_X = "X_"
                    key_name_y = "y_"
    b = time.time()
    print("Make_Dataset_tuple is Done.")
    print()
    #print("time: ", b-a)

def Check_Dataset(dir_path, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate, data_files_dir):
    print("Check_Dataset is Started.")
    a = time.time()

    X = {}
    y = {}
    idx = []

    key_name_X = "X_"
    key_name_y = "y_"
    for idx_time_unit in time_unit:
        for idx_window_size in window_size:
            for idx_gap in gap:
                for idx_margix_rate in margin_rate:
                    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margix_rate)
                    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margix_rate)
                    
                    with open(dir_path + key_name_X+".pickle", 'rb') as handle:
                        data = pickle.load(handle)
    
                    if type(data) == tuple:
#                         print()
                        os.system('rm '+dir_path + key_name_X+".pickle")
                        print(key_name_X,".pickle is removed.")
                        os.system('rm '+dir_path + key_name_y+".pickle")
                        print(key_name_y,".pickle is removed.")
        
                    if (os.path.isfile(dir_path+key_name_X+".pickle")) is not True:
                        print(key_name_X)
                        X[key_name_X] =                         make_cryptocurrency_dataset_X(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margix_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_X+".pickle", 'wb') as handle:
                            pickle.dump(X[key_name_X], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        X = {}
                        print("[SUCCESS] ",key_name_X,".pickle is created.")
                    if (os.path.isfile(dir_path+key_name_y+".pickle")) is not True:
                        print(key_name_y)
                        y[key_name_y] =                         make_cryptocurrency_dataset_y(
                                                    coin_list, 
                                                    start_date, 
                                                    end_date,
                                                    idx_time_unit,
                                                    idx_window_size,
                                                    idx_gap,
                                                    idx_margix_rate,
                                                    data_files_dir)
                        with open(dir_path + key_name_y+".pickle", 'wb') as handle:
                            pickle.dump(y[key_name_y], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        y = {}
                        print("[SUCCESS] ",key_name_y,".pickle is created.")
                    key_name_X = "X_"
                    key_name_y = "y_"
    b = time.time()
    print("Check_Dataset is Done.")
    #print("time: ", b-a)


# In[3]:


# Create Dataset
data_files_dir = "/Users/dohyung/OneDrive/2018-RNN/RNN_python/AWS_dataset/"
dataset_dir_path_tuple_type = "./dataset_pickle_tuple_type/"
dataset_dir_path_numpy_type = "./dataset_pickle_numpy.ndarray_type)/"
coin_list = ["KRW", "BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
start_date = "2017-08-04 21:40:00"
end_date = "2018-08-20 23:50:00"
# time_unit = [10,30,60]     # candle stick minutes
# window_size = [10,25,50,75,100]  # Unit: num. of candle sticks
# gap = [1,2,3]            # Unit: num. of candle sticks
# margin_rate = [0.1,0.25,0.5]  # Unit: percent

# for slave04
#time_unit = [10,30,60]     # candle stick minutes
#window_size = [10,25,50]  # Unit: num. of candle sticks
#gap = [1,2,3]            # Unit: num. of candle sticks
#margin_rate = [0.1,0.25,0.5]  # Unit: percent

# for slave05
# time_unit = [10,30,60]     # candle stick minutes
# window_size = [75,100]  # Unit: num. of candle sticks
# gap = [1,2,3]            # Unit: num. of candle sticks
# margin_rate = [0.1,0.25,0.5]  # Unit: percent


# In[4]:


#Make_Dataset_tuple(dataset_dir_path_tuple_type, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate)
#Make_Dataset_numpy(dataset_dir_path_numpy_type, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate)
#Check_Dataset(dataset_dir_path, coin_list, start_date, end_date, time_unit, window_size, gap, margin_rate)
# Tuple 형태의 데이터셋이 나오지 않도록.


# In[5]:


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[6]:


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

def f1_score_(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / (c2 + 1e-7)
    
    # How many relevant items are selected?
    recall = c1 / (c3 +  + 1e-7)

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-7)
    return f1_score 


def create_model_RNN(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        RNN(  n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


def create_model_SimpleRNN(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        SimpleRNN(n_state_units, 
                  input_shape=(window_size, 32),
                  use_bias=True, 
                  activation='tanh',
                  kernel_initializer='glorot_uniform', 
                  recurrent_initializer='orthogonal', 
                  bias_initializer='zeros', 
                  dropout=0.0,
                  recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])
        
    return model

def create_model_LSTM(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        LSTM( n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])
    
    return model


def create_model_GRU(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        GRU(  n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
    model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


def create_model_RNN_non_GPU(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        RNN(  n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


def create_model_SimpleRNN_non_GPU(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        SimpleRNN(n_state_units, 
                  input_shape=(window_size, 32),
                  use_bias=True, 
                  activation='tanh',
                  kernel_initializer='glorot_uniform', 
                  recurrent_initializer='orthogonal', 
                  bias_initializer='zeros', 
                  dropout=0.0,
                  recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model

def create_model_LSTM_non_GPU(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        LSTM( n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


def create_model_GRU_non_GPU(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        GRU(  n_state_units, 
              input_shape=(window_size, 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model



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

def Start_Model(pickle_load_dir_path, data_files_dir, epochs, pickle_result_dir_path, boost_up_result_dir_path, MODEL, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate, _TEST, _ENHANCE, _GPU, n_jobs, machine, Internet_connection, params):
    X = {}
    y = {}
    #if (_TEST == True): 
        #MODEL_list = ["SimpleRNN", "LSTM", "GRU"]
    #    time_unit = [10]     # candle stick minutes
    #    window_size = [10]  # Unit: num. of candle sticks
    #    gap = [1]            # Unit: num. of candle sticks
    #    margin_rate = [0.1]  # Unit: percent
    #elif (_TEST == False):
        #MODEL_list = ["SimpleRNN", "LSTM", "GRU"]

    key_name_X = "X_"
    key_name_y = "y_"

    #for MODEL_idx in MODEL_list:
        #MODEL = MODEL_idx
        #for idx_time_unit in time_unit:
                #for idx_window_size in window_size:
                    #for idx_gap in gap:
                        #for idx_margin_rate in margin_rate:
    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)

    X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)
    y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)

    y_single = {}
    y_single['BTC'] = y[:, 1]
    y_single['ETH'] = y[:, 2]
    y_single['XRP'] = y[:, 3]
    y_single['BCH'] = y[:, 4]
    y_single['LTC'] = y[:, 5]
    y_single['DASH'] = y[:, 6]
    y_single['ETC'] = y[:, 7]

    coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

#                         for coin in coin_list2:
#                             print("y_single["+coin+"]"+".shape")
#                             print(y_single[coin].shape)
#                             print()


    if (_TEST == False and _ENHANCE == False):
        for coin in coin_list2:
            if (os.path.isfile(pickle_result_dir_path +                               MODEL + "_" +                               coin + "_" +                               str(idx_time_unit) + "_" +                               str(idx_window_size) + "_" +                               str(idx_gap) + "_" +                               str(idx_margin_rate) +                               "_result.pickle")) is True:
                print(MODEL + "_" +                       coin + "_" +                       str(idx_time_unit) + "_" +                       str(idx_window_size) + "_" +                       str(idx_gap) + "_" +                       str(idx_margin_rate) +                       "_result.pickle FILE ALREADY EXIST.")
                continue
            else:
                y2 = onehottify(y_single[coin], n=2)

                Evaluate(pickle_load_dir_path, 
                         data_files_dir, 
                         epochs, 
                         pickle_result_dir_path,
                         _TEST, 
                         _ENHANCE,
                         coin,
                         X, y2,
                         key_name_X,
                         key_name_y,
                         idx_time_unit,
                         idx_window_size,
                         idx_gap,
                         idx_margin_rate, 
                         MODEL,
                         _GPU,
                         n_jobs,
                         machine,
                         Internet_connection,
                         params)
            #Javascript('IPython.notebook.kernel.restart()')
            #time.sleep(1)
            #Javascript('IPython.notebook.execute_all_cells()')


    if (_TEST == True and _ENHANCE == False):
        # for test                                
        for coin in range(1):
            if (os.path.isfile(pickle_result_dir_path +                               "_test_" +                               MODEL + "_" +                               "BTC" + "_" +                               str(idx_time_unit) + "_" +                               str(idx_window_size) + "_" +                               str(idx_gap) + "_" +                               str(idx_margin_rate) +                               "_result.pickle")) is True:
                print("_test_" +                       MODEL + "_" +                       "BTC" + "_" +                       str(idx_time_unit) + "_" +                       str(idx_window_size) + "_" +                       str(idx_gap) + "_" +                       str(idx_margin_rate) +                       "_result.pickle FILE ALREADY EXIST.")
                #Javascript('IPython.notebook.kernel.restart()')
                #Javascript('IPython.notebook.execute_all_cells()')
                continue
            else:
                y2 = onehottify(y_single['BTC'], n=2)                          

                Evaluate(pickle_load_dir_path, 
                         data_files_dir, 
                         epochs, 
                         pickle_result_dir_path, 
                         _TEST, 
                         _ENHANCE,
                         coin,
                         X, y2,
                         key_name_X,
                         key_name_y,
                         idx_time_unit,
                         idx_window_size,
                         idx_gap,
                         idx_margin_rate, 
                         MODEL,
                         _GPU,
                         n_jobs, 
                         machine,
                         Internet_connection,
                         params)
    #Javascript('IPython.notebook.kernel.restart()')
    #time.sleep(1)
    #Javascript('IPython.notebook.execute_all_cells()')
                #Javascript('IPython.notebook.kernel.restart()')
                #Javascript('IPython.notebook.execute_all_cells()')
    if (_TEST == False and _ENHANCE == True):
        for coin in coin_list2:
            if (os.path.isfile(boost_up_result_dir_path +                               MODEL + "_" +                               coin + "_" +                               str(idx_time_unit) + "_" +                               str(idx_window_size) + "_" +                               str(idx_gap) + "_" +                               str(idx_margin_rate) +                               "_boost_up_result.pickle")) is True:
                print(MODEL + "_" +                       coin + "_" +                       str(idx_time_unit) + "_" +                       str(idx_window_size) + "_" +                       str(idx_gap) + "_" +                       str(idx_margin_rate) +                       "_boost_up_result.pickle FILE ALREADY EXIST.")
                continue
            else:
                print("[BOOST UP THE "+MODEL + "_" +                                         coin + "_" +                                         str(idx_time_unit) + "_" +                                         str(idx_window_size) + "_" +                                         str(idx_gap) + "_" +                                         str(idx_margin_rate) + " STARTED.]")
                y2 = onehottify(y_single[coin], n=2)

                Evaluate(pickle_load_dir_path, 
                         data_files_dir, 
                         epochs, 
                         boost_up_result_dir_path,
                         _TEST, 
                         _ENHANCE,
                         coin,
                         X, y2,
                         key_name_X,
                         key_name_y,
                         idx_time_unit,
                         idx_window_size,
                         idx_gap,
                         idx_margin_rate, 
                         MODEL,
                         _GPU,
                         n_jobs,
                         machine,
                         Internet_connection,
                         params)
            #Javascript('IPython.notebook.kernel.restart()')
            #time.sleep(1)
            #Javascript('IPython.notebook.execute_all_cells()')

# 저장된 pickle 파일의 데이터 구조
# tmp = {}
# tmp["10_1_1_0.1"] = {"grid_result.best_score_":{}}, {"grid_result.best_params_":{}}
# type(tmp["10_1_1_0.1"][0])
# print(tmp["10_1_1_0.1"])
# print(tmp["10_1_1_0.1"])
# print(tmp["10_1_1_0.1"][0])
# print(tmp["10_1_1_0.1"][0]['grid_result.best_score_'])


# In[7]:


def Evaluate(pickle_load_dir_path, 
             data_files_dir, 
             epochs, 
             pickle_result_dir_path,
             _TEST, 
             _ENHANCE, 
             coin, 
             X, y2, 
             key_name_X,
             key_name_y,
             idx_time_unit,
             idx_window_size,
             idx_gap,
             idx_margin_rate, 
             MODEL,
             _GPU,
             n_jobs, 
             machine,
             Internet_connection,
             parameter):
    

    
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.1, random_state=42)
#     print("X_train.shape")
#     print(X_train.shape)
#     print("y_train.shape")
#     print(y_train.shape)
#     print()
#     print("X_test.shape")
#     print(X_test.shape)
#     print("y_test.shape")
#     print(y_test.shape)
#     print()

    n_coins = 8
    n_price = 4
    n_steps = idx_window_size # 원래 100이었음. reshape 문제 때문에 수정함

    X_train_2 = X_train.transpose([0, 2, 1, 3])
    X_test_2 = X_test.transpose([0, 2, 1, 3])
#     print("X_train_2.shape")
#     print(X_train_2.shape)
#     print("X_test_2.shape")
#     print(X_test_2.shape)
#     print()

    X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
    X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
#     print("X_train_3.shape")
#     print(X_train_3.shape)
#     print("X_test_3.shape")
#     print(X_test_3.shape)
#     print()

    if (_TEST==True and _ENHANCE==False):
        param_grid = {'window_size' : [n_steps], 
                      'n_state_units': [100],
                      'activation': ['relu'], 
                      'optimizer': ['rmsprop'], #sgd 추가
                      'init': ['glorot_uniform'], #he 추가
                      'batch_size': [2048]}

    elif (_TEST==False and _ENHANCE==False):
        param_grid = {'window_size' : [n_steps], 
                      'n_state_units': [40, 80, 160],
                      'activation': ['relu', 'softmax'], 
                      'optimizer': ['rmsprop', 'adam'], #sgd 추가
                      'init': ['glorot_uniform', 'uniform', 'he_uniform'], #he 추가
                      'batch_size': [64,128,256]}
        
    elif (_TEST==False and _ENHANCE==True):
        param_grid = parameter
        
        



    X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps*n_coins * n_price])
    X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps*n_coins * n_price])
#     print("X_train_reshape.shape")
#     print(X_train_reshape.shape)
#     print("X_test_reshape.shape")
#     print(X_test_reshape.shape)
#     print()

    scaler = MinMaxScaler()
    scaler.fit(X_train_reshape)
    X_train_scaled = scaler.transform(X_train_reshape)
    X_test_scaled = scaler.transform(X_test_reshape)

    X_train_scaled = X_train_scaled.reshape(-1, n_steps, n_coins * n_price)

    X_test_scaled = X_test_scaled.reshape(-1, n_steps, n_coins * n_price)

    if _GPU == True:
        if MODEL == "SimpleRNN" :
            model = KerasClassifier(build_fn=create_model_SimpleRNN, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
#                                     batch_size=100, 
                                    verbose=True)

        elif MODEL == "LSTM":
            model = KerasClassifier(build_fn=create_model_LSTM, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
#                                     batch_size=100, 
                                    verbose=True)

        elif MODEL == "RNN":
            model = KerasClassifier(build_fn=create_model_RNN, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
#                                     batch_size=100, 
                                    verbose=True)

        elif MODEL == "GRU":
            model = KerasClassifier(build_fn=create_model_GRU, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
#                                     batch_size=100, 
                                    verbose=True)
    elif _GPU == False:
        if MODEL == "SimpleRNN" :
            model = KerasClassifier(build_fn=create_model_SimpleRNN_non_GPU, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
                                    batch_size=10, 
                                    verbose=True)

        elif MODEL == "LSTM":
            model = KerasClassifier(build_fn=create_model_LSTM_non_GPU, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
                                    batch_size=10, 
                                    verbose=True)

        elif MODEL == "RNN":
            model = KerasClassifier(build_fn=create_model_RNN_non_GPU, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
                                    batch_size=10, 
                                    verbose=True)

        elif MODEL == "GRU":
            model = KerasClassifier(build_fn=create_model_GRU_non_GPU, 
                                    epochs=epochs, # epochs는 실험을 최종적으로 수행하고자 할 때 높일 것(100~150정도)
                                    batch_size=10, 
                                    verbose=True)
    if (_ENHANCE == False):
        grid = GridSearchCV(
            estimator=model, 
            cv=5, 
            n_jobs=n_jobs, # test
            param_grid=param_grid,
            verbose=1)
      
        X_train_scaled, X_test_scaled = input_reshape(X_train_scaled, X_test_scaled, n_steps, n_coins, n_price)

        if (_TEST == True): 
            print()
            print()
            print("TEST!")
            print()
            print("----------------------")
            print("<"+MODEL+">")
            print("----------------------")
            print("__"+"BTC"+"__" +                     "time unit: "+str(idx_time_unit) + "  |  " +                     "window_size :"+str(idx_window_size) + "  |  " +                     "gap :"+str(idx_gap) + "  |  " +                     "margin_rate :"+str(idx_margin_rate) +                     "  started.")
        elif (_TEST == False):
            print()
            print()
            print("----------------------")
            print("<"+MODEL+">")
            print("----------------------")
            print("__"+coin+"__" +                     "time unit: "+str(idx_time_unit) + "  |  " +                     "window_size :"+str(idx_window_size) + "  |  " +                     "gap :"+str(idx_gap) + "  |  " +                     "margin_rate :"+str(idx_margin_rate) +                     "  started.")

        grid_result = grid.fit(X_train_scaled, 
                               y_train, 
                               validation_data=(X_test_scaled,y_test))



        print("----------------------")

        # 
    #     means = grid_result.cv_results_['mean_test_score']
    #     stds = grid_result.cv_results_['std_test_score']
    #     params = grid_result.cv_results_['params']
    #     print("grid_result.cv_results_",grid_result.cv_results_)
    #     print("grid_result.best_estimator_",grid_result.best_estimator_)
        print("grid_result.score(X_test_scaled, y_test): ",grid_result.score(X_test_scaled, y_test))



        evaluate_result = {}

        if (_TEST == True): 
            test_score = grid_result.score(X_test_scaled, y_test)
            evaluate_result[MODEL + "_" +                           "BTC" + "_" +                           str(idx_time_unit) + "_" +                           str(idx_window_size) + "_" +                           str(idx_gap) + "_" +                           str(idx_margin_rate)] = {"MODEL":MODEL,                                            "Cryptocurrency":"BTC",    #                                         "grid_result.cv_results_":grid_result.cv_results_, \
    #                                         "grid_result.best_estimator_":grid_result.best_estimator_, \
                                            "Score":grid_result.cv_results_['mean_test_score'], \
                                            "Params":grid_result.cv_results_['params'],\
                                            "test_score":test_score}  


        elif (_TEST == False): 
            test_score = grid_result.score(X_test_scaled, y_test)
            evaluate_result[MODEL + "_" +                           coin + "_" +                           str(idx_time_unit) + "_" +                           str(idx_window_size) + "_" +                           str(idx_gap) + "_" +                           str(idx_margin_rate)] = {"MODEL":MODEL,                                            "Cryptocurrency":coin,                                             "Score":grid_result.cv_results_['mean_test_score'],                                             "Params":grid_result.cv_results_['params'],                                            "test_score":test_score}
    #     print()
    #     print("evaluate result dict: ", evaluate_result)
    #     print()

        # summarize results
        print()
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print()
        
    elif (_ENHANCE == True):
        
        
    # for checking pickle file exist
    print("---pickle saving..")
    if (_TEST == True):
        X = {}
        y = {}
        key_name_X = "X_"
        key_name_y = "y_"
        for idx_time_unit in time_unit:
            for idx_window_size in window_size:
                for idx_gap in gap:
                    for idx_margin_rate in margin_rate:
                        key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                        key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
                        if (os.path.isfile(pickle_result_dir_path +                                           "_test_" +                                           MODEL + "_" +                                           "BTC" + "_" +                                           str(idx_time_unit) + "_" +                                           str(idx_window_size) + "_" +                                           str(idx_gap) + "_" +                                           str(idx_margin_rate) +                                           "_result.pickle")) is not True:
                            with open(pickle_result_dir_path +                                       "_test_" +                                       MODEL + "_" +                                       "BTC" + "_" +                                       str(idx_time_unit) + "_" +                                       str(idx_window_size) + "_" +                                       str(idx_gap) + "_" +                                       str(idx_margin_rate) +                                       "_result.pickle", 'wb') as handle:
                                pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                
                                # Sending Email
                                if Internet_connection == True:
                                    smtp = smtplib.SMTP('smtp.naver.com', 587)
                                    smtp.ehlo()      # say Hello
                                    smtp.starttls()  # TLS 사용시 필요
                                    smtp.login('dhgdohk@naver.com', '30892793@dohk')

                                    msg = MIMEText(pickle_result_dir_path +                                                    "_test_" +                                                    MODEL + "_" +                                                    "BTC" + "_" +                                                    str(idx_time_unit) + "_" +                                                    str(idx_window_size) + "_" +                                                    str(idx_gap) + "_" +                                                    str(idx_margin_rate) +                                                    "_result.pickle")
                                    msg['Subject'] =   pickle_result_dir_path +                                                        MODEL + "_" +                                                        "BTC" + "_" +                                                        str(idx_time_unit) + "_" +                                                        str(idx_window_size) + "_" +                                                        str(idx_gap) + "_" +                                                        str(idx_margin_rate) +                                                        "_result.pickle"
                                    msg['To'] = 'dhgdohk@naver.com'
                                    smtp.sendmail('dhgdohk@naver.com', 'dhgdohk@naver.com', msg.as_string())

                                    smtp.quit()
                        else:
                            print("Already exist the file: ", pickle_result_dir_path +                                                               "_test_" +                                                               MODEL + "_" +                                                               "BTC" + "_" +                                                               str(idx_time_unit) + "_" +                                                               str(idx_window_size) + "_" +                                                               str(idx_gap) + "_" +                                                               str(idx_margin_rate) +                                                               "_result.pickle")

    elif (_TEST == False): 
        X = {}
        y = {}
        key_name_X = "X_"
        key_name_y = "y_"
        
        key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
        key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
        if (os.path.isfile(pickle_result_dir_path +                           MODEL + "_" +                           coin + "_" +                           str(idx_time_unit) + "_" +                           str(idx_window_size) + "_" +                           str(idx_gap) + "_" +                           str(idx_margin_rate) +                           "_result.pickle")) is not True:
            with open(pickle_result_dir_path +                       MODEL + "_" +                       coin + "_" +                       str(idx_time_unit) + "_" +                       str(idx_window_size) + "_" +                       str(idx_gap) + "_" +                       str(idx_margin_rate) +                       "_result.pickle", 'wb') as handle:
                pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Sending Email
                if Internet_connection == True:
                    smtp = smtplib.SMTP('smtp.naver.com', 587)
                    smtp.ehlo()      # say Hello
                    smtp.starttls()  # TLS 사용시 필요
                    smtp.login('dhgdohk@naver.com', '30892793@dohk')

                    msg = MIMEText(machine +                                    pickle_result_dir_path +                                    MODEL + "_" +                                    coin + "_" +                                    str(idx_time_unit) + "_" +                                    str(idx_window_size) + "_" +                                    str(idx_gap) + "_" +                                    str(idx_margin_rate) +                                    "_result.pickle")
                    msg['Subject'] =   machine +                                        pickle_result_dir_path +                                        MODEL + "_" +                                        coin + "_" +                                        str(idx_time_unit) + "_" +                                        str(idx_window_size) + "_" +                                        str(idx_gap) + "_" +                                        str(idx_margin_rate) +                                        "_result.pickle"
                    msg['To'] = 'dhgdohk@naver.com'
                    smtp.sendmail('dhgdohk@naver.com', 'dhgdohk@naver.com', msg.as_string())

                    smtp.quit()
        else:
            print("Already exist the file: ", pickle_result_dir_path +                                               "_test_" +                                               MODEL + "_" +                                               "BTC" + "_" +                                               str(idx_time_unit) + "_" +                                               str(idx_window_size) + "_" +                                               str(idx_gap) + "_" +                                               str(idx_margin_rate) +                                               "_result.pickle")
    
    
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #Javascript('Jupyter.notebook.session.delete()')
    #print("Jupyter notebook kernel restart")
    #time.sleep(1)
    #Javascript('Jupyter.notebook.kernel.restart()')
    #time.sleep(1)
    #print("Done")
    #Javascript('Jupyter.notebook.execute_all_cells()')
    print()


#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
#     print()


    key_name_X = "X_"
    key_name_y = "y_"
    
    
#     return eval_result


# In[8]:


def start(machine, Internet_connection, pickle_result_dir_path, boost_up_result_dir_path, _TEST, _GPU, _ENHANCED, n_jobs, MODEL, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate, epochs, best_params):
    '''
        [ATTENTION] In create_model METHOD part, need to set appropriate about GPU
        
        LINK01 -> GPU OFF
        MSI -> GPU OFF
        SLAVE04 -> GPU ON
        SLAVE05 -> GPU ON
    ''' 
    # 
    
    
    if (_ENHANCED == False and machine=="slave05"):
        #time_unit = [10,30,60]     # candle stick minutes
        #window_size = [25]  # Unit: num. of candle sticks
        #gap = [1]            # Unit: num. of candle sticks
        #margin_rate = [0.1]  # Unit: percent

        start_time = time.time()
        Start_Model( pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                     data_files_dir = dataset_dir_path_tuple_type, 
                     epochs=epochs, 
                     pickle_result_dir_path = "./evaluate_result/", 
                     boost_up_result_dir_path = "./boost_up_result/",
                     MODEL=MODEL, 
                    idx_time_unit=idx_time_unit,
                    idx_window_size=idx_window_size, 
                    idx_gap=idx_gap, 
                    idx_margin_rate=idx_margin_rate, 
                     _TEST=False, 
                     _ENHANCE=_ENHANCED,
                     _GPU=True,
                     n_jobs=2,
                     machine=machine, 
                     Internet_connection=Internet_connection, 
                     params=best_params)
        
        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')

    elif (_ENHANCED == False and machine=="link-koreatech"):
        #time_unit = [10]     # candle stick minutes
        #window_size = [10,25,50]  # Unit: num. of candle sticks
        #gap = [1]            # Unit: num. of candle sticks
        #margin_rate = [0.1]  # Unit: percent

        Start_Model( pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                     data_files_dir = dataset_dir_path_tuple_type, 
                     epochs=epochs, 
                     pickle_result_dir_path = "./evaluate_result/", 
                    boost_up_result_dir_path = "./boost_up_result/",
                     MODEL=MODEL, 
                    idx_time_unit=idx_time_unit,
                    idx_window_size=idx_window_size, 
                    idx_gap=idx_gap, 
                    idx_margin_rate=idx_margin_rate,
                     _TEST=False, 
                     _ENHANCE=_ENHANCED,
                     _GPU=False,
                     n_jobs=1,
                     machine=machine, 
                     Internet_connection=Internet_connection, 
                     params=best_params)
        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')

    elif (_ENHANCED == False and machine=="slave04"):
        #time_unit = [10]     # candle stick minutes
        #window_size = [75]  # Unit: num. of candle sticks
        #gap = [1]            # Unit: num. of candle sticks
        #margin_rate = [0.1]  # Unit: percent

        start_time = time.time()

        Start_Model(pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                 data_files_dir = dataset_dir_path_tuple_type, 
                 epochs=epochs, 
                 pickle_result_dir_path = "./evaluate_result/", 
                    boost_up_result_dir_path = "./boost_up_result/",
                 MODEL=MODEL, 
                idx_time_unit=idx_time_unit,
                idx_window_size=idx_window_size, 
                idx_gap=idx_gap, 
                idx_margin_rate=idx_margin_rate,
                 _TEST=False, 
                 _ENHANCE=_ENHANCED,
                 _GPU=True,
                 n_jobs=2,
                 machine=machine,
                 Internet_connection=Internet_connection, 
                 params=best_params)

        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')

    elif (_ENHANCED == False and machine=="link01"):
        #time_unit = [10]     # candle stick minutes
        #window_size = [25]  # Unit: num. of candle sticks
        #gap = [1,2,3]            # Unit: num. of candle sticks
        #margin_rate = [0.1]  # Unit: percent

        start_time = time.time()

        Start_Model(pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                 data_files_dir = dataset_dir_path_tuple_type, 
                 epochs=epochs, 
                 pickle_result_dir_path = "./evaluate_result/", 
                    boost_up_result_dir_path = "./boost_up_result/",
                 MODEL=MODEL, 
                idx_time_unit=idx_time_unit,
                idx_window_size=idx_window_size, 
                idx_gap=idx_gap, 
                idx_margin_rate=idx_margin_rate,
                 _TEST=False,
                 _ENHANCE=_ENHANCED,
                 _GPU=False,
                 n_jobs=1,
                 machine=machine, 
                 Internet_connection=Internet_connection,
                 params=best_params)

        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')

    elif (_ENHANCED == False and machine=="MSI"):
        #time_unit = [10]     # candle stick minutes
        #window_size = [25]  # Unit: num. of candle sticks
        #gap = [1]            # Unit: num. of candle sticks
        #margin_rate = [0.1,0.25,0.5]  # Unit: percent

        start_time = time.time()

        Start_Model(pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                 data_files_dir = dataset_dir_path_tuple_type, 
                 epochs=epochs, 
                 pickle_result_dir_path = "./evaluate_result/", 
                    boost_up_result_dir_path = "./boost_up_result/",
                 MODEL=MODEL, 
                idx_time_unit=idx_time_unit,
                idx_window_size=idx_window_size, 
                idx_gap=idx_gap, 
                idx_margin_rate=idx_margin_rate,
                 _TEST=False, 
                 _ENHANCE=_ENHANCED,
                 _GPU=False,
                 n_jobs=1,
                 machine=machine, 
                 Internet_connection=Internet_connection,
                 params=best_params)

        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')
        
    elif (_ENHANCED == False and machine=="test"):
        
        #time_unit = [10]     # candle stick minutes
        #window_size = [10]  # Unit: num. of candle sticks
        #gap = [1]            # Unit: num. of candle sticks
        #margin_rate = [0.1]  # Unit: percent

        start_time = time.time()

        Start_Model(pickle_load_dir_path = "./dataset_pickle_tuple_type/",  
                 data_files_dir = dataset_dir_path_tuple_type, 
                 epochs=epochs, 
                 pickle_result_dir_path = "./evaluate_result/", 
                    boost_up_result_dir_path = "./boost_up_result/",
                 MODEL=MODEL, 
                idx_time_unit=idx_time_unit,
                idx_window_size=idx_window_size, 
                idx_gap=idx_gap, 
                idx_margin_rate=idx_margin_rate,
                 _TEST=_TEST, 
                 _ENHANCE=_ENHANCED,
                 _GPU=_GPU,
                 n_jobs=1,
                 machine="test", 
                 Internet_connection=Internet_connection,
                 params=best_params)

        end_time = time.time()
        print()
        print("TIME: ", end_time-start_time)
        Javascript('IPython.notebook.kernel.restart()')
        time.sleep(1)
        Javascript('IPython.notebook.execute_all_cells()')
    
    elif (_ENHANCED == True):
        


# In[9]:


# # evaluate model using test data
# score = 


# In[10]:


# # load pickle file 
# import pickle
# b_x = pickle.load(open("./evaluate_result/_test_SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
# b_x


# In[11]:


# # Boost-up Acc, F1
# evaluate_result_dir_path = "./evaluate_result/acc_f1/"
# dataset_dir_path = dataset_dir_path_tuple_type 
# # dataset_dir_path = dataset_dir_path_numpy_type
# epochs = 100
# Evaluate(dataset_dir_path, data_files_dir, epochs, evaluate_result_dir_path, time_unit, window_size, gap, margin_rate)


# In[12]:


model_info = {}
model_info["test"] = {#"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "MODEL_list":["SimpleRNN"],
                      "time_unit":[10], 
                      "window_size":[10],
                      "gap":[1], 
                      "margin_rate":[0.1]}

model_info["slave05"] = {"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "time_unit":[10,30,60], 
                      "window_size":[25],
                      "gap":[1], 
                      "margin_rate":[0.1]}
model_info["slave04"] = {"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "time_unit":[10], 
                      "window_size":[10,50,75],
                      "gap":[1], 
                      "margin_rate":[0.1]}
model_info["link01"] = {"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "time_unit":[10], 
                      "window_size":[25],
                      "gap":[2,3], 
                      "margin_rate":[0.1]}
model_info["link-koreatech"] = {"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "time_unit":[10], 
                      "window_size":[25],
                      "gap":[1], 
                      "margin_rate":[0.25,0.5]}

model_info["MSI"] = {"MODEL_list":["SimpleRNN", "LSTM", "GRU"],
                      "time_unit":[10], 
                      "window_size":[25],
                      "gap":[1], 
                      "margin_rate":[0.1,0.25,0.5]}


# # Hyperio test

# In[13]:


# from __future__ import print_function
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K

# def create_model_SimpleRNN_non_GPU_test(x_train, y_train, x_val, y_val, params):
#     model = Sequential()
#     model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
#         SimpleRNN(
# #             n_state_units, 
#                   params['first_neuron'],
#                   input_dim=x_train.shape[2], # [dataset 크기, 윈도우 사이즈, 32(코인개수*OLHC)]
# #                   input_shape=(window_size, 32),
# #                   use_bias=True, 
#                   #activation='relu'
#         ))
# #                   kernel_initializer='glorot_uniform', 
# #                   recurrent_initializer='orthogonal', 
# #                   bias_initializer='zeros', 
# #                   dropout=0.0,
# #                   recurrent_dropout=0.0))
    
# #     model.add(Dense(units=neurons))
# #     model.add(Dropout(dropout_rate))
        
#     model.add(Dropout(params['dropout']))
#     model.add(Dense(y_train.shape[1],
#                     activation=params['last_activation']))
        
# #     model.add(Dense(units=2))
# #     model = multi_gpu_model(model, gpus=2)
# #     model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
# #                   loss=params['loss'],
# #                   metrics=['acc'])
#     model = multi_gpu_model(model, gpus=2)
#     model.compile(optimizer=params['optimizer'](),
#                   loss=params['loss'],
#                   metrics=['acc', f1_score])
    
#     out = model.fit(x_train, y_train,
#                     batch_size=params['batch_size'],
#                     epochs=params['epochs'],
#                     verbose=1,
#                     validation_data=[x_val, y_val])
# #                     callbacks=early_stopper(params['epochs'], mode='strict'))
    
#     return out


# In[14]:


# from keras.optimizers import Adam, Nadam
# from keras.activations import softmax
# from keras.losses import categorical_crossentropy, logcosh

# pickle_load_dir_path = "./dataset_pickle_tuple_type/"
# X = {}
# y = {}


# MODEL = ["SimpleRNN"]
# idx_time_unit = 10     # candle stick minutes
# idx_window_size = 25  # Unit: num. of candle sticks
# idx_gap = 1            # Unit: num. of candle sticks
# idx_margix_rate = 0.1  # Unit: percent

# key_name_X = "X_"
# key_name_y = "y_"


# key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margix_rate)
# key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margix_rate)

# X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margix_rate)
# y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margix_rate)

# y_single = {}
# y_single['BTC'] = y[:, 1]
# y_single['ETH'] = y[:, 2]
# y_single['XRP'] = y[:, 3]
# y_single['BCH'] = y[:, 4]
# y_single['LTC'] = y[:, 5]
# y_single['DASH'] = y[:, 6]
# y_single['ETC'] = y[:, 7]

# coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

# y2 = onehottify(y_single['BTC'], n=2) 
# #                         for coin in coin_list2:
# #                             print("y_single["+coin+"]"+".shape")
# #                             print(y_single[coin].shape)
# #                             print()




# X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.1, random_state=42)
# #     print("X_train.shape")
# #     print(X_train.shape)
# #     print("y_train.shape")
# #     print(y_train.shape)
# #     print()
# #     print("X_test.shape")
# #     print(X_test.shape)
# #     print("y_test.shape")
# #     print(y_test.shape)
# #     print()

# n_coins = 8
# n_price = 4
# n_steps = idx_window_size # 원래 100이었음. reshape 문제 때문에 수정함

# X_train_2 = X_train.transpose([0, 2, 1, 3])
# X_test_2 = X_test.transpose([0, 2, 1, 3])
# #     print("X_train_2.shape")
# #     print(X_train_2.shape)
# #     print("X_test_2.shape")
# #     print(X_test_2.shape)
# #     print()

# X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
# X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
# #     print("X_train_3.shape")
# #     print(X_train_3.shape)
# #     print("X_test_3.shape")
# #     print(X_test_3.shape)
# #     print()

# X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps*n_coins * n_price])
# X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps*n_coins * n_price])
# #     print("X_train_reshape.shape")
# #     print(X_train_reshape.shape)
# #     print("X_test_reshape.shape")
# #     print(X_test_reshape.shape)
# #     print()

# scaler = MinMaxScaler()
# scaler.fit(X_train_reshape)
# X_train_scaled = scaler.transform(X_train_reshape)
# X_test_scaled = scaler.transform(X_test_reshape)

# X_train_scaled = X_train_scaled.reshape(-1, n_steps, n_coins * n_price)

# X_test_scaled = X_test_scaled.reshape(-1, n_steps, n_coins * n_price)

# X_train_scaled, X_test_scaled = input_reshape(X_train_scaled, X_test_scaled, n_steps, n_coins, n_price)

                        
# p = {'lr': (0.1, 0.01, 0.001),
#      'first_neuron':[4, 8, 16, 32, 64, 128],
#      'batch_size': [64,128,256],
#      'epochs': [100],
#      'activation':['relu', 'softmax'],
#      'dropout': (0, 0.40, 10),
#      'optimizer': [Adam, Nadam],
#      'kernel_initializer':['glorot_uniform', 'uniform', 'he_uniform'],
#      'recurrent_initializer':['orthogonal'], 
#      'bias_initializer':['zeros'],
#      'loss': ['categorical_crossentropy', 'logcosh'],
#      'last_activation': ['softmax'],
#      'weight_regulizer':[None],
#      'emb_output_dims': [None]}

# import hyperio as hy
# h = hy.Hyperio(X_train_scaled, y_train, 
#                params=p, 
#                dataset_name='coin', 
#                experiment_no='1', 
#                model=create_model_SimpleRNN_non_GPU_test,
#                # create_model_SimpleRNN_non_GPU_test(x_train, y_train, x_val, y_val, params)
#                grid_downsample=.01)


# In[15]:


# h


# In[16]:


# h.result


# # Boost-Up

# In[17]:


evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
key = str(evaluate_result.keys())[12:-3]
print(evaluate_result[key]['Best_Params'])
best_params = evaluate_result[key]['Best_Params']


# In[20]:


# Machine = "test"
# # If Machine is "test" then, _TEST argument must be True
# test = False
# _GPU= False
# _ENHANCED = True

# Internet_connection=True
# n_jobs=-1
# epochs=1

# pickle_result_dir_path = "./evaluate_result/"
# boost_up_result_dir_path = "./boost_up_result/"

# for model in model_info[Machine]["MODEL_list"]:
#     for idx_time_unit in model_info[Machine]["time_unit"]:
#         for idx_window_size in model_info[Machine]["window_size"]:
#             for idx_gap in model_info[Machine]["gap"]:
#                 for idx_margin_rate in model_info[Machine]["margin_rate"]:
#                     start(machine=Machine, 
#                          Internet_connection=Internet_connection, 
#                          pickle_result_dir_path=pickle_result_dir_path, 
#                          boost_up_result_dir_path=boost_up_result_dir_path,
#                          _TEST=test, 
#                          _GPU=_GPU, 
#                          _ENHANCED=_ENHANCED,
#                          n_jobs=n_jobs,
#                          MODEL = model, 
#                          idx_time_unit=idx_time_unit, 
#                          idx_window_size=idx_window_size, 
#                          idx_gap=idx_gap, 
#                          idx_margin_rate=idx_margin_rate,
#                          epochs=epochs,
#                          best_params=best_params)
#                     Javascript('IPython.notebook.kernel.restart()')
#                     time.sleep(1)
#                     Javascript('IPython.notebook.execute_all_cells()')


# In[21]:


evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
key = str(evaluate_result.keys())[12:-3]
print(evaluate_result[key]['Best_Params'])
best_params = evaluate_result[key]['Best_Params']


# In[60]:


def Boost_up_SimpleRNN(window_size, n_state_units=32, activation='softmax', optimizer='adam', init='glorot_uniform', dropout_rate=0.0, neurons=2):
    model = Sequential()
    model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
        SimpleRNN(n_state_units, 
                  input_shape=(window_size, 32),
                  use_bias=True, 
                  activation='tanh',
                  kernel_initializer='glorot_uniform', 
                  recurrent_initializer='orthogonal', 
                  bias_initializer='zeros', 
                  dropout=0.0,
                  recurrent_dropout=0.0))
    
    model.add(Dense(units=neurons))
    model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=optimizer, 
                  metrics=["accuracy", f1_score])

    return model


# In[66]:


pickle_load_dir_path = "./dataset_pickle_tuple_type/"
idx_time_unit = 10
idx_window_size = 10
idx_gap = 1
idx_margin_rate = 0.1

X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)
y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)

y_single = {}
y_single['BTC'] = y[:, 1]
y_single['ETH'] = y[:, 2]
y_single['XRP'] = y[:, 3]
y_single['BCH'] = y[:, 4]
y_single['LTC'] = y[:, 5]
y_single['DASH'] = y[:, 6]
y_single['ETC'] = y[:, 7]

coin = "BTC"
y2 = onehottify(y_single[coin], n=2)

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.1, random_state=42)

n_coins = 8
n_price = 4
n_steps = idx_window_size # 원래 100이었음. reshape 문제 때문에 수정함

X_train_2 = X_train.transpose([0, 2, 1, 3])
X_test_2 = X_test.transpose([0, 2, 1, 3])

X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])

X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps*n_coins * n_price])
X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps*n_coins * n_price])

scaler = MinMaxScaler()
scaler.fit(X_train_reshape)
X_train_scaled = scaler.transform(X_train_reshape)
X_test_scaled = scaler.transform(X_test_reshape)

X_train_scaled = X_train_scaled.reshape(-1, n_steps, n_coins * n_price)

X_test_scaled = X_test_scaled.reshape(-1, n_steps, n_coins * n_price)

# X_train_scaled, X_test_scaled = input_reshape(X_train_scaled, X_test_scaled, n_steps, n_coins, n_price)

# Boost_up_SimpleRNN(best_params, X_train_scaled, y_train, X_test_scaled, y_test, epochs=100)

model = Sequential()
model.add(# if문을 통해 여러 RNN모델 쓸 수 있도록 하기, SimpleRNN외에 다른 RNN모델 찾아보기
    SimpleRNN(units=best_params['n_state_units'][0], 
              input_shape=(best_params['window_size'][0], 32),
              use_bias=True, 
              activation='tanh',
              kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', 
              bias_initializer='zeros', 
              dropout=0.0,
              recurrent_dropout=0.0))

model.add(Dense(units=neurons))
model.add(Dropout(dropout_rate))
#     model.add(Dense(units=2))
#     model = multi_gpu_model(model, gpus=2)

model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=optimizer, 
              metrics=["accuracy", f1_score])

history = model.fit(X_train, y_train, epochs=100, verbose=1)


# In[63]:


best_params


# In[72]:


evaluate_result = pickle.load(open("./evaluate_result/SimpleRNN_BTC_10_10_1_0.1_result.pickle", "rb"))
key = str(evaluate_result.keys())[12:-3]
# print(evaluate_result[key]['Best_Params'])
best_params = evaluate_result[key]['Best_Params']
train_score = evaluate_result[key]['train_score']
test_score = evaluate_result[key]['test_score']
print(best_params)
print(train_score)
print(test_score)

