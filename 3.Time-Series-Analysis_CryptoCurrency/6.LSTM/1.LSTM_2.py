import pprint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from os import listdir
import csv
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import tensorflow as tf
import keras
from keras import __version__
from keras import backend as K
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, BatchNormalization, Embedding, CuDNNLSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os.path
import os
from os.path import isfile, join
import boto3
import pickle
import time
import warnings
import smtplib
from email.mime.text import MIMEText
from keras.utils import multi_gpu_model
# import email_info
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
# stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.001)
early_stop = keras.callbacks.EarlyStopping(monitor='f1_metric', mode='max', patience = 2, verbose=1)
'''
slave05> [30, 60] BTC, ETH, XRP
slave04> [30, 60] BCH, LTC, DASH
link> [10, 30, 60] ETC
'''
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
pickle_result_dir_path = '../evaluate_result/eXtream_Gradient_Boosting/'
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.getcwd())
# from link_aws_key import *
pp = pprint.PrettyPrinter(indent=4)
print('Using Keras version:', __version__, 'backend:', K.backend())

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
def sending_email(message):
    f = open('../../../../email_info.bin', 'rb')
    a = pickle.load(f)

    smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
    smtp.ehlo()  # say Hello
    # smtp.starttls()  # TLS 사용시 필요
    smtp.login(a['email'], a['pw'])

    msg = MIMEText(str(message))
    msg['Subject'] = 'slave04-실험 완료'
    msg['To'] = 'ulujo_dohk@naver.com'
    smtp.sendmail(a['email'], 'ulujo_dohk@naver.com', msg.as_string())

    smtp.quit()
def get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir):
    trading_files = []
    for coin_name in coin_name_list:
        for data_file_name in [f for f in listdir(data_files_dir) if isfile(join(data_files_dir, f))]:
            if coin_name in data_file_name:
                trading_files.append(data_file_name)

    start_ms_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    end_ms_time = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000

    raw_data = {}  # 전체 CSV Raw 데이터
    for coin_name in coin_name_list:
        raw_data[coin_name] = []

    # KRW 제외한 나머지 CSV Raw 데이터 수집
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
def recall(y_true, y_pred):
    K.set_epsilon(1e-05)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    K.set_epsilon(1e-05)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def make_cryptocurrency_dataset(coin_name_list, start_date, end_date, time_unit, window_size, gap, margin_rate):
    y_trv = []
    y_btv = []
    num_coins = len(coin_name_list)
    # raw_data = get_all_raw_data_from_aws(coin_name_list, start_date, end_date)
    data_files_dir = './data/old_cleanup_data/'
    raw_data = get_all_raw_data(coin_name_list, start_date, end_date, data_files_dir)

    num_sticks = len(raw_data['BCH'])

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
                X[idx][idx_coin][idx_in_window].append(
                    float(raw_data[coin_name][idx_stick][3]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(
                    float(raw_data[coin_name][idx_stick][4]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(
                    float(raw_data[coin_name][idx_stick][5]) / close_price_in_last_idx_in_window)
                X[idx][idx_coin][idx_in_window].append(float(raw_data[coin_name][idx_stick][6]))

            target_idx_for_window = int(idx + time_unit / 10 * window_size - 1 + gap)
            # print('raw data: ', raw_data)
            # print('type of raw data:' , type(raw_data))
            # print('key of raw data:' , (raw_data.keys()))
            # print('-----------------')
            # print('raw data[coin_name]: ', raw_data[coin_name])
            # print('type of raw data[coin_name]: ', type(raw_data[coin_name]))
            # print('-----------------')
            # print('raw data[coin_name][target_idx_for_window]: ', raw_data[coin_name][target_idx_for_window])
            # print('type of raw data[coin_name][target_idx_for_window]: ', type(raw_data[coin_name][target_idx_for_window]))
            # print('len of raw data[coin_name][target_idx_for_window]: ', len(raw_data[coin_name][target_idx_for_window]))
            # print('-----------------')
            # print((raw_data[coin_name][target_idx_for_window][3]), end="\n\n")
            target_price = float(raw_data[coin_name][target_idx_for_window][3])
            # print("-----$$$$$$$$$-------")

            target = 0
            if target_price >= close_price_in_last_idx_in_window * (1.0 + float(margin_rate) / 100.0):
                target = 1
            y[idx].append(target)

            idx_coin += 1

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y
def create_model(n_state_units,
                 neurons,
                 init,
                 activation,
                 activation_1,
                 window_size,
                 optimizer,
                 weight_constraint,
                 dropout_rate):

    model = Sequential()
    model.add(
        LSTM(units=n_state_units,
             activation=activation,
             input_shape=(window_size, 32)))
    model.add(Dense(neurons, input_dim=8,
                    init=init,
                    activation=activation_1,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(8, init=init, activation=activation_1))
    model.add(Dense(2, init=init, activation=activation_1))
    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[f1_metric, 'accuracy', recall, precision])

    return model

coin_list = ["KRW", "BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
start_date = "2017-01-01 00:00:00"
end_date = "2017-12-31 23:50:00"

time_unit_list = [10, 30, 60]  # candle stick minutes
pickle_load_dir_path = './data/RNN_coin/'
pickle_result_dir_path = './evaluate_result/'
pickle_result_for_email_transfer = './evaluate_result/for_email_transfer/'
# coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
coin_list2 = ["ETH", "XRP", "BCH", "LTC", "DASH"]

idx_time_unit = 10  # 10, 30, 60
idx_window_size = 100  # 25, 50, 100
idx_gap = 1
idx_margin_rate = 0.1
# _GPU = True
n_jobs = 1
cv = 2
n_iter = 100  # maximum 30
dataset_scale = -1  # [:10000] for test

# scaler = MinMaxScaler(feature_range=(-1, 1))






for time_unit in time_unit_list:
    window_size = 50  # Unit: num. of candle sticks
    gap = 1  # Unit: num. of candle sticks
    margin_rate = 0.1  # Unit: percent
    print("~~~~~~~{}~~~~~~~".format([time_unit, window_size, gap, margin_rate]))

    if os.path.isfile(str(time_unit) + '_' + str(window_size) + '_' + str(gap) + '_' + str(margin_rate) + '_X.csv'):
        with open(str(time_unit) + '_' + str(window_size) + '_' + str(gap) + '_' + str(margin_rate) + '_X.csv',
                  'rb') as X_file:
            X = pickle.load(X_file)

        with open(str(time_unit) + '_' + str(window_size) + '_' + str(gap) + '_' + str(margin_rate) + '_y.csv',
                  'rb') as y_file:
            y = pickle.load(y_file)
    else:

        X, y = make_cryptocurrency_dataset(
            coin_list,
            start_date,
            end_date,
            time_unit,
            window_size,
            gap,
            margin_rate
        )

        with open(str(time_unit) + '_' + str(window_size) + '_' + str(gap) + '_' + str(margin_rate) + '_X.csv',
                  'wb') as X_file:
            pickle.dump(X, X_file)

        with open(str(time_unit) + '_' + str(window_size) + '_' + str(gap) + '_' + str(margin_rate) + '_y.csv',
                  'wb') as y_file:
            pickle.dump(y, y_file)

    print(X.shape)
    print(y.shape)


    X_reshape = np.reshape(X, (X.shape[0], -1))

    y_single = {}
    y_single['BTC'] = y[:, 1]
    y_single['ETH'] = y[:, 2]
    y_single['XRP'] = y[:, 3]
    y_single['BCH'] = y[:, 4]
    y_single['LTC'] = y[:, 5]
    y_single['DASH'] = y[:, 6]
    y_single['ETC'] = y[:, 7]







    coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}

    for coin in coin_list2:
        y2 = onehottify(y_single[coin], n=2)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y2,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=True)

        print("####################### {} #######################".format(coin))

        best_f1_score = 0.0
        best_conf_matrix = None
        best_metrics = None
        best_param = None
        best_y_pred = None
        best_idx = 0

        # scaler = MinMaxScaler(feature_range=(-1, 1))

        n_coins = 8
        n_price = 4
        n_steps = idx_window_size

        X_train_2 = X_train.transpose([0, 2, 1, 3])
        X_test_2 = X_test.transpose([0, 2, 1, 3])

        X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, (n_coins * n_price)//2])
        X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, (n_coins * n_price)//2])

        X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps * n_coins * n_price // 2])
        X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps * n_coins * n_price // 2])

        scaler = StandardScaler()
        scaler.fit(X_train_reshape)

        X_train_scaled = scaler.fit_transform(X_train_reshape)
        X_test_scaled = scaler.fit_transform(X_test_reshape)

        X_train_scaled = X_train_scaled.reshape(-1,
                                                n_steps,
                                                n_coins * n_price)
        X_test_scaled = X_test_scaled.reshape(-1,
                                              n_steps,
                                              n_coins * n_price)

        X_train_scaled, X_test_scaled = input_reshape(X_train_scaled,
                                                      X_test_scaled,
                                                      n_steps,
                                                      n_coins,
                                                      n_price)

        model = KerasClassifier(build_fn=create_model, verbose=True)

        #     model = KerasClassifier(build_fn=vanilla_LSTM,
        #                             epochs=epochs,
        #                             batch_size=300,
        #                             verbose=True)

        param_grid_create_model = {'batch_size': [128],
                                   'epochs': [1000],
                                   'neurons': [5, 30],
                                   'window_size': [idx_window_size],
                                   'init': ['normal', 'glorot_normal', 'he_normal'],
                                   'n_state_units': [32, 64, 128],
                                   'activation': ['softmax', 'relu', 'tanh', 'sigmoid'],
                                   'activation_1': ['tanh', 'sigmoid', 'relu'],
                                   'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adam'],
                                   'weight_constraint': [1],
                                   'dropout_rate': [0.0]
                                   }

        grid = RandomizedSearchCV(estimator=model,
                                  param_distributions=param_grid_create_model,
                                  n_iter=n_iter,
                                  cv=cv,
                                  random_state=42,
                                  n_jobs=n_jobs,
                                  verbose=1)

        with K.tf.device('/gpu:0'):
            #                 grid_result = grid.fit(X_train_scaled,
            #                                        y_train,
            #                                        validation_data=(X_test_scaled,
            #                                                         y_test),
            #                                        batch_size=512,
            #                                        callbacks=[stopper])
            grid_result = grid.fit(X_train_scaled,
                                   y_train,
                                   validation_data=(X_test_scaled,
                                                    y_test),
                                   callbacks=[early_stop])
        # grid_result = grid_result.reset_states()

        print("----------------------")
        print("grid_result.score(X_test_scaled, y_test): ", grid_result.score(X_test_scaled, y_test))






        evaluate_result = {}
        test_score = grid_result.score(X_test_scaled, y_test)
        evaluate_result[coin + "_" + \
                        str(idx_time_unit) + "_" + \
                        str(idx_window_size) + "_" + \
                        str(idx_gap) + "_" + \
                        str(idx_margin_rate)] = {"Cryptocurrency": coin, \
                                                 "Score": grid_result.cv_results_['mean_test_score'], \
                                                 "Params": grid_result.cv_results_['params'], \
                                                 "test_score": test_score}




        params_XGB = {'learning_rate': [0.005, 0.05, 0.5, 1.0],
                      'n_estimators': [10, 50, 100, 500],
                      'max_depth': [3, 5, 10],
                      'min_child_weight': [1, 3, 5, 10],
                      'gamma': [0, 0.1, 0.5, 1.0],
                      'subsample': [0.8],
                      'colsample_bytree': [0.8],
                      'objective': ['multi:softmax', 'binary:logistic'],
                      'nthread': [3, 5, 10],
                      'scale_pos_weight': [1],
                      'seed': [42]
                      }

        xgb = XGBClassifier(silent=False)
        folds = 5  # 5이상 높여서 실험
        param_comb = 100  # 5이상 높여서 실험
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(xgb,
                                           param_distributions=params_XGB,
                                           n_iter=param_comb,
                                           scoring='f1',
                                           n_jobs=1,
                                           cv=skf.split(X_train[coin], y_train[coin]),
                                           verbose=2,
                                           random_state=1001)

        grid_result = random_search.fit(X_train[coin],
                                        y_train[coin])
        y_pred = grid_result.predict(X_test[coin])
        m = confusion_matrix(y_test[coin], y_pred)
        metrics = precision_recall_fscore_support(y_test[coin], y_pred)

        f1_score = (metrics[2][0] * metrics[3][0] + metrics[2][1] * metrics[3][1]) / (metrics[3][0] + metrics[3][1])
        print("f1_score:", f1_score)

        evaluate_result = {}
        test_score = grid_result.score(X_test[coin], y_test[coin])
        evaluate_result[coin + "_" + \
                        str(time_unit) + "_" + \
                        str(window_size) + "_" + \
                        str(gap) + "_" + \
                        str(margin_rate)] = {"Cryptocurrency": coin,
                                             "Score": grid_result.cv_results_['mean_test_score'],
                                             "Params": grid_result.cv_results_['params'],
                                             "f1_score": f1_score,
                                             "metrics":metrics}


        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_y_pred = y_pred
            best_conf_matrix = m
            best_metrics = metrics
            # best_param = param
            # best_idx = idx

        print(best_idx, ":", best_param)
        print(best_conf_matrix)

        print(classification_report(y_test[coin], best_y_pred))

        print("Accuracy: {:.4}".format(
            (best_conf_matrix[0][0] + best_conf_matrix[1][1]) / (
                (best_conf_matrix[0][0] + best_conf_matrix[0][1] + best_conf_matrix[1][0] + best_conf_matrix[1][1])))
        )

        print("Precision: {:.4}".format(
            (best_metrics[0][0] * best_metrics[3][0] + best_metrics[0][1] * best_metrics[3][1])
            /
            (best_metrics[3][0] + best_metrics[3][1])
        ))

        print("Recall: {:.4}".format(
            (best_metrics[1][0] * best_metrics[3][0] + best_metrics[1][1] * best_metrics[3][1])
            /
            (best_metrics[3][0] + best_metrics[3][1])
        ))

        print("F1-score: {:.4}".format(
            (best_metrics[2][0] * best_metrics[3][0] + best_metrics[2][1] * best_metrics[3][1])
            /
            (best_metrics[3][0] + best_metrics[3][1])
        ))

        print(evaluate_result)
        if (os.path.isfile(pickle_result_dir_path + \
                           coin + "_" + \
                           str(time_unit) + "_" + \
                           str(window_size) + "_" + \
                           str(gap) + "_" + \
                           str(margin_rate) + \
                           "_param.pickle")) is not True:
            with open(pickle_result_dir_path + \
                      coin + "_" + \
                      str(time_unit) + "_" + \
                      str(window_size) + "_" + \
                      str(gap) + "_" + \
                      str(margin_rate) + \
                      "_param.pickle", 'wb') as handle:
                pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                sending_email(evaluate_result)