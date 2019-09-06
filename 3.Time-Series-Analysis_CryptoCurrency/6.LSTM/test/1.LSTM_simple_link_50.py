"""
LSTM
순환 신경망

"""

# import library
import tensorflow as tf
import keras
from keras import __version__
from keras import backend as K
from keras.constraints import maxnorm

print('Using Keras version:', __version__, 'backend:', K.backend())
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

# stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=0.001)
early_stop = keras.callbacks.EarlyStopping(monitor='f1_metric', mode='max', patience = 2, verbose=1)

sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')

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
def _f1_score(y_true, y_pred):
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

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
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
    f = open('../../../email_info.bin', 'rb')
    a = pickle.load(f)

    smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
    smtp.ehlo()  # say Hello
    # smtp.starttls()  # TLS 사용시 필요
    smtp.login(a['email'], a['pw'])

    msg = MIMEText(str(message))
    msg['Subject'] = 'slave05-실험 완료'
    msg['To'] = 'ulujo_dohk@naver.com'
    smtp.sendmail(a['email'], 'ulujo_dohk@naver.com', msg.as_string())

    smtp.quit()


def sending_email_for_file_transfer(message, pickle_file_path):
    f = open('../../../email_info.bin', 'rb')
    a = pickle.load(f)

    smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
    smtp.ehlo()  # say Hello
    # smtp.starttls()  # TLS 사용시 필요
    smtp.login(a['email'], a['pw'])

    # msg = MIMEText('본문 테스트 메시지')
    msg = MIMEMultipart()
    part = MIMEText(str(message))
    msg.attach(part)
    path = pickle_file_path
    msg['Subject'] = 'slave05-실험 완료'
    msg['To'] = 'link.dohk@gmail.com'
    with open(path, 'rb') as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())  # payload: osi 7-layers
        encoders.encode_base64(part)  # base64 encoding: 영상, 이미지 파일을 문자열 형태로 변환
        part.add_header('Content-Disposition', 'attachment', filename=path)
        msg.attach(part)

    smtp.sendmail('dhgdohk@naver.com', 'link.dohk@gmail.com', msg.as_string())
    smtp.quit()

class SortedDisplayDict(dict):
    def __str__(self):
        return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

    def ordered_keys(self):
        return sorted(self.keys())
def search(dirname):
    filenames = os.listdir(dirname)
    fileList = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        fileList.append(full_filename)
    return fileList
def drawGraph(dir):
    fileList = search(dir)
    temp_list = []
    cluster_coef_value_list_lstm = []
    for file in fileList:
        temp_list.append(pd.read_pickle(file))

        cluster_coef_value_list_lstm = []
    for i in range(len(temp_list)):
        # print(temp_list[i])
        # print()
        # print(temp_list[i][list(temp_list[i].keys())[0]])
        # print()
        cluster_coef_value_list_lstm.append(temp_list[i][list(temp_list[i].keys())[0]]['Score'][0])
        # print(score)
        # print()

    cluster_coef_value_list_gradientBoosting = [0.671, 0.616, 0.622, 0.672, 0.7, 0.69, 0.69]

    first_legend_label = 'xgboost'
    second_legend_label = 'lstm'
    x_label = 'Cryptocurrency'
    y_label = 'f1-score'
    filename = str(datetime.now())

    graph(cluster_coef_value_list_gradientBoosting,
          cluster_coef_value_list_lstm[:7],
          first_legend_label,
          second_legend_label,
          x_label,
          y_label,
          filename)
def graph(cluster_coef_value_list_xgboost,
          cluster_coef_value_list_lstm,
          first_legend_label,
          second_legend_label,
          x_label,
          y_label,
          filename):
    '''
    < EXAMPLE >
    f1_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 9개
    first_legend_label = 'xgboost'
    second_legend_label = 'lstm'
    x_label = 'Cryptocurrency'
    y_label = 'f1-score'
    filename = '_clustering_and_diameter'
    '''
    link_addition_ratio = f1_score
    data_profile = {
        "Infocom05": {
            "file_name": "data_infocom05.csv",
            "num_nodes": 41,
            "median": 2684,
            "mean": 9961,
            "std": 26513,
            "contact_weight_map": [((6, 9), 2187), ((12, 1), 4943), ((7, 12), 40849), ((1, 6), 768)],

            "graph": {},
            "numberOfNodes": {},
            "numberOfEdges": {},
            "durationThreshold": {},

            "density": {},
            "clustering_coefficient": {},
            "diameter_cc": {},

            "global_bet": {},
            "Brandes_ego_bet": {},
            "Brandes_ego_elapsed_time": {},
            "Brandes_xego_bet": {},
            "Brandes_xego_elapsed_time": {},

            "Proposed_ego_bet": {},
            "Proposed_ego_elapsed_time": {},
            "Proposed_xego_bet": {},
            "Proposed_xego_elapsed_time": {},

            "ego_global_pearson_corr": {},
            "xego_global_pearson_corr": {},

            "ego_global_spearman_corr": {},
            "xego_global_spearman_corr": {},

            "ego_node_coverage_in_connected_component": {},
            "ego_edge_coverage_in_connected_component": {},

            "xego_node_coverage_in_connected_component": {},
            "xego_edge_coverage_in_connected_component": {}
        }
    }
    xticklabels = [r'$BTC$', r'$ETH$', r'$XRP$', r'$BCH$', r'$LTC$', r'$DASH$', r'$ETC$']
    yticklabels = [r'$0.40$', r'$0.45$', r'$0.50$', r'$0.55$', r'$0.60$', r'$0.65$', r'$0.70$', r'$0.75$', r'$0.80$']
    # yticklabels2 = [r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$']

    ind = np.arange(len(xticklabels))

    fig, axes = plt.subplots(1, 1, figsize=(8.719, 6.07))
    subfigures = {}
    data_name = 'Infocom05'
    subfigures[data_name] = axes
    subfigures[data_name].set_xticks(ind)
    subfigures[data_name].set_xticklabels(xticklabels, fontsize=21)
    subfigures[data_name].set_ylim([0.4, 0.8])
    subfigures[data_name].set_yticklabels(yticklabels, fontsize=21)

    cluster_coef_dic = SortedDisplayDict(data_profile[data_name]['clustering_coefficient'])
    cluster_coef_value_list = [cluster_coef_dic[x] for x in cluster_coef_dic.ordered_keys()]

    # cluster_coef_value_list_xgboost = [0.56, 0.67, 0.66, 0.78, 0.45, 0.47, 0.65]
    # cluster_coef_value_list_lstm = [0.66, 0.77, 0.68, 0.68, 0.55, 0.57, 0.66]
    cluster_coef_value_list_xgboost = cluster_coef_value_list_xgboost
    cluster_coef_value_list_lstm = cluster_coef_value_list_lstm
    subfigures[data_name].plot(ind, cluster_coef_value_list_xgboost,
                               color='k', linestyle='-', marker='s', markersize=8,
                               label=first_legend_label)
    subfigures[data_name].plot(ind, cluster_coef_value_list_lstm,
                               color='k', linestyle='--', marker='^', markersize=8,
                               label=second_legend_label)
    subfigures[data_name].set_xlabel(x_label, fontsize=21)
    subfigures[data_name].set_ylabel(y_label, fontsize=21)

    # subfigures[data_name].set_title(data_name, fontsize=21)
    subfigures[data_name].grid(True)
    if data_name == 'Infocom05':
        subfigures[data_name].legend(loc=30, fontsize=18)

    # subfigures[data_name] = subfigures[data_name].twinx()
    # subfigures[data_name].set_ylim([0, 12])
    # subfigures[data_name].set_yticklabels(yticklabels2, fontsize=21)

    diameter_dic = SortedDisplayDict(data_profile[data_name]['diameter_cc'])
    diameter_list = [diameter_dic[x] for x in diameter_dic.ordered_keys()]
    # subfigures[data_name].bar(ind, diameter_list, barWidth, color='k', alpha=0.3, label='Diameter of Connected Component')
    # subfigures[data_name].set_ylabel('Diameter of Connected Component', fontsize=21)
    if data_name == 'Infocom05':
        subfigures[data_name].legend(loc=4, fontsize=18)
    subfigures[data_name].grid(True)

    fig.savefig('./img/' + filename + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()

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

def vanilla_LSTM(window_size, units_1, units_2, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    #     global metrics
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  # what portion of gpu to use
    session = tf.Session(config=config)
    K.set_session(session)

    model = Sequential()
    model.add(
        LSTM(units=n_state_units,
             activation=activation_1,
             input_shape=(window_size, 32)))
    model.add(Dense(2))
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def stacked_LSTM(window_size, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        LSTM(units=n_state_units,
             activation=activation_1,
             return_sequences=True,
             input_shape=(window_size, 32)))
    model.add(
        LSTM(units=n_state_units,
             activation=activation_2))
    model.add(Dense(2))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def bidirectioanl_LSTM(window_size, units_1, units_2, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_1),
            input_shape=(window_size, 32)))
    model.add(Dense(2))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def bidirectioanl_LSTM_with_BN(window_size, units_1, units_2, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    # https://keras.io/layers/normalization/
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_1),
            input_shape=(window_size, 32)))
    model.add(BatchNormalization())
    model.add(Dense(2))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def stacked_bidirectioanl_LSTM(window_size, units_1, units_2, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_1,
                 return_sequences=True),
            input_shape=(window_size, 32)))
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_2)))
    model.add(Dense(2))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def cuDNN_LSTM(window_size, units_1, units_2, n_state_units=32, activation_1='softmax', activation_2='relu', optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_1,
                 return_sequences=True),
            input_shape=(window_size, 32)))
    model.add(
        Bidirectional(
            LSTM(units=n_state_units,
                 activation=activation_2)))
    model.add(Dense(2))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model
def advanced_LSTM(window_size, activation, optimizer='adam', n_state_units=32):
    # https://keras.io/layers/normalization/
    #     global metrics
    K.clear_session()
    model = Sequential([
        Bidirectional(LSTM(n_state_units, activation=activation, recurrent_activation='sigmoid')),
        # BatchNormalization(), # BatchNormailization을 적용하면, nan값이 뜬다.
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy', recall, precision])

    return model


def Start_Model():
    pickle_load_dir_path = './data/RNN_coin/'
    pickle_result_dir_path = './evaluate_result/'
    pickle_result_for_email_transfer = './evaluate_result/for_email_transfer/'

    idx_time_unit = 10  # 10, 30, 60
    idx_window_size = 50  # 25, 50, 100
    idx_gap = 1
    idx_margin_rate = 0.1
    # _GPU = True
    n_jobs = 1
    cv = 2
    n_iter = 100  # maximum 30
    dataset_scale = -1  # [:10000] for test

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()

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

    param_grid_vanilla = {
                  'window_size': [idx_window_size],
                  'units_1': [32, 64],
                  'units_2': [32, 64],
                  'n_state_units': [64, 128],
                  'activation_1': ['tanh', 'sigmoid', 'relu'],
                  'activation_2': ['tanh', 'sigmoid', 'relu'],
                  'optimizer': ['rmsprop', 'Adam', 'SGD']}

    param_grid_test = {'window_size': [idx_window_size],
                  'units_1': [16],
                  'n_state_units': [32],
                  'activation_1': ['relu'],
                  'activation_2': ['relu'],
                  'optimizer': ['Adam']}

    param_grid_advanced_LSTM = {'window_size': [idx_window_size],
                  'n_state_units': [32, 128],
                  'activation': ['tanh', 'sigmoid', 'relu'],
                  'optimizer': ['rmsprop', 'Adam', 'SGD']}

    param_grid_stacked_LSTM = {'window_size': [idx_window_size],
                                'n_state_units': [32],
                                'activation_1': ['sigmoid', 'relu'],
                                'activation_2': ['tanh', 'sigmoid', 'relu'],
                                'optimizer': ['rmsprop', 'Adam', 'SGD']}
    model = KerasClassifier(build_fn=create_model, verbose=True)

#     model = KerasClassifier(build_fn=vanilla_LSTM,
#                             epochs=epochs,
#                             batch_size=300,
#                             verbose=True)

    grid = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid_create_model,
                              n_iter=n_iter,
                              cv=cv,
                              random_state=42,
                              n_jobs=n_jobs,
                              verbose=1)

    key_name_X = "X_"
    key_name_y = "y_"

    key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
    key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)

    # remove [:10000], when real training
    X = Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[0][
        :dataset_scale]
    y = Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[1][
        :dataset_scale]

    y_single = {}
    #     print("[INFO] y : {}".format(y))
    #     y = np.asarray(y[0])
    #     print("[INFO] y.shape : {}".format(y.shape))
    #     print("[INFO] y : {}".format(y))
    y_single['BTC'] = y[:, 1]
    y_single['ETH'] = y[:, 2]
    y_single['XRP'] = y[:, 3]
    y_single['BCH'] = y[:, 4]
    y_single['LTC'] = y[:, 5]
    y_single['DASH'] = y[:, 6]
    y_single['ETC'] = y[:, 7]

    coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

    for coin in coin_list2:
        if (os.path.isfile(pickle_result_dir_path + \
                           coin + "_" + \
                           str(idx_time_unit) + "_" + \
                           str(idx_window_size) + "_" + \
                           str(idx_gap) + "_" + \
                           str(idx_margin_rate) + \
                           "_param.pickle")) is True:
            print(coin + "_" + \
                  str(idx_time_unit) + "_" + \
                  str(idx_window_size) + "_" + \
                  str(idx_gap) + "_" + \
                  str(idx_margin_rate) + \
                  "_param.pickle FILE ALREADY EXIST.")
            continue
        elif (os.path.isfile(pickle_result_dir_path + \
                             coin + "_" + \
                             str(idx_time_unit) + "_" + \
                             str(idx_window_size) + "_" + \
                             str(idx_gap) + "_" + \
                             str(idx_margin_rate) + \
                             "_result.pickle")) is True:
            print(coin + "_" + \
                  str(idx_time_unit) + "_" + \
                  str(idx_window_size) + "_" + \
                  str(idx_gap) + "_" + \
                  str(idx_margin_rate) + \
                  "_result.pickle FILE ALREADY EXIST.")
            continue
        else:
            y2 = onehottify(y_single[coin], n=2)

            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y2,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=True)

            print("[INFO] X_train.shape : {}".format(X_train.shape))
            print("[INFO] y_train.shape : {}".format(y_train.shape))
            print("[INFO] X_test.shape : {}".format(X_test.shape))
            print("[INFO] y_test.shape : {}".format(y_test.shape))
            print()

            n_coins = 8
            n_price = 4
            n_steps = idx_window_size

            X_train_2 = X_train.transpose([0, 2, 1, 3])
            X_test_2 = X_test.transpose([0, 2, 1, 3])
            print("[INFO] X_train_2.shape: {}".format(X_train_2.shape))
            print("[INFO] X_test_2.shape: {}".format(X_test_2.shape))
            print()

            X_train_3 = X_train_2.reshape([X_train.shape[0], n_steps, n_coins * n_price])
            X_test_3 = X_test_2.reshape([X_test.shape[0], n_steps, n_coins * n_price])
            print("[INFO] X_train_3.shape: {}".format(X_train_3.shape))
            print("[INFO] X_test_3.shape: {}".format(X_test_3.shape))
            print()

            X_train_reshape = X_train_2.reshape([X_train.shape[0], n_steps * n_coins * n_price])
            X_test_reshape = X_test_2.reshape([X_test.shape[0], n_steps * n_coins * n_price])
            print("[INFO] X_train_reshape.shape: {}".format(X_train_reshape.shape))
            print("[INFO] X_test_reshape.shape: {}".format(X_test_reshape.shape))
            print()

            scaler.fit(X_train_reshape)

            # X_train_scaled = scaler.transform(X_train_reshape)
            # X_test_scaled = scaler.transform(X_test_reshape)
            X_train_scaled = scaler.fit_transform(X_train_reshape)
            X_test_scaled = scaler.fit_transform(X_test_reshape)

            X_train_scaled = X_train_scaled.reshape(-1,
                                                    n_steps,
                                                    n_coins * n_price)
            X_test_scaled = X_test_scaled.reshape(-1,
                                                  n_steps,
                                                  n_coins * n_price)


            # model = KerasClassifier(build_fn=stacked_LSTM,
            #                         epochs=epochs,
            #                         batch_size=300,
            #                         verbose=True)


            #             grid = GridSearchCV(estimator=model,
            #                                 cv=cv,
            #                                 n_jobs=n_jobs, # test
            #                                 param_grid=param_grid,
            #                                 verbose=1)

            # grid = RandomizedSearchCV(estimator=model,
            #                           param_distributions=param_grid,
            #                           n_iter=n_iter,
            #                           cv=cv,
            #                           random_state=42,
            #                           n_jobs=n_jobs,
            #                           verbose=1)

            X_train_scaled, X_test_scaled = input_reshape(X_train_scaled,
                                                          X_test_scaled,
                                                          n_steps,
                                                          n_coins,
                                                          n_price)

            print()
            print()
            print("----------------------")
            print("__" + coin + "__" + \
                  "time unit: " + str(idx_time_unit) + "  |  " + \
                  "window_size :" + str(idx_window_size) + "  |  " + \
                  "gap :" + str(idx_gap) + "  |  " + \
                  "margin_rate :" + str(idx_margin_rate) + \
                  "  started.")

            #             fit_params = dict(callbacks=[stopper])

            with K.tf.device('/cpu:0'):
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
            #     print()
            #     print("evaluate result dict: ", evaluate_result)
            #     print()

            # summarize results
            print()
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            print()
            # for checking pickle file exist
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            print("---pickle saving..")

            X = {}
            y = {}
            key_name_X = "X_"
            key_name_y = "y_"

            key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(
                idx_margin_rate)
            key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(
                idx_margin_rate)
            if (os.path.isfile(pickle_result_dir_path + \
                               coin + "_" + \
                               str(idx_time_unit) + "_" + \
                               str(idx_window_size) + "_" + \
                               str(idx_gap) + "_" + \
                               str(idx_margin_rate) + \
                               "_param.pickle")) is not True:
                with open(pickle_result_dir_path + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate) + \
                          "_param.pickle", 'wb') as handle:
                    pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(pickle_result_for_email_transfer + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate) + \
                          "_param.pickle", 'wb') as handle:
                    pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    sending_email_for_file_transfer(pickle_result_for_email_transfer + \
                                                    coin + "_" + \
                                                    str(idx_time_unit) + "_" + \
                                                    str(idx_window_size) + "_" + \
                                                    str(idx_gap) + "_" + \
                                                    str(idx_margin_rate) + \
                                                    "_param.pickle", pickle_result_for_email_transfer)
                    os.remove(pickle_result_for_email_transfer + \
                              coin + "_" + \
                              str(idx_time_unit) + "_" + \
                              str(idx_window_size) + "_" + \
                              str(idx_gap) + "_" + \
                              str(idx_margin_rate) + \
                              "_param.pickle")

            return grid_result

if __name__ == "__main__":
    Start_Model()

    # dir = "./evaluate_result/"
    # dir = dir + model_info_2 + '/'
    # time_unit_list = [10]
    # window_size_list = [10, 50, 75, 100]
    #
    # for time_unit in time_unit_list:
    #     # dir = dir + str(time_unit) + '/'
    #     dir = "./evaluate_result/"
    #     print(dir)
    #     drawGraph(dir)
