
# import library
import tensorflow as tf
import keras
from keras import __version__
from keras import backend as K
print('Using Keras version:', __version__, 'backend:', K.backend())
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
# https://stackoverflow.com/questions/50065484/getting-precision-recall-and-f1-score-per-class-in-keras
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import sys
import os.path
from os.path import isfile, join
import boto3
import pickle
import time
# from xgboost import XGBClassifier
import warnings
# from link_aws_key import *
# pp = pprint.PrettyPrinter(indent=4)

stopper = EarlyStopping(monitor='loss', patience=3, verbose=1, min_delta=0.00001)
sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')

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
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

        # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def f1_score_(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def _f1_score(y_true, y_pred):
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


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def Start_Model(pickle_load_dir_path,
                pickle_result_dir_path,
                idx_time_unit,
                idx_window_size,
                idx_gap,
                idx_margin_rate,
                epochs,
                MODEL,
                _GPU,
                n_jobs,
                cv,
                dataset_scale,
                param_grid,
                search_param,
                machine,
                n_iter):
    X = {}
    y = {}

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
                           MODEL + "_" + \
                           coin + "_" + \
                           str(idx_time_unit) + "_" + \
                           str(idx_window_size) + "_" + \
                           str(idx_gap) + "_" + \
                           str(idx_margin_rate) + \
                           "_param.pickle")) is True:
            print(MODEL + "_" + \
                  coin + "_" + \
                  str(idx_time_unit) + "_" + \
                  str(idx_window_size) + "_" + \
                  str(idx_gap) + "_" + \
                  str(idx_margin_rate) + \
                  "_param.pickle FILE ALREADY EXIST.")
            continue
        elif (os.path.isfile(pickle_result_dir_path + \
                             MODEL + "_" + \
                             coin + "_" + \
                             str(idx_time_unit) + "_" + \
                             str(idx_window_size) + "_" + \
                             str(idx_gap) + "_" + \
                             str(idx_margin_rate) + \
                             "_result.pickle")) is True:
            print(MODEL + "_" + \
                  coin + "_" + \
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
                                                                test_size=0.1,
                                                                random_state=42)
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

            param_grid = param_grid

            scaler = MinMaxScaler()
            scaler.fit(X_train_reshape)

            X_train_scaled = scaler.transform(X_train_reshape)
            X_test_scaled = scaler.transform(X_test_reshape)

            X_train_scaled = X_train_scaled.reshape(-1,
                                                    n_steps,
                                                    n_coins * n_price)
            X_test_scaled = X_test_scaled.reshape(-1,
                                                  n_steps,
                                                  n_coins * n_price)

            if _GPU == True:
                model = KerasClassifier(build_fn=create_model_LSTM,
                                        epochs=epochs,
                                        batch_size=100,
                                        verbose=True)

            elif _GPU == False:
                model = KerasClassifier(build_fn=create_model_LSTM_non_GPU,
                                        epochs=epochs,
                                        batch_size=100,
                                        verbose=True)

            #             grid = GridSearchCV(estimator=model,
            #                                 cv=cv,
            #                                 n_jobs=n_jobs, # test
            #                                 param_grid=param_grid,
            #                                 verbose=1)

            grid = RandomizedSearchCV(estimator=model,
                                      param_distributions=param_grid,
                                      n_iter=n_iter,
                                      cv=cv,
                                      random_state=42,
                                      n_jobs=n_jobs,
                                      verbose=1)

            X_train_scaled, X_test_scaled = input_reshape(X_train_scaled,
                                                          X_test_scaled,
                                                          n_steps,
                                                          n_coins,
                                                          n_price)

            print()
            print()
            print("----------------------")
            print("<" + MODEL + ">")
            print("----------------------")
            print("__" + coin + "__" + \
                  "time unit: " + str(idx_time_unit) + "  |  " + \
                  "window_size :" + str(idx_window_size) + "  |  " + \
                  "gap :" + str(idx_gap) + "  |  " + \
                  "margin_rate :" + str(idx_margin_rate) + \
                  "  started.")

            #             fit_params = dict(callbacks=[stopper])
            grid_result = grid.fit(X_train_scaled,
                                   y_train,
                                   validation_data=(X_test_scaled,
                                                    y_test),
                                   callbacks=[stopper])

            print("----------------------")
            print("grid_result.score(X_test_scaled, y_test): ", grid_result.score(X_test_scaled, y_test))

            evaluate_result = {}
            test_score = grid_result.score(X_test_scaled, y_test)
            evaluate_result[MODEL + "_" + \
                            coin + "_" + \
                            str(idx_time_unit) + "_" + \
                            str(idx_window_size) + "_" + \
                            str(idx_gap) + "_" + \
                            str(idx_margin_rate)] = {"MODEL": MODEL, \
                                                     "Cryptocurrency": coin, \
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
                               MODEL + "_" + \
                               coin + "_" + \
                               str(idx_time_unit) + "_" + \
                               str(idx_window_size) + "_" + \
                               str(idx_gap) + "_" + \
                               str(idx_margin_rate) + \
                               "_param.pickle")) is not True:
                with open(pickle_result_dir_path + \
                          MODEL + "_" + \
                          coin + "_" + \
                          str(idx_time_unit) + "_" + \
                          str(idx_window_size) + "_" + \
                          str(idx_gap) + "_" + \
                          str(idx_margin_rate) + \
                          "_param.pickle", 'wb') as handle:
                    pickle.dump(evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

            key_name_X = "X_"
            key_name_y = "y_"

            return grid_result


def create_model_LSTM(window_size,
                      units_1,
                      units_2,
                      n_state_units=32,
                      activation_1='softmax',
                      activation_2='relu',
                      optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        LSTM(units=n_state_units,
             input_shape=(window_size, 32)))

    model.add(Dense(units=units_1))
    #     model.add(Dropout(dropout_rate))
    model.add(Dense(units=units_1, activation=activation_1))
    model.add(Dense(units=units_1, activation=activation_2))
    model.add(Dense(units=units_2, activation=activation_2))
    model.add(Dense(2, activation=activation_2))

    #     model.add(Dense(units=2))
    # #     model.add(Dropout(dropout_rate))
    #     model = multi_gpu_model(model, gpus=2)
    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #                   optimizer=optimizer,
    #                   metrics=[_f1_score, 'accuracy'])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy'])

    return model


def create_model_LSTM_non_GPU(window_size,
                              units_1,
                              units_2,
                              n_state_units=32,
                              activation_1='softmax',
                              activation_2='relu',
                              optimizer='adam'):
    #     global metrics
    K.clear_session()
    model = Sequential()
    model.add(
        LSTM(units=n_state_units,
             input_shape=(window_size, 32)))

    model.add(Dense(units=units_1))
    model.add(Dense(units=units_1, activation=activation_1))
    model.add(Dense(units=units_1, activation=activation_2))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=units_2, activation=activation_2))
    model.add(Dense(units=units_1, activation=activation_1))
    model.add(Dense(units=units_1, activation=activation_2))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=units_2, activation=activation_2))
    model.add(Dense(units=units_1, activation=activation_1))
    model.add(Dense(units=units_1, activation=activation_2))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=units_2, activation=activation_2))
    model.add(Dense(units=units_1, activation=activation_1))
    model.add(Dense(units=units_1, activation=activation_2))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=units_2, activation=activation_2))
    model.add(Dense(2, activation=activation_2))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=[_f1_score, 'accuracy'])

    return model


def start(machine,
          search_param,
          _GPU,
          n_jobs,
          MODEL,
          idx_time_unit,
          idx_window_size,
          idx_gap,
          idx_margin_rate,
          cv,
          dataset_scale,
          param_grid,
          epochs,
          dataset_file_dir,
          pickle_save_folder,
          #           scoring,
          n_iter):
    '''
        [ATTENTION] In create_model METHOD part, need to set appropriate about GPU

        LINK01 -> GPU OFF
        MSI -> GPU OFF
        SLAVE04 -> GPU ON
        SLAVE05 -> GPU ON
    '''

    start_time = time.time()
    grid_result = Start_Model(pickle_load_dir_path=dataset_file_dir,
                              pickle_result_dir_path=pickle_save_folder,
                              idx_time_unit=idx_time_unit,
                              idx_window_size=idx_window_size,
                              idx_gap=idx_gap,
                              idx_margin_rate=idx_margin_rate,
                              epochs=epochs,
                              MODEL=MODEL,
                              _GPU=_GPU,
                              n_jobs=n_jobs,
                              cv=cv,
                              dataset_scale=dataset_scale,
                              param_grid=param_grid,
                              search_param=search_param,
                              machine=machine,
                              n_iter=n_iter)
    #     print("DEBUG")
    end_time = time.time()
    print()
    print("TIME: ", end_time - start_time)
    return grid_result


class SortedDisplayDict(dict):
    def __str__(self):
        return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

    def ordered_keys(self):
        return sorted(self.keys())


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

def run_Model(run_test, epochs, cv, n_iter, _GPU, model_info_):
    if run_test == True:
        model_info = {}
        if model_info_ == "window_size_standard":
            model_info[model_info_] = {"MODEL_list":["LSTM"],
                                      "time_unit":[10],
                                      "window_size":[10, 50, 75, 100],
                                      "gap":[1],
                                      "margin_rate":[0.1],
                                      "lr":[0.01, 0.001]}
        elif model_info == "time_unit_standard":
            model_info[model_info_] = {"MODEL_list":["LSTM"],
                                      "time_unit":[10,30,60],
                                      "window_size":[25],
                                      "gap":[1],
                                      "margin_rate":[0.1],
                                      "lr":[0.01, 0.001]}

        Machine = model_info_
        _GPU = _GPU
        n_jobs = 1
        epochs = 1
        cv = 2 # at least 2
        dataset_scale = -1
        test_list = []
        coin_list = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
        search_param = True

        for model in model_info[Machine]["MODEL_list"]:
            for idx_time_unit in model_info[Machine]["time_unit"]:
                for idx_window_size in model_info[Machine]["window_size"]:
                    for idx_gap in model_info[Machine]["gap"]:
                        for idx_margin_rate in model_info[Machine]["margin_rate"]:
                            param_grid = {'window_size': [idx_window_size],
                                          'units_1': [16, 32, 64, 128],
                                          'units_2': [16, 32, 64, 128],
                                          'n_state_units': [32, 64, 128],
                                          'activation_1': ['tanh', 'sigmoid', 'relu'],
                                          'activation_2': ['softmax', 'tanh', 'sigmoid', 'relu'],
                                          'optimizer': ['rmsprop', 'Adam', 'SGD']}

                            try:
                                test_list.append(start(machine=Machine,
                                                       search_param=search_param,
                                                       _GPU=_GPU,
                                                       n_jobs=n_jobs,
                                                       MODEL=model,
                                                       idx_time_unit=idx_time_unit,
                                                       idx_window_size=idx_window_size,
                                                       idx_gap=idx_gap,
                                                       idx_margin_rate=idx_margin_rate,
                                                       cv=cv,
                                                       dataset_scale=dataset_scale,
                                                       param_grid=param_grid,
                                                       epochs=epochs,
                                                       dataset_file_dir='../RNN_coin/',
                                                       pickle_save_folder='./evaluate_result/',
                                                       n_iter=1))

                            except KeyError or ValueError:
                                # print("[INFO] Appropriate value of {:4s} is not exist.".format(coin))
                                continue
    elif run_test == False:
        model_info = {}

        if model_info_ == "window_size_standard":
            model_info[model_info_] = {"MODEL_list":["LSTM"],
                                      "time_unit":[10],
                                      "window_size":[10, 50, 75, 100],
                                      "gap":[1],
                                      "margin_rate":[0.1],
                                      "lr":[0.01, 0.001]}
        elif model_info_ == "time_unit_standard":
            model_info[model_info_] = {"MODEL_list":["LSTM"],
                                      "time_unit":[10,30,60],
                                      "window_size":[25],
                                      "gap":[1],
                                      "margin_rate":[0.1],
                                      "lr":[0.01, 0.001]}

        Machine = model_info_
        _GPU = _GPU
        n_jobs = 1
        epochs = epochs
        cv = cv  # at least 2
        dataset_scale = -1
        test_list = []
        coin_list = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
        search_param = True

        for model in model_info[Machine]["MODEL_list"]:
            for idx_time_unit in model_info[Machine]["time_unit"]:
                for idx_window_size in model_info[Machine]["window_size"]:
                    for idx_gap in model_info[Machine]["gap"]:
                        for idx_margin_rate in model_info[Machine]["margin_rate"]:
                            param_grid = {'window_size': [idx_window_size],
                                          'units_1': [16, 32, 64, 128],
                                          'units_2': [16, 32, 64, 128],
                                          'n_state_units': [32, 64, 128],
                                          'activation_1': ['tanh', 'sigmoid', 'relu'],
                                          'activation_2': ['softmax', 'tanh', 'sigmoid', 'relu'],
                                          'optimizer': ['rmsprop', 'Adam', 'SGD']}

                            try:
                                test_list.append(start(machine=Machine,
                                                       search_param=search_param,
                                                       _GPU=_GPU,
                                                       n_jobs=n_jobs,
                                                       MODEL=model,
                                                       idx_time_unit=idx_time_unit,
                                                       idx_window_size=idx_window_size,
                                                       idx_gap=idx_gap,
                                                       idx_margin_rate=idx_margin_rate,
                                                       cv=cv,
                                                       dataset_scale=dataset_scale,
                                                       param_grid=param_grid,
                                                       epochs=epochs,
                                                       dataset_file_dir='../data/RNN_coin/',
                                                       pickle_save_folder='./evaluate_result/',
                                                       n_iter=n_iter))

                            except KeyError or ValueError:
                                # print("[INFO] Appropriate value of {:4s} is not exist.".format(coin))
                                continue

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