"""
Precision metric.

Only computes a batch-wise average of precision.

Computes the precision, a metric for multi-label classification of
how many selected items are relevant.
"""

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
from xgboost import XGBClassifier
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
class LSTM:
    def __init__(self):
        self.key_name_X = None
        self.key_name_y = None
        self.b_x = None
        self.b_y = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.X_train_reshape = None
        self.X_test_reshape = None
        self.x = None
        self.n = None
        self.start_time = None
        self.end_time = None
        self.thour = None
        self.temp_sec = None
        self.tmin = None
        self.tsec = None
        self.X = {}
        self.y = {}
        self.y_single = {}
        self.coin_list2 = None
        self.y2 = None
        self.X_train = None
        self.X_train_2 = None
        self.X_train_3 = None
        self.X_train_reshape = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_2 = None
        self.X_test_3 = None
        self.X_test_reshape = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.n_coins = None
        self.n_price = None
        self.n_steps = None
        self.param_grid = None
        self.scaler = None
        self.model = None
        self.grid = None
        self.grid_result = None
        self.evaluate_result = {}
        self.test_score = None
        self.xticklabels = None
        self.yticklabels = None
        self.ind = None
        self.fig = None
        self.axes = None
        self.subfigures = {}
        self.data_name = None
        self.cluster_coef_value_list_gradientBoosting = None
        self.cluster_coef_value_list_lstm = None
        self.diameter_dic = None
        self.model_info = {}
        self.Machine = None
        self._GPU = None
        self.n_jobs = None
        self.epochs = None
        self.cv = None
        self.dataset_scale = None
        self.test_list = []
        self.coin_list = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
        self.search_param = None
        self.filenames = None
        self.fileList = []
        self.temp_list = []
        self.cluster_coef_value_list_lstm = []
        self.first_legend_label = 'xgboost'
        self.second_legend_label = 'lstm'
        self.x_label = 'Cryptocurrency'
        self.y_label = 'f1-score'
        self.filename = None


    def Load_Dataset_X(self, dir_path, time_unit, window_size, gap, margin_rate):
        self.key_name_X = "X_" + \
                     str(time_unit) + "_" + \
                     str(window_size) + "_" + \
                     str(gap) + "_" + \
                     str(margin_rate)

        with open(dir_path + self.key_name_X + ".pickle", 'rb') as handle:
            self.b_x = pickle.load(handle)
        return self.b_x


    def Load_Dataset_y(self, dir_path, time_unit, window_size, gap, margin_rate):
        self.key_name_y = "y_" + \
                     str(time_unit) + "_" + \
                     str(window_size) + "_" + \
                     str(gap) + "_" + \
                     str(margin_rate)

        with open(dir_path + self.key_name_y + ".pickle", 'rb') as handle:
            self.b_y = pickle.load(handle)
        return self.b_y


    def f1_score(self, y_true, y_pred):
        # Count positive samples.
        self.c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        self.c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        self.c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if self.c3 == 0:
            return 0

            # How many selected items are relevant?
            self.precision = self.c1 / self.c2

        # How many relevant items are selected?
            self.recall = self.c1 / self.c3

        # Calculate f1_score
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self.f1_score


    def f1_score_(self, y_true, y_pred):
        # Count positive samples.
        self.c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        self.c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        self.c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if self.c3 == 0:
            return 0

        # How many selected items are relevant?
        self.precision = self.c1 / self.c2

        # How many relevant items are selected?
        self.recall = self.c1 / self.c3

        # Calculate f1_score
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self.f1_score


    def _f1_score(self, y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            self.true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            self.possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            self.recall = self.true_positives / (self.possible_positives + K.epsilon())
            return self.recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            self.true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            self.predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            self.precision = self.true_positives / (self.predicted_positives + K.epsilon())
            return precision

        self.precision = self.precision(y_true, y_pred)
        self.recall = self.recall(y_true, y_pred)
        return 2 * ((self.precision * self.recall) / (self.precision + self.recall + K.epsilon()))


    def input_reshape(self, X_train_data, X_test_data, n_steps, n_coins, n_price):
        self.X_train_reshape = X_train_data.reshape(
            -1,
            n_steps,
            n_coins * n_price
        )

        self.X_test_reshape = X_test_data.reshape(
            -1,
            n_steps,
            n_coins * n_price
        )
        return self.X_train_reshape, self.X_test_reshape


    def onehottify(self, x, n=None, dtype=np.int):
        """1-hot encode x with the max value n (computed from data if n is None)."""
        self.x = np.asarray(x)
        self.n = np.max(self.x) + 1 if n is None else n
        return np.eye(self.n, dtype=dtype)[self.x]


    def timer(self, start_time=None):
        if not start_time:
            self.start_time = datetime.now()
            return self.start_time
        elif start_time:
            self.thour, self.temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            self.tmin, self.tsec = divmod(self.temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (self.thour, self.tmin, round(self.tsec, 2)))


    def Start_Model(self, pickle_load_dir_path,
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

        # self.X = {}
        # self.y = {}
        self.key_name_X = "X_"
        self.key_name_y = "y_"

        self.key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)
        self.key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(idx_margin_rate)

        # remove [:10000], when real training
        self.X = self.Load_Dataset_X(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[0][
            :dataset_scale]
        self.y = self.Load_Dataset_y(pickle_load_dir_path, idx_time_unit, idx_window_size, idx_gap, idx_margin_rate)[1][
            :dataset_scale]

        # self.y_single = {}
        #     print("[INFO] y : {}".format(y))
        #     y = np.asarray(y[0])
        #     print("[INFO] y.shape : {}".format(y.shape))
        #     print("[INFO] y : {}".format(y))
        self.y_single['BTC'] = self.y[:, 1]
        self.y_single['ETH'] = self.y[:, 2]
        self.y_single['XRP'] = self.y[:, 3]
        self.y_single['BCH'] = self.y[:, 4]
        self.y_single['LTC'] = self.y[:, 5]
        self.y_single['DASH'] = self.y[:, 6]
        self.y_single['ETC'] = self.y[:, 7]

        self.coin_list2 = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]

        for coin in self.coin_list2:
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
                self.y2 = self.onehottify(self.y_single[coin], n=2)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                    self.y2,
                                                                    test_size=0.1,
                                                                    random_state=42)
                print("[INFO] X_train.shape : {}".format(self.X_train.shape))
                print("[INFO] y_train.shape : {}".format(self.y_train.shape))
                print("[INFO] X_test.shape : {}".format(self.X_test.shape))
                print("[INFO] y_test.shape : {}".format(self.y_test.shape))
                print()

                self.n_coins = 8
                self.n_price = 4
                self.n_steps = idx_window_size

                self.X_train_2 = self.X_train.transpose([0, 2, 1, 3])
                self.X_test_2 = self.X_test.transpose([0, 2, 1, 3])
                print("[INFO] X_train_2.shape: {}".format(self.X_train_2.shape))
                print("[INFO] X_test_2.shape: {}".format(self.X_test_2.shape))
                print()
                self.X_train_3 = self.X_train_2.reshape([self.X_train.shape[0], self.n_steps, self.n_coins * self.n_price])
                self.X_test_3 = self.X_test_2.reshape([self.X_test.shape[0], self.n_steps, self.n_coins * self.n_price])
                print("[INFO] X_train_3.shape: {}".format(self.X_train_3.shape))
                print("[INFO] X_test_3.shape: {}".format(self.X_test_3.shape))
                print()

                self.X_train_reshape = self.X_train_2.reshape([self.X_train.shape[0], self.n_steps * self.n_coins * self.n_price])
                self.X_test_reshape = self.X_test_2.reshape([self.X_test.shape[0], self.n_steps * self.n_coins * self.n_price])
                print("[INFO] X_train_reshape.shape: {}".format(self.X_train_reshape.shape))
                print("[INFO] X_test_reshape.shape: {}".format(self.X_test_reshape.shape))
                print()
                self.param_grid = param_grid

                self.scaler = MinMaxScaler()
                self.scaler.fit(self.X_train_reshape)

                self.X_train_scaled = self.scaler.transform(self.X_train_reshape)
                self.X_test_scaled = self.scaler.transform(self.X_test_reshape)

                self.X_train_scaled = self.X_train_scaled.reshape(-1,
                                                                  self.n_steps,
                                                                  self.n_coins * self.n_price)
                self.X_test_scaled = self.X_test_scaled.reshape(-1,
                                                                self.n_steps,
                                                                self.n_coins * self.n_price)

                if _GPU == True:
                    self.model = KerasClassifier(build_fn=self.create_model_LSTM,
                                            epochs=epochs,
                                            batch_size=100,
                                            verbose=True)

                elif _GPU == False:
                    self.model = KerasClassifier(build_fn=self.create_model_LSTM_non_GPU,
                                            epochs=epochs,
                                            batch_size=100,
                                            verbose=True)

                #             grid = GridSearchCV(estimator=model,
                #                                 cv=cv,
                #                                 n_jobs=n_jobs, # test
                #                                 param_grid=param_grid,
                #                                 verbose=1)

                    self.grid = RandomizedSearchCV(estimator=self.model,
                                          param_distributions=param_grid,
                                          n_iter=n_iter,
                                          cv=cv,
                                          random_state=42,
                                          n_jobs=n_jobs,
                                          verbose=1)

                    self.X_train_scaled, self.X_test_scaled = self.input_reshape(self.X_train_scaled,
                                                                                 self.X_test_scaled,
                                                                                 self.n_steps,
                                                                                 self.n_coins,
                                                                                 self.n_price)


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
                self.grid_result = self.grid.fit(self.X_train_scaled,
                                                 self.y_train,
                                       validation_data=(self.X_test_scaled,
                                                        self.y_test),
                                       callbacks=[stopper])

                print("----------------------")
                print("grid_result.score(X_test_scaled, y_test): ", self.grid_result.score(self.X_test_scaled, self.y_test))

                # self.evaluate_result = {}
                self.test_score = self.grid_result.score(self.X_test_scaled, self.y_test)
                self.evaluate_result[MODEL + "_" + \
                                coin + "_" + \
                                str(idx_time_unit) + "_" + \
                                str(idx_window_size) + "_" + \
                                str(idx_gap) + "_" + \
                                str(idx_margin_rate)] = {"MODEL": MODEL, \
                                                         "Cryptocurrency": coin, \
                                                         "Score": self.grid_result.cv_results_['mean_test_score'], \
                                                         "Params": self.grid_result.cv_results_['params'], \
                                                         "test_score": self.test_score}
                #     print()
                #     print("evaluate result dict: ", evaluate_result)
                #     print()

                # summarize results
                print()
                print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))
                print()
                # for checking pickle file exist
                print("---pickle saving..")

                self.X = {}
                self.y = {}
                self.key_name_X = "X_"
                self.key_name_y = "y_"

                self.key_name_X += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(
                    idx_margin_rate)
                self.key_name_y += str(idx_time_unit) + "_" + str(idx_window_size) + "_" + str(idx_gap) + "_" + str(
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
                        pickle.dump(self.evaluate_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

                key_name_X = "X_"
                key_name_y = "y_"

                return self.grid_result


    def create_model_LSTM(self, window_size,
                          units_1,
                          units_2,
                          n_state_units=32,
                          activation_1='softmax',
                          activation_2='relu',
                          optimizer='adam'):
        #     global metrics
        K.clear_session()
        self.model = Sequential()
        self.model.add(
            LSTM(units=n_state_units,
                 input_shape=(window_size, 32)))

        self.model.add(Dense(units=units_1))
        #     model.add(Dropout(dropout_rate))
        self.model.add(Dense(units=units_1, activation=activation_1))
        self.model.add(Dense(units=units_1, activation=activation_2))
        self.model.add(Dense(units=units_2, activation=activation_2))
        self.model.add(Dense(2, activation=activation_2))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=[self._f1_score, 'accuracy'])
        #     model.add(Dense(units=2))
        # #     model.add(Dropout(dropout_rate))
        #     model = multi_gpu_model(model, gpus=2)
        #     model.compile(loss=keras.losses.categorical_crossentropy,
        #                   optimizer=optimizer,
        #                   metrics=[_f1_score, 'accuracy'])

        return self.model


    def create_model_LSTM_non_GPU(self,
                                  window_size,
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
                          metrics=[self._f1_score, 'accuracy'])

        return model


    def start(self, machine,
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

        self.start_time = time.time()
        self.grid_result = self.Start_Model(pickle_load_dir_path=dataset_file_dir,
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
        self.end_time = time.time()
        print()
        print("TIME: ", self.end_time - self.start_time)
        return self.grid_result



    def graph(self,
              cluster_coef_value_list_gradientBoosting,
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
        link_addition_ratio = self.f1_score
        self.data_profile = {
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

        self.xticklabels = [r'$BTC$', r'$ETH$', r'$XRP$', r'$BCH$', r'$LTC$', r'$DASH$', r'$ETC$']
        self.yticklabels = [r'$0.40$', r'$0.45$', r'$0.50$', r'$0.55$', r'$0.60$', r'$0.65$', r'$0.70$', r'$0.75$', r'$0.80$']
        # yticklabels2 = [r'$0$', r'$2$', r'$4$', r'$6$', r'$8$', r'$10$', r'$12$']

        self.ind = np.arange(len(self.xticklabels))

        self.fig, self.axes = plt.subplots(1, 1, figsize=(8.719, 6.07))
        # self.subfigures = {}
        self.data_name = 'Infocom05'
        self.subfigures[self.data_name] = self.axes
        self.subfigures[self.data_name].set_xticks(self.ind)
        self.subfigures[self.data_name].set_xticklabels(self.xticklabels, fontsize=21)
        self.subfigures[self.data_name].set_ylim([0.4, 0.8])
        self.subfigures[self.data_name].set_yticklabels(self.yticklabels, fontsize=21)

        self.cluster_coef_dic = SortedDisplayDict(self.data_profile[self.data_name]['clustering_coefficient'])
        cluster_coef_value_list = [self.cluster_coef_dic[x] for x in self.cluster_coef_dic.ordered_keys()]

        # cluster_coef_value_list_xgboost = [0.56, 0.67, 0.66, 0.78, 0.45, 0.47, 0.65]
        # cluster_coef_value_list_lstm = [0.66, 0.77, 0.68, 0.68, 0.55, 0.57, 0.66]
        self.cluster_coef_value_list_gradientBoosting = cluster_coef_value_list_gradientBoosting
        self.cluster_coef_value_list_lstm = cluster_coef_value_list_lstm
        self.subfigures[self.data_name].plot(self.ind, self.cluster_coef_value_list_gradientBoosting,
                                   color='k', linestyle='-', marker='s', markersize=8,
                                   label=first_legend_label)
        self.subfigures[self.data_name].plot(self.ind, cluster_coef_value_list_lstm,
                                   color='k', linestyle='--', marker='^', markersize=8,
                                   label=second_legend_label)
        self.subfigures[self.data_name].set_xlabel(x_label, fontsize=21)
        self.subfigures[self.data_name].set_ylabel(y_label, fontsize=21)

        # subfigures[data_name].set_title(data_name, fontsize=21)
        self.subfigures[self.data_name].grid(True)
        if self.data_name == 'Infocom05':
            self.subfigures[self.data_name].legend(loc=30, fontsize=18)

        # subfigures[data_name] = subfigures[data_name].twinx()
        # subfigures[data_name].set_ylim([0, 12])
        # subfigures[data_name].set_yticklabels(yticklabels2, fontsize=21)

        self.diameter_dic = SortedDisplayDict(self.data_profile[self.data_name]['diameter_cc'])
        diameter_list = [self.diameter_dic[x] for x in self.diameter_dic.ordered_keys()]
        # subfigures[data_name].bar(ind, diameter_list, barWidth, color='k', alpha=0.3, label='Diameter of Connected Component')
        # subfigures[data_name].set_ylabel('Diameter of Connected Component', fontsize=21)
        if self.data_name == 'Infocom05':
            self.subfigures[self.data_name].legend(loc=4, fontsize=18)
            self.subfigures[self.data_name].grid(True)

        self.fig.savefig('./img/' + filename + '.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def run_Model(self, run_test, epochs, cv, n_iter, _GPU, model_info_):
        if run_test == True:
            # self.model_info = {}
            if model_info_ == "window_size_standard":
                self.model_info[model_info_] = {"MODEL_list":["LSTM"],
                                          "time_unit":[10],
                                          "window_size":[10, 50, 75, 100],
                                          "gap":[1],
                                          "margin_rate":[0.1],
                                          "lr":[0.01, 0.001]}
            elif model_info_ == "time_unit_standard":
                self.model_info[model_info_] = {"MODEL_list":["LSTM"],
                                          "time_unit":[10,30,60],
                                          "window_size":[25],
                                          "gap":[1],
                                          "margin_rate":[0.1],
                                          "lr":[0.01, 0.001]}

            self.Machine = model_info_
            self._GPU = _GPU
            self.n_jobs = 1
            self.epochs = 1
            self.cv = 2 # at least 2
            self.dataset_scale = -1
            # self.test_list = []
            # self.coin_list = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
            self.search_param = True

            for model in self.model_info[self.Machine]["MODEL_list"]:
                for idx_time_unit in self.model_info[self.Machine]["time_unit"]:
                    for idx_window_size in self.model_info[self.Machine]["window_size"]:
                        for idx_gap in self.model_info[self.Machine]["gap"]:
                            for idx_margin_rate in self.model_info[self.Machine]["margin_rate"]:
                                self.param_grid = {'window_size': [idx_window_size],
                                                  'units_1': [16, 32, 64, 128],
                                                  'units_2': [16, 32, 64, 128],
                                                  'n_state_units': [32, 64, 128],
                                                  'activation_1': ['tanh', 'sigmoid', 'relu'],
                                                  'activation_2': ['softmax', 'tanh', 'sigmoid', 'relu'],
                                                  'optimizer': ['rmsprop', 'Adam', 'SGD']}

                                try:
                                    self.test_list.append(self.start(machine=self.Machine,
                                                           search_param=self.search_param,
                                                           _GPU=self._GPU,
                                                           n_jobs=self.n_jobs,
                                                           MODEL=model,
                                                           idx_time_unit=idx_time_unit,
                                                           idx_window_size=idx_window_size,
                                                           idx_gap=idx_gap,
                                                           idx_margin_rate=idx_margin_rate,
                                                           cv=self.cv,
                                                           dataset_scale=self.dataset_scale,
                                                           param_grid=self.param_grid,
                                                           epochs=self.epochs,
                                                           dataset_file_dir='./data/RNN_coin/',
                                                           pickle_save_folder='./evaluate_result/',
                                                           n_iter=1))

                                except KeyError or ValueError:
                                    # print("[INFO] Appropriate value of {:4s} is not exist.".format(coin))
                                    continue
        elif run_test == False:
            # self.model_info = {}

            if model_info_ == "window_size_standard":
                self.model_info[model_info_] = {"MODEL_list":["LSTM"],
                                          "time_unit":[10],
                                          "window_size":[10, 50, 75, 100],
                                          "gap":[1],
                                          "margin_rate":[0.1],
                                          "lr":[0.01, 0.001]}
            elif model_info_ == "time_unit_standard":
                self.model_info[model_info_] = {"MODEL_list":["LSTM"],
                                          "time_unit":[10,30,60],
                                          "window_size":[25],
                                          "gap":[1],
                                          "margin_rate":[0.1],
                                          "lr":[0.01, 0.001]}

            self.Machine = model_info_
            self._GPU = _GPU
            self.n_jobs = 1
            self.epochs = epochs
            self.cv = cv  # at least 2
            self.dataset_scale = -1
            # self.test_list = []
            # self.coin_list = ["BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
            self.search_param = True

            for model in self.model_info[self.Machine]["MODEL_list"]:
                for idx_time_unit in self.model_info[self.Machine]["time_unit"]:
                    for idx_window_size in self.model_info[self.Machine]["window_size"]:
                        for idx_gap in self.model_info[self.Machine]["gap"]:
                            for idx_margin_rate in self.model_info[self.Machine]["margin_rate"]:
                                self.param_grid = {'window_size': [idx_window_size],
                                                  'units_1': [16, 32, 64, 128],
                                                  'units_2': [16, 32, 64, 128],
                                                  'n_state_units': [32, 64, 128],
                                                  'activation_1': ['tanh', 'sigmoid', 'relu'],
                                                  'activation_2': ['softmax', 'tanh', 'sigmoid', 'relu'],
                                                  'optimizer': ['rmsprop', 'Adam', 'SGD']}

                                try:
                                    self.test_list.append(self.start(machine=self.Machine,
                                                           search_param=self.search_param,
                                                           _GPU=self._GPU,
                                                           n_jobs=self.n_jobs,
                                                           MODEL=model,
                                                           idx_time_unit=idx_time_unit,
                                                           idx_window_size=idx_window_size,
                                                           idx_gap=idx_gap,
                                                           idx_margin_rate=idx_margin_rate,
                                                           cv=self.cv,
                                                           dataset_scale=self.dataset_scale,
                                                           param_grid=self.param_grid,
                                                           epochs=self.epochs,
                                                           dataset_file_dir='./data/RNN_coin/',
                                                           pickle_save_folder='./evaluate_result/',
                                                           n_iter=n_iter))

                                except KeyError or ValueError:
                                    # print("[INFO] Appropriate value of {:4s} is not exist.".format(coin))
                                    continue

    def search(self, dirname):
        self.filenames = os.listdir(dirname)
        # self.fileList = []
        for filename in self.filenames:
            self.full_filename = os.path.join(dirname, filename)
            self.fileList.append(self.full_filename)
        return self.fileList


    def drawGraph(self, dir):
        self.fileList = self.search(dir)
        # self.temp_list = []
        # self.cluster_coef_value_list_lstm = []
        for file in self.fileList:
            self.temp_list.append(pd.read_pickle(file))

        for i in range(len(self.temp_list)):
            # print(temp_list[i])
            # print()
            # print(temp_list[i][list(temp_list[i].keys())[0]])
            # print()
            self.cluster_coef_value_list_lstm.append(max(self.temp_list[i][list(self.temp_list[i].keys())[0]]['Score']))
            # print(score)
            # print()

        self.cluster_coef_value_list_gradientBoosting = [0.671, 0.616, 0.622, 0.672, 0.7, 0.69, 0.69]

        # self.first_legend_label = 'xgboost'
        # self.second_legend_label = 'lstm'
        # self.x_label = 'Cryptocurrency'
        # self.y_label = 'f1-score'
        self.filename = str(datetime.now())

        self.graph(self.cluster_coef_value_list_gradientBoosting,
                   self.cluster_coef_value_list_lstm[:7],
                   self.first_legend_label,
                   self.second_legend_label,
                   self.x_label,
                   self.y_label,
                   self.filename)

class SortedDisplayDict(dict):
    def __str__(self):
        return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

    def ordered_keys(self):
        return sorted(self.keys())




if __name__ == "__main__":

    lstm = LSTM()

    run_test = True
    epochs = 300
    cv = 5
    n_iter = 20
    _GPU = False
    model_info_1 = "window_size_standard"
    model_info_2 = "time_unit_standard"

    lstm.run_Model(run_test=run_test,
                   epochs=epochs,
                   cv=cv,
                   n_iter=n_iter,
                   _GPU=_GPU,
                   model_info_=model_info_1)

    # dir = "./evaluate_result/"
    # dir = dir + model_info_2 + '/'
    # time_unit_list = [10]
    # window_size_list = [10, 50, 75, 100]
    #
    # for time_unit in time_unit_list:
    #     dir = dir + str(time_unit) + '/'
    #     print(dir)
    #     lstm.drawGraph(dir)
