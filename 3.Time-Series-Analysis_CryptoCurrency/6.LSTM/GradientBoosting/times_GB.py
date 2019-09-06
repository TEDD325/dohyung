from datetime import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys, os
import pprint
import boto3
import pickle
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import os.path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from os import listdir
from os.path import isfile, join
import csv
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib


sys.path.append(os.getcwd())
# from link_aws_key import *
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

# f = open('../../../../link_aws_key.bin', 'rb')
# a = pickle.load(f)
#
# aws_client = boto3.client(
#     's3',
#     aws_access_key_id=a["LINK_AWSAccessKeyId"],
#     aws_secret_access_key=a["LINK_AWSSecretKey"]
# )
bucket = "bithumb10"
cleanup_file_name = "coin_{0}_{1}_cleanup.csv"
pickle_result_dir_path = '../evaluate_result/eXtream_Gradient_Boosting/'

# def get_all_raw_data_from_aws(coin_name_list, start_date, end_date):
#     start_ms_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
#     end_ms_time = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
#
#     year_temp = start_date[:4]
#     years = [year_temp]
#     while year_temp < end_date[:4]:
#         year_temp = str(int(start_date[:4]) + 1)
#         years.append(year_temp)
#     raw_data = {}  # 전체 CSV Raw 데이터
#     for coin_name in coin_name_list:
#         raw_data[coin_name] = []
#
#     # KRW 제외한 나머지 CSV Raw 데이터 수집
#     for coin_name in coin_name_list:
#         if coin_name == 'KRW':
#             continue
#         lines = []
#         for year in years:
#             obj = aws_client.get_object(
#                 Bucket=bucket,
#                 Key='cleanup/' + year + '/' + cleanup_file_name.format(coin_name, year)
#             )
#             if lines != []:
#                 lines += obj.get('Body')._raw_stream.readlines()
#             else:
#                 lines = obj.get('Body')._raw_stream.readlines()
#
#         for line in lines:
#             line = str(line.strip())[2:-1]
#             line = line.split(',')
#             if start_ms_time <= int(line[0]) and int(line[0]) <= end_ms_time:
#                 raw_data[coin_name].append(line)
#
#     raw_data['KRW'] = list()
#     for line in raw_data['BTC']:
#         raw_data['KRW'].append([line[0], line[1], 1, 1, 1, 1, 1.0, 'normal'])
#
#     return raw_data

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


def make_cryptocurrency_dataset(coin_name_list, start_date, end_date, time_unit, window_size, gap, margin_rate):
    y_trv = []
    y_btv = []
    num_coins = len(coin_name_list)
    # raw_data = get_all_raw_data_from_aws(coin_name_list, start_date, end_date)
    data_files_dir = '../data/old_cleanup_data/'
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


coin_list = ["KRW", "BTC", "ETH", "XRP", "BCH", "LTC", "DASH", "ETC"]
start_date = "2017-01-01 00:00:00"
end_date = "2017-12-31 23:50:00"

time_unit_list = [10, 30, 60]  # candle stick minutes
#
# param_list = [
#     {'learning_rate': 0.5,
#      'loss': 'exponential',
#      'max_depth': 5,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 0.9},
#
#     {'learning_rate': 1.0,
#      'loss': 'exponential',
#      'max_depth': 5,
#      'max_features': 'sqrt',
#      'n_estimators': 50,
#      'subsample': 1.0},
#
#     {'learning_rate': 0.5,
#      'loss': 'deviance',
#      'max_depth': 5,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 1.0},
#
#     {'learning_rate': 1.0,
#      'loss': 'exponential',
#      'max_depth': 4,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 1.0},
#
#     {'learning_rate': 1.0,
#      'loss': 'exponential',
#      'max_depth': 4,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 0.9},
#
#     {'learning_rate': 1.0,
#      'loss': 'deviance',
#      'max_depth': 4,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 0.9},
#
#     {'learning_rate': 0.5,
#      'loss': 'exponential',
#      'max_depth': 5,
#      'max_features': 'sqrt',
#      'n_estimators': 100,
#      'subsample': 0.9}
# ]

# params_XGB = {'learning_rate': [0.005, 0.05, 0.5, 1.0],
#               'n_estimators': [10, 50, 100, 500],
#               'max_depth': [1, 3, 5, 10],
#               'min_child_weight': [1, 3, 5, 10],
#               'gamma': [0, 0.1, 0.5, 1.0],
#               'subsample': [0.8],
#               'colsample_bytree': [0.8],
#               'objective': ['multi:softmax', 'binary:logistic'],
#               'nthread': [1, 3, 5, 10],
#               'scale_pos_weight': [1],
#               'seed': [42]
#               }


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
        X_train[coin], X_test[coin], y_train[coin], y_test[coin] = train_test_split(
            X_reshape,
            y_single[coin],
            train_size=0.8,
            stratify=y_single[coin],
            random_state=42
        )

    for coin in coin_list2:
        print("####################### {} #######################".format(coin))

        best_f1_score = 0.0
        best_conf_matrix = None
        best_metrics = None
        best_param = None
        best_y_pred = None
        best_idx = 0

        # scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = StandardScaler()

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