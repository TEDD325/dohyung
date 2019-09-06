'''
나이브 베이즈 분류기는 LinearSVC보다 훈련 속도가 빠른 편이지만, 일반화 성능이 뒤쳐진다.<br>
GaussianNB는 연속적인 어떤 데이터에도 적용할 수 있다.<br>
GaussianNBsms 클래스별로 각 특성의 표준편차와 평균을 저장한다. 예측시엔 데이터 포인트를 클래스의 통계 값과 비교해서 가장 잘 맞는 클래스를 예측값으로 한다.<br>
GaussianNB는 대부분 매우 고차원인 데이터셋에 사용한다.<br>
선형 모델로는 학습 시간이 너무 오래 걸리는 매우 큰 데이터셋에는 나이브 베이즈 모델을 시도해볼 만하며 종종 사용된다.
'''

import os
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import mglearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

def search(dirname):
    filenames = os.listdir(dirname)
    fileList = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        fileList.append(full_filename)
    return fileList

fileList = search("./data/RNN_coin/")
# print(fileList)
fileList.remove("./data/RNN_coin/.DS_Store")

coin_data_dict = {}
coin_data_x_np = {}
coin_data_y_np = {}
coin_data_x_np_reshaped = {}
coin_data_y_np_reshaped = {}
coin_data_x_pd = {}
coin_data_y_pd = {}
X_train = {}
X_test = {}
y_train = {}
y_test = {}

for file in fileList:
    print(file)
    data = pd.read_pickle(file)
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

    # X_train = MinMaxScaler(X_train)
    # X_test = MinMaxScaler(X_test)
    # y_train = MinMaxScaler(y_train)
    # y_test = MinMaxScaler(y_test)
    #
    # GaussianNB의 경우 scaling 적용 시 오히려 정확도가 떨어지는 문제

    Gaussian = GaussianNB()

    # C=1.0, max_iter=10
    train_score_sum = []
    test_score_sum = []

    clf_2 = Gaussian.fit(X_train, y_train)
    param_info = clf_2.get_params()
    train_score = clf_2.score(X_train, y_train)
    test_score = clf_2.score(X_test, y_test)
    train_score_sum.append(train_score)
    test_score_sum.append(test_score)
    print("train_score : {}".format(train_score))
    print("test_score : {}".format(test_score))
    print()
    print()
