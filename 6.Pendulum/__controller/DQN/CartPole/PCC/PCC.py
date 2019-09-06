import pickle
import pandas as pd
import numpy as np

def get_pcc(filename_1, filename_2, col_num):
    with open(filename_1, 'rb') as f:
        data_pen_1 = pickle.load(f)

    with open(filename_2, 'rb') as f:
        data_pen_2 = pickle.load(f)

    len_data_pen_1 = len(data_pen_1)
    len_data_pen_2 = len(data_pen_2)
    min_len = 0
    # print(len_data_pen_1)
    # print(len_data_pen_2)
    if len_data_pen_1 != len_data_pen_2:
        min_len = min(len_data_pen_1, len_data_pen_2)
        data_pen_1 = data_pen_1[:min_len]
        data_pen_2 = data_pen_2[:min_len]
    # print(len(data_pen_1))
    # print(len(data_pen_2))

    np_data_pen_1 = np.array(data_pen_1)
    np_data_pen_2 = np.array(data_pen_2)
    # print(np_data_pen_1[:,:,1,:][:,-1,-1])
    # print(np_data_pen_1[:,:,1,:].shape)
    # print(np_data_pen_1)
    # print(np_data_pen_1)
    # print(len(np_data_pen_1[:,:,1,:][:,-1,-1]))

    lst = []
    lst.append(np_data_pen_1[:, :, col_num, :][:, -1, -1]) # 2: pendulum radian
    lst.append(np_data_pen_2[:, :, col_num, :][:, -1, -1])
    df = pd.DataFrame(lst).T
    print(df)
    corr = df.corr(method = 'pearson')
    print("[pandas] corr:", corr)

col_num = 2 # pendulum_radian
filename_1 = 'pcc_pendulum_1-2.pickle'
filename_2 = 'pcc_pendulum_2-2.pickle'
# 'pcc_pendulum_1-2.pickle'
# 'pcc_pendulum_1-1.pickle'
# 'pcc_pendulum_2-1.pickle'
# 'pcc_pendulum_2-2.pickle'

get_pcc(filename_1, filename_2, col_num)