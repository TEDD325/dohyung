import csv
from os import listdir
from os.path import isfile, join
from datetime import datetime
import _pickle as pickle
import numpy as np


def makeDataset(data_files_dir, tradable_coin_list, start_date, test_start_date, end_date, data_shape):
    y = []
    data_files_dir = data_files_dir
    all_data_files = [f for f in listdir(data_files_dir) if isfile(join(data_files_dir, f))]
    trading_coins = tradable_coin_list
    trading_files = [f for f in all_data_files for coin in trading_coins if coin in f]
    data_shape = data_shape
    print("Trading File List : ", trading_files)

    all_data_list = dict() #전체 데이터
    for f in trading_files:
        all_data_list[f] = list()

    #27000, 1501850400000 2017.08.04 21:40 ~ 1518049800000 2018.02.08 9:30
    #1800, for BT:1518050400000 2018.02.08 9:40 ~ 1519159200000 2018.02.21 5:40

    start_ms_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp()*1000 #1501850400000
    test_start_ms_time = datetime.strptime(test_start_date, "%Y-%m-%d %H:%M:%S").timestamp()*1000
    end_ms_time = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp()*1000 #1519159200000

    for f in trading_files:
        file = open(data_files_dir + f, 'r', encoding='utf-8')
        rdr = csv.reader(file)
        for line in rdr:
            if start_ms_time <= int(line[0]) and end_ms_time >= int(line[0]):
                all_data_list[f].append(line)
            else:
                pass
        file.close()

    print("Data Length ({0} ~ {1})".format(start_ms_time, end_ms_time))
    for i in trading_files:
        print(i, ":", len(all_data_list[i]))

    trv = list() #train value : [[v_t, v_t_hi, v_t_low], ...], shape : (27000, 3, 7, 50)
    btv = list() #backtest value : [v_t, v_t_hi, v_t_low], ...], shape : (1800, 3, 7, 50)

    trv_file = open(data_files_dir + 'dataset/trv', 'wb')
    btv_file = open(data_files_dir + 'dataset/btv', 'wb')

    data_iterator = dict()
    for file in trading_files:
        data_iterator[file] = iter(all_data_list[file])


    v_t = list() #shape : (6, 50)
    v_t_hi = list() #(6, 50)
    v_t_low = list() #(6, 50)

    v_t_f_dict = dict()
    iter_index = 0
    v_t_tmp = []; v_t_hi_tmp = []; v_t_low_tmp = []  
    y_t = []; y_t_hi = []; y_t_low = []

    while True:
        try:
            for f in trading_files:
                if iter_index > 0:
                    v = next(data_iterator[f])
                    if v == None:
                        break
                    v_t_f_dict[f]['v'] = v_t_f_dict[f]['v'][1:]
                    v_t_f_dict[f]['hi'] = v_t_f_dict[f]['hi'][1:]
                    v_t_f_dict[f]['low'] = v_t_f_dict[f]['low'][1:]
                    v_t_f_dict[f]['v'].append(v[3])
                    v_t_f_dict[f]['hi'].append(v[4])
                    v_t_f_dict[f]['low'].append(v[5])
                else:
                    v_t_f_dict[f] = dict()
                    v_t_f_dict[f]['v'] = list()
                    v_t_f_dict[f]['hi'] = list()
                    v_t_f_dict[f]['low'] = list()
                    for i in range(data_shape[2]):
                        v = next(data_iterator[f])
                        v_t_f_dict[f]['v'].append(v[3])
                        v_t_f_dict[f]['hi'].append(v[4])
                        v_t_f_dict[f]['low'].append(v[5])
                v_t.append(v_t_f_dict[f]['v'])
                v_t_hi.append(v_t_f_dict[f]['hi'])
                v_t_low.append(v_t_f_dict[f]['low'])

            priceList = []
            if len(v_t_tmp) != 0:
		for i in v_t_tmp:
                    priceList.append(i[-1])

		priceListIter = iter(priceList)
		for i in v_t:
                    if i[0] >= next(priceListIter):
			y_t.append(1)
                    elif i[0] < next(priceListIter):
			y_t.append(0)

		for i in v_t_hi_tmp:
                    priceList.append(i[-1])

		priceListIter = iter(priceList)
		for i in v_t_hi:
                    if i[0] >= next(priceListIter):
			y_t_hi.append(1)
                    elif i[0] < next(priceListIter):
			y_t_hi.append(0)

		for i in v_t_low_tmp:
                    priceList.append(i[-1])

		priceListIter = iter(priceList)
		for i in v_t_low:
                    if i[0] >= next(priceListIter):
			y_t_low.append(1)
                    elif i[0] < next(priceListIter):
			y_t_low.append(0)
				
            v_t_tmp = v_t
            v_t_hi_tmp = v_t_hi
            v_t_low_tmp = v_t_low

            y_t = []; y_t_hi = []; y_t_low = []

            y.append([y_t, y_t_hi, y_t_low])

            if int(v[0]) < int(test_start_ms_time):
                trv.append([v_t, v_t_hi, v_t_low])
                v_t = list()
                v_t_hi = list()
                v_t_low = list()
            else:
                btv.append([v_t, v_t_hi, v_t_low])
                v_t = list()
                v_t_hi = list()
                v_t_low = list()
            iter_index += 1
        except StopIteration:
            break

    trv = np.array(trv, dtype=np.float64)
    btv = np.array(btv, dtype=np.float64)

    y = np.array(y, dtype=np.int64)

    print("train dataset's shape :", trv.shape)
    print("backtest dataset's shape :", btv.shape)

    print("example : ", trv[0])

    pickle.dump(trv, trv_file, protocol=4)
    pickle.dump(btv, btv_file, protocol=4)

    trv_file.close()
    btv_file.close()

    x = {}

    x["trv"] = trv
    x["btv"] = btv

    
    
   return x, y


if __main__ == "__main__":
    data_files_dir = os.getcwd() + "/coinData/dataset/"
    tradable_coin_list = ["BCH", "BTC", "DASH", "ETC", "ETH", "LTC", "XRP"]
    start_date = "2017-08-04 09:40:00"
    test_start_date = "2018-01-07 08:40:00"
    end_date = "2018-01-31 08:40:00"
    data_shape = [3, len(tradable_coin_list), 100]
    
    x, y = makeDataset(data_files_dir, tradable_coin_list, start_date, test_start_date, end_date, data_shape)


    print(x.shape)
    print(y.shape)