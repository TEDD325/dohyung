import csv
from os import listdir
from os.path import isfile, join
import _pickle as pickle
import numpy as np

data_files_dir = "/Users/jubong/git/RL/cleanup"
all_data_files = [f for f in listdir(data_files_dir) if isfile(join(data_files_dir, f))]
trading_coins = ['BCH', 'BTC', 'DASH', 'ETC', 'ETH', 'LTC', 'XRP']
trading_files = [f for f in all_data_files for coin in trading_coins if coin in f]
print("Trading File List : ", trading_files)

all_data_list = dict() #전체 데이터
for f in trading_files:
    all_data_list[f] = list()

#27000, 1501850400000 2017.08.04 21:40 ~ 1518049800000 2018.02.08 9:30
#1800, for BT:1518050400000 2018.02.08 9:40 ~ 1519159200000 2018.02.21 5:40

start_ms_time = 1501850400000
end_ms_time = 1519159200000

for f in trading_files:
    file = open('cleanup/' + f, 'r', encoding='utf-8')
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

trv_file = open('cleanup/dataset/trv', 'wb')
btv_file = open('cleanup/dataset/btv', 'wb')

data_iterator = dict()
for file in trading_files:
    data_iterator[file] = iter(all_data_list[file])


v_t = list() #shape : (7, 50)
v_t_hi = list() #(7, 50)
v_t_low = list() #(7, 50)
v_t_f = list() #shape : (50)
v_t_hi_f = list() #(50)
v_t_low_f = list() #(50)
while True:
    try:
        for f in trading_files:
            if not v_t_f == list():
                v = next(data_iterator[f])
                if v == None:
                    break
                v_t_f = v_t_f[1:]
                v_t_hi_f = v_t_hi_f[1:]
                v_t_low_f = v_t_low_f[1:]
                v_t_f.append(v[3])
                v_t_hi_f.append(v[4])
                v_t_low_f.append(v[5])
            else:
                for i in range(50):
                    v = next(data_iterator[f])
                    v_t_f.append(v[3])
                    v_t_hi_f.append(v[4])
                    v_t_low_f.append(v[5])
            v_t.append(v_t_f)
            v_t_hi.append(v_t_hi_f)
            v_t_low.append(v_t_low_f)

        if len(trv) < 27000:
            trv.append([v_t, v_t_hi, v_t_low])
            v_t = list()
            v_t_hi = list()
            v_t_low = list()
        else:
            btv.append([v_t, v_t_hi, v_t_low])
            v_t = list()
            v_t_hi = list()
            v_t_low = list()
    except StopIteration:
        break

trv = np.array(trv, dtype=np.float64)
btv = np.array(btv, dtype=np.float64)

print("train dataset's shape :", trv.shape)
print("backtest dataset's shape :", btv.shape)

print("example : ", trv[0])

pickle.dump(trv, trv_file, protocol=4)
pickle.dump(btv, btv_file, protocol=4)

trv_file.close()
btv_file.close()