
# coding: utf-8

# In[1]:


import csv
from os import listdir
from os.path import isfile, join
import _pickle as pickle
import numpy as np


# In[2]:


data_files_dir = "/Users/dohk/GoogleDrive/Bithumb_Data_Collector/cleanup"
# cleanup된 raw data의 위치


# In[3]:


all_data_files = [f for f in listdir(data_files_dir) if isfile(join(data_files_dir, f))]
# 디렉토리에 존재하는 모든 파일의 이름을 리스트로 저장


# In[4]:


all_data_files


# In[5]:


trading_coins = ['BTC']
trading_files = [f for f in all_data_files for coin in trading_coins if coin in f]
trading_files
# 데이터셋으로 만들고자 하는 코인의 이름을 리스트로 저장


# unix timestamp: 1518050400000 ~ <br>
# human readable time: 2018.02.08 9:30 ~ <br>
# unix timestamp: 1519159200000 <br>
# human readable time: 2018.02.21 5:40<br>

# In[6]:


all_data_list = dict() #전체 데이터
for f in trading_files:
    all_data_list[f] = list()
    
start_ms_time = 1494633000000
end_ms_time = 1520293200000 #1519129800000 #1519159200000

for f in trading_files:  # 각 코인별로
    file = open(data_files_dir + '/' + f, 'r', encoding='utf-8') 
    rdr = csv.reader(file)  # read mode로 열고
    for line in rdr:  # 읽은 csv파일을 라인별로
        if start_ms_time <= int(line[0]) and end_ms_time >= int(line[0]):  # 뭘 검사하냐면, 시간을.
            all_data_list[f].append(line)  # 해당 코인의 key값에 대한 value로 해당 라인을 저장.
        else:
            pass  # starttime~endtime범위를 넘어간 데이터는 패스
    file.close()
    
for i in trading_files:
    print(i, ":", len(all_data_list[i]))


# In[8]:


all_data_list['coin_BTC_cleanup.csv'][0]


# In[9]:


all_data_list['coin_BTC_cleanup.csv'][-1]


# In[10]:


all_data_list = dict() #전체 데이터
for f in trading_files:
    all_data_list[f] = list()
    
start_ms_time = 1494633000000
end_ms_time = 1520293200000 #1519129800000 #1519159200000

for f in trading_files:  # 각 코인별로
    file = open(data_files_dir + '/' + f, 'r', encoding='utf-8') 
    rdr = csv.reader(file)  # read mode로 열고
    for line in rdr:  # 읽은 csv파일을 라인별로
        if start_ms_time <= int(line[0]) and end_ms_time >= int(line[0]):  # 뭘 검사하냐면, 시간을.
            all_data_list[f].append(line)  # 해당 코인의 key값에 대한 value로 해당 라인을 저장.
        else:
            pass  # starttime~endtime범위를 넘어간 데이터는 패스
    file.close()
    
for i in trading_files:
    print(i, ":", len(all_data_list[i]))


# In[18]:


len(all_data_list)


# In[20]:


len(all_data_list['coin_BTC_cleanup.csv'])


# In[22]:


int(len(all_data_list['coin_BTC_cleanup.csv'])*0.8) # 80%


# In[25]:


trv = list() #train value : [[v_t, v_t_hi, v_t_low], ...], shape : (27000, 3, 7, 50)
btv = list() #backtest value : [v_t, v_t_hi, v_t_low], ...], shape : (1800, 3, 7, 50)

trv_file = open(data_files_dir + '/dataset/trv', 'wb')
btv_file = open(data_files_dir + '/dataset/btv', 'wb')


# In[26]:


data_iterator = dict()
for file in trading_files:
    data_iterator[file] = iter(all_data_list[file])


# In[27]:


v_t = list() #shape : (7, 50)
v_t_hi = list() #(7, 50)
v_t_low = list() #(7, 50)
v_t_f = list() #shape : (50)
v_t_hi_f = list() #(50)
v_t_low_f = list() #(50)


# In[28]:


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

        if len(trv) < int(len(all_data_list['coin_BTC_cleanup.csv'])*0.8):
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


# In[30]:


print(len(trv))
print(len(btv))


# In[33]:


type(trv)

