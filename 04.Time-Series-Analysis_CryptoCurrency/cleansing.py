
# coding: utf-8

# In[1]:


import csv
import os
import time
import datetime
import copy


# In[2]:


user_dir_path = "/Users/dohk/GoogleDrive/Bithumb_Data_Collector/"
saved_file_name = "coin_{0}.csv"
save_file_name = "coin_{0}_cleanup.csv"
save_missing_index_file_name = "coin_{0}_when_missed.csv"
coin_list = ['BTC']
window_size_for_find_start_point = 10
all_data_list = dict() #전체 데이터
for coin in coin_list:
    all_data_list[coin] = list()


# In[3]:


def openAllDataList(coin):
    global all_data_list
    f = open(saved_file_name.format(coin), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        all_data_list[coin].append(line)
    f.close()


# In[4]:


def findStartIndex(coin, data_list):
    for row_index, row in enumerate(data_list[0:-(window_size_for_find_start_point-1)]):
        finder = 0
        for window_index in range(window_size_for_find_start_point):
            if True in [True for i in range(4) if data_list[row_index+window_index][2+i] == '0']:
                break
            else:
                finder = finder + 1
        if finder == window_size_for_find_start_point:
            return row_index
    return False


# In[5]:


if __name__ == "__main__":
    for coin in coin_list:
        delete_file_flag = False
        if os.path.exists(user_dir_path + saved_file_name.format(coin)):
            openAllDataList(coin)
        if len(all_data_list[coin]) == 0:
            print(coin + " could not make a coin's graph data list.")
            continue
            
        start_index = findStartIndex(coin, all_data_list[coin])
        if start_index is False:
            print(coin + " can't find start point")
            break
        start_time = int(all_data_list[coin][start_index][0])
        f = open(save_file_name.format(coin), 'w', encoding='utf-8')
        f_m = open(save_missing_index_file_name.format(coin), 'w', encoding='utf-8')
        wr = csv.writer(f)
        wr_m = csv.writer(f_m)
        data_list_iterator = iter(all_data_list[coin][start_index:])
        ms_time = start_time
        row = next(data_list_iterator)
        temp_row = list()
        while True:
            if row is None:
                break
            time_adjustment_flag = False
            #Time adjustment
            if not(ms_time == int(row[0])) and ms_time - 300000 < int(row[0]) < ms_time + 300000:
                row[0] = str(ms_time)
                row[1] = datetime.datetime.fromtimestamp(int(ms_time/1000)).strftime('%Y-%m-%d %H:%M:%S')
                time_adjustment_flag = True
            if not(ms_time == int(row[0])) and ms_time - 300000 > int(row[0]):
                row = next(data_list_iterator, None)
                continue
            #Zero data
            if ms_time == int(row[0]) and True in [True for i in range(4) if row[2+i] == '0']:
                zero_row = copy.deepcopy(temp_row)
                zero_row[0] = str(ms_time)
                zero_row[1] = datetime.datetime.fromtimestamp(int(ms_time/1000)).strftime('%Y-%m-%d %H:%M:%S')
                if time_adjustment_flag:
                    zero_row.append('adjustment+zero')
                else:
                    zero_row.append('zero')
                wr.writerow(zero_row)
                wr_m.writerow([ms_time, datetime.datetime.fromtimestamp(int(ms_time/1000)).strftime('%Y-%m-%d %H:%M:%S')])
                temp_row = copy.deepcopy(zero_row[:-1])
                row = next(data_list_iterator, None)
            #Normal data
            elif ms_time == int(row[0]) and [True for i in range(4) if row[2+i] == '0'] == []:
                normal_row = copy.deepcopy(row)
                if time_adjustment_flag:
                    normal_row.append('adjustment')
                else:
                    normal_row.append('normal')
                wr.writerow(normal_row)
                temp_row = copy.deepcopy(row)
                row = next(data_list_iterator, None)
            #No data
            elif ms_time < int(row[0]):
                nodata_row = copy.deepcopy(temp_row)
                nodata_row[0] = str(ms_time)
                nodata_row[1] = datetime.datetime.fromtimestamp(int(ms_time/1000)).strftime('%Y-%m-%d %H:%M:%S')
                nodata_row.append('nodata')
                wr.writerow(nodata_row)
                wr_m.writerow([ms_time, datetime.datetime.fromtimestamp(int(ms_time/1000)).strftime('%Y-%m-%d %H:%M:%S')])
                temp_row = copy.deepcopy(nodata_row[:-1])
            elif ms_time > int(row[0]) + 300000:
                row = next(data_list_iterator, None)
            else:
                print(coin + " can't clean up.")
                delete_file_flag = True
                print(row)
                print(temp_row)
                print(ms_time)
                break
            ms_time = ms_time + 600000
        f.close()
        f_m.close()
        if delete_file_flag:
            os.remove(save_file_name.format(coin))
            os.remove(save_missing_index_file_name.format(coin))

