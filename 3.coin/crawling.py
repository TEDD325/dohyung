
# coding: utf-8

# In[1]:


import urllib.request as urq
import sys
from bs4 import BeautifulSoup
import csv
import copy
import os
import threading
import time
import datetime


# In[24]:


user_dir_path = "/Users/dohk/GoogleDrive/coinCrawling/Bithumb_Data_Collector"
save_file_name = "coin_{0}.csv"
bithumb_graph_url = 'http://index.bithumb.com/api/bithumb/localAPI.php?api=graph&coin={0}&back={1}'
coin_list = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'EOS', 'DASH', 'XMR', 'ETC', 'QTUM', 'BTG', 'ZEC']

all_coin_msec_ago = dict() #얼마나 이전의 데이터를 가져올지
all_data_list = dict() #전체 데이터
for coin in coin_list:
    all_data_list[coin] = list()
    all_coin_msec_ago[coin] = 0
current_data_list = list() #현재 수집 데이터
long_msec_for_back_to = 3600000000 #1000시간
much_long_msec_for_back_to = 90000000 #25시간

loop_sleep_time = 3


# In[12]:


def dataParser(coin, msec):
    if coin in coin_list:
        pass
    else:
        print('The coin({0}) you wrote is not on coin-list!!'.format(coin))
        return False
    try:
        html = urq.urlopen(bithumb_graph_url.format(coin, msec))
    except:
        print("URL_OPEN_ERROR")
        return False
    data = BeautifulSoup(html.read(), "html.parser")
    data_list = list()
    text_data = data.text
    text_data = text_data[text_data.find('[')+1:text_data.rfind(']')]
    l_index = 0
    r_index = 0
    append_l_flag = False
    append_r_flag = False
    for index, letter in enumerate(text_data):
        if letter == '[':
            l_index = index
            append_l_flag = True
        elif letter == ']':
            r_index = index
            append_r_flag = True
        if append_l_flag and append_r_flag:
            row = text_data[l_index+1:r_index].split(',')
            row.insert(1, datetime.datetime.fromtimestamp(int(int(row[0])/1000)).strftime('%Y-%m-%d %H:%M:%S'))
            data_list.append(row)
            append_l_flag = False
            append_r_flag = False
    return data_list


# In[13]:


# Loop 1
def longBackTo(coin):
    print(coin + " - Loop 1 start at " + str(datetime.datetime.now()))
    time.sleep(loop_sleep_time)
    global current_data_list
    global all_data_list
    global all_coin_msec_ago
    current_data_list = dataParser(coin, str(all_coin_msec_ago[coin]))
    if not current_data_list:
        print("Can't search long ago data - {0}.".format(coin))
        print(coin + " - Loop 1 end")
        return False
    if len(all_data_list[coin]) == 0:
        all_data_list[coin] = current_data_list
        #더 이전의 데이터를 가져오기 위함
        all_coin_msec_ago[coin] = all_coin_msec_ago[coin] + long_msec_for_back_to
        print(coin + " - Loop 1 end")
        return current_data_list
    else:
        #row[0] = 현재 수집된 데이터 한 줄의 시간
        for row in current_data_list:
            if int(all_data_list[coin][0][0]) > int(row[0]):
                all_data_list[coin].insert(0, row)
            elif int(all_data_list[coin][-1][0]) < int(row[0]):
                all_data_list[coin].append(row)
            else:
                for i, a_row in enumerate(all_data_list[coin][0:-1]):
                    if int(a_row[0]) == int(row[0]):
                        break
                    elif int(a_row[0]) < int(row[0]) and int(all_data_list[coin][i+1][0]) > int(row[0]):
                        all_data_list[coin].insert(i+1, row)
                        break
        #더 이전의 데이터를 가져오기 위함
        all_coin_msec_ago[coin] = all_coin_msec_ago[coin] + long_msec_for_back_to
    print(coin + " - Loop 1 end")


# In[14]:


# Loop 2
def muchLongBackTo(coin):
    print(coin + " - Loop 2 start at " + str(datetime.datetime.now()))
    time.sleep(loop_sleep_time)
    global current_data_list
    global all_data_list
    global all_coin_msec_ago
    current_data_list = dataParser(coin, str(all_coin_msec_ago[coin] + much_long_msec_for_back_to))
    if not current_data_list:
        print("Can't search much long ago data - {0}.".format(coin))
        print(coin + " - Loop 2 end")
        return False
    if len(all_data_list[coin]) == 0:
        all_data_list[coin] = current_data_list
        #더욱 더 이전의 데이터를 가져오기 위함
        all_coin_msec_ago[coin] = all_coin_msec_ago[coin] + much_long_msec_for_back_to
        print(coin + " - Loop 2 end")
        return current_data_list
    else:
        #row[0] = 현재 수집된 데이터 한 줄의 시간
        for row in current_data_list:
            if int(all_data_list[coin][0][0]) > int(row[0]):
                all_data_list[coin].insert(0, row)
            elif int(all_data_list[coin][-1][0]) < int(row[0]):
                all_data_list[coin].append(row)
            else:
                for i, a_row in enumerate(all_data_list[coin][0:-1]):
                    if int(a_row[0]) == int(row[0]):
                        break
                    elif int(a_row[0]) < int(row[0]) and int(all_data_list[coin][i+1][0]) > int(row[0]):
                        all_data_list[coin].insert(i+1, row)
                        break
        #더욱 더 이전의 데이터를 가져오기 위함
        all_coin_msec_ago[coin] = all_coin_msec_ago[coin] + much_long_msec_for_back_to
    print(coin + " - Loop 2 end")


# In[15]:


# Loop 3
def currentOnly(coin):
    print(coin + " - Loop 3 start at " + str(datetime.datetime.now()))
    time.sleep(loop_sleep_time)
    global current_data_list
    global all_data_list
    current_data_list = dataParser(coin, str(0))
    if not current_data_list:
        print("Can't search long ago data - {0}.".format(coin))
        print(coin + " - Loop 3 end")
        return False
    if len(all_data_list[coin]) == 0:
        all_data_list[coin] = current_data_list
        print(coin + " - Loop 3 end")
        return current_data_list
    else:
        #row[0] = 현재 수집된 데이터 한 줄의 시간
        for row in current_data_list:
            if int(all_data_list[coin][0][0]) > int(row[0]):
                all_data_list[coin].insert(0, row)
            elif int(all_data_list[coin][-1][0]) < int(row[0]):
                all_data_list[coin].append(row)
            else:
                for i, a_row in enumerate(all_data_list[coin][0:-1]):
                    if int(a_row[0]) == int(row[0]):
                        break
                    elif int(a_row[0]) < int(row[0]) and int(all_data_list[coin][i+1][0]) > int(row[0]):
                        all_data_list[coin].insert(i+1, row)
                        break
    print(coin + " - Loop 3 end")


# In[16]:


# Loop 4
def saveAllDataList(coin):
    print(coin + " - Saving data start at " + str(datetime.datetime.now()))
    time.sleep(loop_sleep_time)
    f = open(save_file_name.format(coin), 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for line in all_data_list[coin]:
        wr.writerow(line)
    f.close()
    print(coin + " - Saved data")


# In[17]:


def openAllDataList(coin):
    global all_data_list
    f = open(save_file_name.format(coin), 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        all_data_list[coin].append(line)
    f.close()


# In[25]:


if __name__ == "__main__":
    timer = 0
    for coin in coin_list:
        if os.path.exists(user_dir_path + save_file_name.format(coin)):
            openAllDataList(coin)
    while True:
        for coin in coin_list:
            longBackTo(coin)
            if timer % 2 == 1:
                muchLongBackTo(coin)
                currentOnly(coin)
            saveAllDataList(coin)
        
        timer = timer + 1

