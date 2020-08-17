# https://www.bithumb.com/u1/US127
# https://poloniex.com/support/api/
import json
import urllib.request
from urllib.request import Request, urlopen
from apscheduler.schedulers.background import BackgroundScheduler
import pymysql
import time
import numpy as np
import random
import sys, os
import datetime
from dbConfig import *

# dbURL = "localhost"
# dbUser = "spring"
# dbPass = "book"
# dbPort = 3306
# dbName = "trade"

unit_time = 10

insertCurrencySql = "INSERT INTO `trade`.`currency` (`timestamp`, `timestamp_time`, `added_time`, " + \
                    "`btc_volume`, `btc_first`, `btc_last`, `btc_high`, `btc_low`, " + \
                    "`eth_volume`, `eth_first`, `eth_last`, `eth_high`, `eth_low`, `etc_volume`, `etc_first`," + \
                    " `etc_last`, `etc_high`, `etc_low`, " + \
                    "`xrp_volume`, `xrp_first`, `xrp_last`, `xrp_high`, `xrp_low`) " +\
                    "VALUES (%s, %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
conn = pymysql.connect(host=dbURL, port=dbPort, user=dbUser, passwd=dbPass, db=dbName, charset='utf8mb4', use_unicode=True)

def getTicker():
    urlTicker = urllib.request.urlopen('https://api.coinone.co.kr/ticker/?currency=all')
    readTicker = urlTicker.read()
    t = json.loads(readTicker)
    cursor = conn.cursor()

    for name in ['btc', 'eth', 'etc', 'xrp']:
        t[name]['volume'] = float(t[name]['volume'])  # volume of completed orders in 24 hours.
        t[name]['first'] = float(t[name]['first'])  # First price in 24 hours.
        t[name]['last'] = float(t[name]['last'])  # Last completed price.
        t[name]['high'] = float(t[name]['high'])  # Highest price in 24 hours.
        t[name]['low'] = float(t[name]['low'])  # Lowest price in 24 hours.

    timestamp_time = datetime.datetime.fromtimestamp(int(t['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute(insertCurrencySql, (t['timestamp'],  timestamp_time,
                                       t['btc']['volume'], t['btc']['first'], t['btc']['last'], t['btc']['high'],
                                       t['btc']['low'],
                                       t['eth']['volume'], t['eth']['first'], t['eth']['last'], t['eth']['high'],
                                       t['eth']['low'],
                                       t['etc']['volume'], t['etc']['first'], t['etc']['last'], t['etc']['high'],
                                       t['etc']['low'],
                                       t['xrp']['volume'], t['xrp']['first'], t['xrp']['last'], t['xrp']['high'],
                                       t['xrp']['low']))
    conn.commit()
    print(
        "{0} (last) - btc: {1}, eth: {2}, etc: {3}, xrp: {4}".format(t['timestamp'], t['btc']['last'], t['eth']['last'],
                                                                     t['etc']['last'], t['xrp']['last']))

if __name__ == "__main__":
    while(True):
        getTicker()
        time.sleep(unit_time)
