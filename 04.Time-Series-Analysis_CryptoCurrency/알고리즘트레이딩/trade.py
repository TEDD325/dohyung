import sys
"""
python filename.py dbtable start_time end_time exchange time_interval
sys.argv[1] : "TRADE_BTC"
sys.argv[2] : '2017-12-14 01:20:00'
sys.argv[3] : '2017-12-14 01:25:00'
sys.argv[4] : 'bithumb'
sys.argv[5] : 1
"""
dbtable_arg = sys.argv[1]
start_time_arg = sys.argv[2]
end_time_arg = sys.argv[3]
exchange_arg = sys.argv[4]
time_interval_arg = int(sys.argv[5] )
Spark_app_name = dbtable_arg + '|' + start_time_arg + '|' + end_time_arg + '|'  + exchange_arg + '|' + str(time_interval_arg )


import config
import pickle
import os
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import datetime
conf = SparkConf()
conf.set("spark.jars", "/home/tfedohk/spark/jars/mysql-connector-java-5.1.18.jar")
spark = SparkSession \
    .builder \
    .appName(Spark_app_name) \
    .getOrCreate()
df = spark.read.format("jdbc").options(url=config.db_url,
      dbtable=dbtable_arg,
      driver="com.mysql.jdbc.Driver", user=config.id, password=config.password).load()
#df.persist()


# datetime ex) 2016-01-06 00:04:21
# time_interval(minute)
class Trading_Data_Preprocessing_For_Candle_Graph():
    def __init__(self, df, start_time, end_time, time_interval, exchange):
        self.start_time = start_time #'2017-12-14 01:20:00'
        self.end_time = end_time #'2017-12-14 01:20:20'
        self.date_time = start_time
        self.date_time_list = []
        self.date_time_list.append(self.date_time)
        self.time_interval = time_interval
        self.exchange = exchange
        self.df = df

    def is_finished(self):
        current_time = datetime.datetime.strptime(self.date_time, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')
        if current_time >= end_time:
            return True
    #- datetime.timedelta(minutes = 1)
    def change_datetime(self, date_time, time_interval):
        date_time = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        changed_datetime = date_time + datetime.timedelta(minutes = time_interval)
        self.date_time = changed_datetime.strftime("%Y-%m-%d %H:%M:%S")
        self.date_time_list.append(self.date_time)
        return changed_datetime

    def df_divided_by_time_interval(self, date_time, time_interval):
        df_divided = self.df.filter(df.date.between(date_time, self.change_datetime(date_time, time_interval)))
        return df_divided

    def get_prices_from_df_divided(self, is_fisrt = False):
        df = self.df_divided_by_time_interval(self.date_time, self.time_interval)
        df = df.filter(df.exchange == self.exchange)
        df_open = df.filter(df.id == df.first().id).rdd.map(lambda x: x.price)
        df_max = df.filter(df.id == df.orderBy('price', ascending=False).first().id).rdd.map(lambda x: x.price)
        df_min = df.filter(df.id == df.orderBy('price', ascending=True).first().id).rdd.map(lambda x: x.price)
        df_close = df.filter(df.id == df.orderBy('id', ascending=False).first().id).rdd.map(lambda x: x.price)


