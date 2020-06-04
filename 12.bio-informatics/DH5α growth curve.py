#!/usr/bin/env python
# coding: utf-8

# In[23]:


def plot_growth_curve(time, rpm_150, rpm_230, group):
    rpm_150_np = np.array(rpm_150)
    rpm_230_np = np.array(rpm_230)
    df1 = pd.DataFrame(rpm_150_np,
                   index=[0, 1, 2, 3, 4, 6, 8, 21],
                   columns=["150rpm"])
    df2 = pd.DataFrame(rpm_230_np,
                   index=[0, 1, 2, 3, 4, 6, 8, 21],
                   columns=["230rpm"])
    df_result = df1.join(df2)
    
#     df_result.plot()
    
    f1 = plt.figure(figsize=(12, 11))
    plt.ylim(0.0, 3.0)
    plt.plot(time, rpm_150,  color='black', marker='o',         linestyle='-',  linewidth=2, markersize=12,  alpha=.5)
    plt.plot(time, rpm_230,  color='black', marker='*',         linestyle='--',  linewidth=2, markersize=12,  alpha=.5)
    plt.title("DH5α growth curve-group "+group)
    plt.rc('font', size=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.xlabel("Time (hours)")
    plt.ylabel("O.D. (600nm)")
    plt.grid(True, lw = 2, ls = '--', c = '.85')
    plt.legend(['150 rpm', '230 rpm'])
    plt.show()


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# scale 문제 해결
# 숫자 폰트 키우기
# 축 이름과 축 사이 간격 벌리기
if __name__ == "__main__":
    time_numeric = [0, 1, 2, 3, 4, 6, 8, 21]
#     time = ["0", "1", "2", "3", "4", "6", "8", "21.5"]
#     rpm_150 = [0.028, 0.055, 0.162, 0.321, 0.42, 0.521, 0.622, 0.737, 0.852, 1,292, 2.268]
#     rpm_230 = [0.028, 0.067, 0.23, 0.505, 0.76, 1,143, 1.526, 1.817, 2.108, 2.461, 2.84]
    rpm_150 = [0.028, 0.055, 0.162, 0.321, 0.42, 0.622, 0.852, 2.268]
    rpm_230 = [0.028, 0.067, 0.23, 0.505, 0.76, 1.526, 2.108, 2.84]
    plot_growth_curve(time_numeric, rpm_150, rpm_230, "2")
    
#     f1 = plt.figure(figsize=(15, 5))
# #     plt.xlim(0.0, 22.0)
#     plt.ylim(0.0, 3.0)
#     plt.plot(time, rpm_150,  color='green', marker='o',\
#          linestyle='dashed',  linewidth=2, markersize=12,  alpha=.5)
#     plt.plot(time, rpm_230,  color='red', marker='o',\
#          linestyle='dashed',  linewidth=2, markersize=12,  alpha=.5)
#     plt.title("DH5α growth curve-"+"2"+" group")
#     plt.xlabel("Time (Hours)")
#     plt.ylabel("OD600")
#     plt.grid(True, lw = 2, ls = '--', c = '.85')
#     plt.legend(['150 rpm', '230 rpm'])
#     plt.show()
    
    
#     plt.plot(time, rpm_150,  color='green', marker='o',\
#          linestyle='dashed',  linewidth=2, markersize=12,  alpha=.5)
    
#     plt.plot(time, rpm_230,  color='red', marker='o',\
#          linestyle='dashed',  linewidth=2, markersize=12,  alpha=.5)
    
#     mymodel_rpm_150 = np.poly1d(np.polyfit(time_numeric, rpm_150, 3))
#     myline_rpm_150 = np.linspace(0, 22, 100)
#     mymodel_rpm_230 = np.poly1d(np.polyfit(time_numeric, rpm_230, 3))
#     myline_rpm_230 = np.linspace(0, 22, 100)
    
#     plt.scatter(time, rpm_150)
#     plt.plot(time, rpm_150, 'o', color='lightskyblue')
#     plt.plot(myline_rpm_150, mymodel_rpm_150(myline_rpm_150), color='royalblue')
#     plt.plot(time, rpm_230, 'o', color='tomato')
#     plt.plot(myline_rpm_230, mymodel_rpm_230(myline_rpm_230), color='orangered')
    
#     plt.grid(True, lw = 2, ls = '--', c = '.85')
    
#     plt.show()
    
    
    
#     plt.plot(time, rpm_150, 'o', color='pink')
#     plt.plot(time, rpm_150, '-', color='red')
#     plt.show()

#     from sklearn.linear_model import LinearRegression
#     x_train = [0, 1, 2, 3, 4, 6, 8, 21.5]
#     x_test = [5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#     y_train = [0.028, 0.055, 0.162, 0.321, 0.42, 0.622, 0.852, 2.268]
#     mlr = LinearRegression()
#     mlr.fit([x_train], [y_train]) 
#     my_predict = mlr.predict([5])


# In[ ]:




