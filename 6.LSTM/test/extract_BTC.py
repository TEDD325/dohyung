import numpy as np
import pandas as pd
f = pd.read_pickle('../data/RNN_coin/X_10_10_1_0.1.pickle')
# print(f)
# print(type(f))
# print(len(f))
# print(f[0])
# print(type(f[0]))
# print(f[0].shape)
data = np.reshape(f[0], [52550, 8, 4, 10])
# print(data.shape)
BTC = data[:,1,0,0]
# print(BTC.shape)
# print(BTC)
# print(type(BTC))
df= pd.DataFrame(BTC)
# print(df)