import pandas as pd

temp = pd.read_pickle("../data/RNN_coin/X_10_10_1_0.1.pickle")
print(temp[0].shape)