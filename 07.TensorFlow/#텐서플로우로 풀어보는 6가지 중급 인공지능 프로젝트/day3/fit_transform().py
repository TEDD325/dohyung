from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler_ss = StandardScaler()
scaler_mms = MinMaxScaler()

my_array = np.array([[1], [2], [3], [4]])
print(my_array, '\n')
print(my_array.shape, '\n\n')

# standard scaler
print("standard scaler")

x = scaler_ss.fit_transform(my_array)
print(x, '\n')
print("mean: ", np.mean(x))
print("std: ", np.std(x), '\n')

y = scaler_ss.inverse_transform(x)
print(y, '\n\n')

# min-max scaler
print("min-max scaler")

x = scaler_mms.fit_transform(my_array)
print(x, '\n')
print("mean: ", np.mean(x))
print("std: ", np.std(x), '\n')

y = scaler_mms.inverse_transform(x)
print(y)