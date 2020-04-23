import pandas as pd
import numpy as np

my_array = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
my_array = my_array.reshape(3, 4)
df = pd.DataFrame(my_array, columns=['A', 'B', 'C', 'D'])
print(df, '\n')

df = df.drop(['B', 'C'], axis=1)
print(df, '\n')

df = df.drop([1], axis=0)
print(df)