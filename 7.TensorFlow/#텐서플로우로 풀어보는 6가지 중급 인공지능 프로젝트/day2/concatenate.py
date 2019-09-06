import numpy as np

a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])

print(a, '\n')
print(b, '\n\n')

print(np.concatenate((a,b), axis=0), '\n') # 행축 기준 결합 
print(np.concatenate((a,b), axis=1), '\n') # 열축 기준 결합
print(np.concatenate((a,b), axis=None))
