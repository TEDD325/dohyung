import numpy as np
X = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(X)
print(X.shape, '\n')

Y = X.reshape(-1)

print(Y)
print(Y.shape, '\n')

Z = X.reshape(-1, 2)
print(Z)
print(Z.shape)