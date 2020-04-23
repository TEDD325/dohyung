import numpy as np

A = np.array([1, 5, 3, 7, 2, 6])

print(A)

print(np.where(A < 3))
print(np.where(A > 5))
print(np.where(A > 10))