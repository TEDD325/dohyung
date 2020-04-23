import numpy as np

# print(np.identity(16))
# print(np.identity(16)[0:2])
# print(np.eye(16))
# for s1 in range(16):
#     print(np.identity(16)[s1:s1+1])

def one_hot(x):
    return np.identity(16)[x:x+1]

tmp = one_hot(1)
print(tmp)