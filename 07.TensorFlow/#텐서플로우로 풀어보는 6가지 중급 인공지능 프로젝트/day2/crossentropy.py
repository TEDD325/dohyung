import numpy as np

j = [0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025, 0.0025, 0.0025, 0.0025] # 추측한 값
k = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] # 정답

log = -(np.log(j))
print(log)

prod = k * log
print(prod)

i = np.sum(prod)
print(i)