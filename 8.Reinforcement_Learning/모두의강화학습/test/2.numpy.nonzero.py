import numpy as np
"""
# 1
요소들 중 0이 아닌 값들의 index 들을 반환
"""

a = np.array([1, 0, 2, 3, 0])
print(a)
b = np.nonzero(a)
print(b) # 0이 아닌 값 1, 2, 3 의 index 인 0, 2, 3 을 array 형태로 리턴
print()

"""
# 2
2D array인 경우
"""
a = np.array([[1, 0, 7], [8, 1, 0], [1, 0, 0]])
print(a)
b = np.nonzero(a)
print(b, end="\n\n")
"""
결과 해석:
행 인덱스 정보> array([0, 0, 1, 1, 2])
열 인덱스 정보> array([0, 2, 0, 1, 0])

2D array의 (0행, 0열) == 1
2D array의 (0행, 2열) == 7
2D array의 (1행, 0열) == 8
2D array의 (1행, 1열) == 1
2D array의 (2행, 0열) == 1

이 결과를 쉽게 보려면 다음과 같이 결과로 주어진 놈을 transpose 시켜주면 된다.
"""
c = np.transpose(np.nonzero(a))
print(c, end="\n\n")

"""
# 3
0이 아닌 요소들의 인덱스 말고, 값을 볼 수 있는 방법
"""
d = a[np.nonzero(a)]
print(d, end="\n\n")

"""
# 4
np.nonzero(조건) 의 형태로 조건을 주면 조건을 참으로 만드는 요소들의 index 가 리턴된다.
"""
a = np.array([[0, 0, 1, 0],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 0]])
a = np.array([0, 0, 1, 0])
condition = (a == 1)
print("shape: ", a.shape)
print(condition)
b = np.nonzero(condition)
print(b)
print("b[0]: ", b[0])
c = np.transpose(b)
print("c: ", c, end="\n\n")
"""
또는 다음과 같은 방식도 가능하다.
"""
d = (condition).nonzero()
print(d)
e = np.transpose((condition).nonzero())
print(e)

print()

