import numpy as np
""" 
array max의 약자.
행렬 (matrix)이나 벡터 (Vector) 에서 element-wise하게 값 하나하나를 검사하여 최댓값을 반환
"""
# a = np.arange(4)
# print(a)
# b = np.amax(a)
# print(b)
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],
              [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [1, 2, 3, 4], [5, 6, 7, 8],
              [9, 10, 11, 12], [13, 14, 15, 100]])
print(a.shape)
print(a)
b = np.amax(a)
print(b)
""" return되는 값은 가장 큰 값 Int 하나다."""

print()


"""
보통, 서양은 축을 따질 때 열을 먼저 세는 경향이 있다. 따라서,
파라미터 axis=0는 열(col), axis=1은 행(row)이다.
numpy.amax함수도 파라미터로 axis를 받을 수 있으며, 이 때엔 값 하나가 아닌, array를 리턴할 수 있다.
"""
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],
              [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [1, 2, 3, 4], [5, 6, 7, 8],
              [9, 10, 11, 12], [13, 14, 15, 100]])
print(a.shape)
print(a)
b = np.amax(a, axis=0)
print(b)