import numpy as np
from sklearn.metrics import confusion_matrix as cmatrix
from sklearn.metrics import f1_score as score
from sklearn.metrics import precision_recall_curve as curve

pred = [1, 0, 0, 0, 1, 1]
Y_test = [1, 0, 1, 0, 1, 1]

matrix = cmatrix(Y_test, pred)
print(matrix)

tn, fp, fn, tp = matrix.ravel()
print('true negative: ', tn) # True인데, True가 아니라고 prediction한 것
print('false positive: ', fp) # False인데, True라고 예측한 것
print('false negative: ', fn) # False인데, False라고 예측한 것
print('true positive: ', tp, '\n') # True인데, True라고 예측한 것

_, recall, threshold = curve(Y_test, pred)

print('recall: ', recall[1])
print('threshold: ', threshold[1])

print('F1 score: {:.2f}'.format(score(Y_test, pred)))