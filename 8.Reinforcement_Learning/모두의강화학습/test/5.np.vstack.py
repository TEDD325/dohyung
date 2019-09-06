import numpy as np
"""
vertical : 수직, 세로
"""
a = np.arange(5)
b = np.arange(5, 10)
c = np.arange(10, 15)
print("[INFO] a: ", a)
print("[INFO] b: ", b)
print("[INFO] c: ", c, end="\n\n")

x = np.vstack([a,b])
print("[INFO] x: \n", x)

y = np.vstack([x,c])
print("[INFO] y: \n", y)