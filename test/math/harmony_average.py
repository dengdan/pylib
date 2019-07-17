import numpy as np

arr = [-2] * 5000 + [2.6] * 5000 + [100] * 800
print(np.mean(arr))

base = 10e-9
brr = [1 / (a + base) for a  in arr]
print(1 / np.mean(brr) - base)