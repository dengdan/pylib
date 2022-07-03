import numpy as np

def num_choices(total, n):
  """
  从total个选项中随机挑选n个选项, 有多少种结果
  """
  return np.prod([total - i * 1. for i in range(0, n)])