import numpy as np
from scipy import stats

def np_str(string_list):
    return np.array(eval(string_list))

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def cor(x, y):
    return stats.pearsonr(x, y)[0]

def sign_acc(x, y):
    return np.mean(np.sign(x) == np.sign(y))