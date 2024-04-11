import numpy as np

def rescale_ratings(y):
    return -1.0 + 2.0 * np.array(y) / 10.0
