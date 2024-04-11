import numpy as np


def unit_constrain(z: np.ndarray, padding=1e-4):
    return padding + (1 - 2 * padding) * (z - z.min()) / (z.max() - z.min())
