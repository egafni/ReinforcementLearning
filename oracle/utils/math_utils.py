import numpy as np


def pct_change(a, b):
    if a == 0:
        return np.nan
    return (b - a) / a * 100
