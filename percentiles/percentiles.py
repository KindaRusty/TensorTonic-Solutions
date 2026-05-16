import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    L = np.percentile(x, q, method='linear')
    x = np.array(L)
    return x