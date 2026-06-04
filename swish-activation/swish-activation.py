import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x, dtype = float)
    ox = 1/(1+ np.exp(-x))
    swish_val = x * ox 
    return swish_val.tolist() if isinstance(x, (list, np.ndarray)) else float(swish_val)
    