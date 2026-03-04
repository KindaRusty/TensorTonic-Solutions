import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x)
    p = np.array(p)
    
    if x.shape != p.shape:
        raise ValueError("Size of x and p must be the same")
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Total probabilities must be 1")
    expected_value = np.sum(x * p)
    return float(expected_value)