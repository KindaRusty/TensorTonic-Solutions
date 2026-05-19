import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x1 = np.array(x)
    p1 = np.array(p)

    if not np.isclose(np.sum(p1), 1.0):
        raise ValueError("The input is not valid")
    Expected_value = np.sum(x1*p1)
    return float(Expected_value)