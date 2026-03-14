import numpy as np

def mean_squared_error(pred, true):
    """
    Returns: float MSE
    """
    pred = np.array(pred)
    true = np.array(true)
    #tính tổng trung bình các giá trị
    mse = np.mean((pred-true)**2)
    return float(mse)
