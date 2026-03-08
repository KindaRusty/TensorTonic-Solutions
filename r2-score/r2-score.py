import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 1. Get mean value
    y_mean = np.mean(y_true)
    # 2. Handle constant
    if np.all(y_true == y_mean):
        if np.array_equal(y_true, y_pred):
            return 1.0
        else:
            return 0.0
    # 3. Calculate
    ss_res = np.sum(np.square(y_true - y_pred)) #Có thể dùng **2 là bình phương thay vì dùng np.square
    ss_tot = np.sum(np.square(y_true - y_mean))
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)