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
    
    y_mean = np.mean(y_true)
    SStot = np.sum((y_true - y_mean)**2)
    SSres = np.sum((y_true - y_pred)**2)
    if SStot == 0:
        return 1.0 if SSres == 0 else 0.0
    return 1 - (SSres / SStot)