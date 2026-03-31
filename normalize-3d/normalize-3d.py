import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length using NumPy.
    Supports shape (3,) and (N, 3).
    """
    v = np.array(v, dtype=float)
    #Tính tích vô hướng
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    #Check >10^-10 để tránh số 0
    mask = norm > 1e-10
    res = np.zeros_like(v)

    #tính
    np.divide(v, norm, out=res, where=mask)
    
    return res