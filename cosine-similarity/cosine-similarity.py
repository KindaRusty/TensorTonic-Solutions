import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    
    c = np.dot(a,b) #Tính tích vô hướng
    A = np.linalg.norm(a) #Tính độ lớn của vector
    B = np.linalg.norm(b)
    #Tính góc
    cos0 = c/(A*B)
    if A * B == 0:
        return 0.0
    return float(cos0)