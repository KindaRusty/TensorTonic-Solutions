import numpy as np
#This one only take O(1), which is more efficient than the previous O(n*m)
def matrix_transpose(A):
    return np.asarray(A).T