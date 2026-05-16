import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    #x là dãy số cần tính
    #q là giá trị phần trăm (tính phân vị là tính giá trị phần trăm các giá trị nhỏ hơn hoặc bằng nó)
    L = np.percentile(x, q, method='linear') 
    x = np.array(L) #chuyển về array
    return x