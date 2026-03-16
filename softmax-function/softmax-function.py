import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    #tính e^x
    x_max = np.max(x, axis=-1, keepdims=True)
    #tìm giá trị lớn nhất
    exp_x = np.exp(x - x_max)
    #tính tổng của các giá trị mũ e
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    #hàm tính softmax
    #bình thường thì không có max nhưng mà khi code thì thêm vào để tránh số mũ lớn vd: mũ 1000,1001,1002 thì trừ 1002 còn mũ -2,-1,0 tính sẽ nhanh hơn và ko làm crash
    softmax = exp_x/sum_exp_x
    return softmax