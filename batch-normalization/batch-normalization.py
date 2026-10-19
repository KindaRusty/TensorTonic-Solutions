import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x, dtype = float)
    gamma = np.asarray(gamma, dtype = float) #if it already is in array it will not be copied
    beta = np.asarray(beta, dtype = float)

    if x.ndim == 2:
        axes, shape = (0, ), (1, -1)
    else:
        axes, shape = (0, 2, 3), (1, -1, 1, 1)

    mean = x.mean(axis = axes, keepdims = True)
    var = x.var(axis= axes, keepdims = True)
    x_hat = (x- mean)/ np.sqrt(var + eps)
    return gamma.reshape(shape)*x_hat + beta.reshape(shape)